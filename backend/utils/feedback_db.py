"""
SQLite Feedback Database
Stores user feedback (predicted score vs actual quality + extracted features)
Used to accumulate training data for continuous model retraining.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple


class FeedbackDB:
    """Manage feedback storage in SQLite database."""
    
    def __init__(self, db_path: str = "backend/data/feedback.db"):
        """Initialize feedback database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create feedback table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_hash TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                actual_label INTEGER NOT NULL,
                features TEXT NOT NULL,
                is_quality_gated INTEGER DEFAULT 1,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_count ON feedback(id)")
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback(dataset_hash)")
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hash_label ON feedback(dataset_hash, actual_label)")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        conn.close()
        print(f"✓ Feedback database initialized: {self.db_path}")
    
    def save(
        self,
        dataset_hash: str,
        predicted_score: float,
        actual_label: int,
        features: List[float],
        is_quality_gated: bool = True
    ) -> bool:
        """
        Save feedback record with extracted features.
        
        Args:
            dataset_hash: MD5/SHA hash of the dataset (for deduplication)
            predicted_score: Model's quality prediction (0-100)
            actual_label: User feedback (0=poor, 1=fair, 2=good, 3=excellent)
            features: List of 8 extracted features from MLQualityScorer
            is_quality_gated: Whether feedback passed quality gate (True=accepted, False=rejected)
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            features_json = json.dumps(features) if features else "[]"
            
            cursor.execute("""
                INSERT INTO feedback (
                    dataset_hash,
                    predicted_score,
                    actual_label,
                    features,
                    is_quality_gated
                ) VALUES (?, ?, ?, ?, ?)
            """, (dataset_hash, predicted_score, actual_label, features_json, 1 if is_quality_gated else 0))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"✗ Error saving feedback: {e}")
            return False
    
    def get_all(self) -> List[Tuple[str, int]]:
        """
        Get all feedback records for retraining.
        
        Returns:
            List of tuples: [(features_json, actual_label), ...]
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            rows = cursor.execute(
                "SELECT features, actual_label FROM feedback ORDER BY id ASC"
            ).fetchall()
            
            conn.close()
            return rows
        except Exception as e:
            print(f"✗ Error retrieving feedback: {e}")
            return []
    
    def count(self) -> int:
        """Return total number of feedback records."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            count = cursor.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            
            conn.close()
            return count
        except Exception as e:
            print(f"✗ Error counting feedback: {e}")
            return 0
    
    def clear_feedback(self, keep_last: int = 100) -> int:
        """
        Remove old feedback records, keeping only the last N.
        
        Args:
            keep_last: Number of most recent records to retain
        
        Returns:
            Number of records deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get count
            total = cursor.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            
            if total <= keep_last:
                conn.close()
                return 0
            
            # Delete old records
            to_delete = total - keep_last
            cursor.execute(f"""
                DELETE FROM feedback WHERE id IN (
                    SELECT id FROM feedback ORDER BY id ASC LIMIT {to_delete}
                )
            """)
            
            conn.commit()
            conn.close()
            
            print(f"✓ Cleaned feedback: deleted {to_delete} old records")
            return to_delete
        except Exception as e:
            print(f"✗ Error clearing feedback: {e}")
            return 0
    
    def get_feedback_for_retraining(self, max_samples: Optional[int] = None) -> Tuple[list, list]:
        """
        Get features and labels formatted for model retraining.
        
        Returns:
            Tuple of (features_list, labels_list)
            Features: List of [8-float arrays]
            Labels: List of [0 or 1] where 0-1=bad(0), 2-3=good(1)
        """
        rows = self.get_all()
        
        features_list = []
        labels_list = []
        invalid_count = 0
        
        for features_json, user_label in rows:
            try:
                features = json.loads(features_json)
                if features and len(features) == 8:
                    features_list.append(features)
                    # Convert: 0,1→bad(0), 2,3→good(1)
                    label = 1 if user_label >= 2 else 0
                    labels_list.append(label)
                else:
                    invalid_count += 1
                    print(f"[feedback_db] ⚠️ Skipping feedback with invalid features: {len(features) if features else 0} items (need 8)")
            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"[feedback_db] ⚠️ JSON decode error in feedback: {str(e)[:100]}")
        
        if invalid_count > 0:
            print(f"[feedback_db] ⚠️ WARNING: {invalid_count} feedback records skipped (invalid features)")
        
        if max_samples and len(features_list) > max_samples:
            features_list = features_list[-max_samples:]
            labels_list = labels_list[-max_samples:]
        
        if features_list:
            print(f"[feedback_db] ✓ Using {len(features_list)} valid feedback samples for retraining")
        
        return features_list, labels_list
    
    @staticmethod
    def should_accept_feedback(predicted_score: float, actual_label: int) -> Tuple[bool, str]:
        """
        Quality gate: determine if feedback provides useful signal for retraining.
        Only accept feedback where model prediction significantly contradicts actual label.
        
        Args:
            predicted_score: Model's quality prediction (0-100)
            actual_label: User feedback (0=poor, 1=fair, 2=good, 3=excellent)
        
        Returns:
            Tuple of (should_accept: bool, reason: str)
        
        Logic:
        - ACCEPT: Bad data (label=0) + high prediction (>70) → model overconfident
        - ACCEPT: Good data (label>=2) + low prediction (<50) → model underconfident
        - REJECT: Neutral data (label=1) at any prediction → weak signal
        - REJECT: Model already confident in correct direction → already good
        """
        # Convert label to binary: 0-1=bad(0), 2-3=good(1)
        is_good = actual_label >= 2
        is_bad = actual_label == 0
        is_neutral = actual_label == 1
        
        # Reject neutral labels - low info value
        if is_neutral:
            return False, "Neutral feedback (fair) has low signal value. Use poor/good/excellent."
        
        # Good data + low prediction = useful (model underestimated)
        if is_good and predicted_score < 50:
            return True, "High-value feedback: good data but model predicted low score"
        
        # Bad data + high prediction = useful (model overestimated)
        if is_bad and predicted_score > 70:
            return True, "High-value feedback: poor data but model predicted high score"
        
        # All other cases: model was already correct or close
        return False, "Feedback aligns with model prediction. Choose datasets where model struggles."
    
    def get_feedback_per_dataset(self) -> dict[str, int]:
        """
        Get count of accepted feedback samples per dataset.
        Used to detect if users are drilling the same dataset repeatedly.
        
        Returns:
            Dictionary mapping dataset_hash → count of quality-gated feedback samples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            rows = cursor.execute(
                "SELECT dataset_hash, COUNT(*) FROM feedback WHERE is_quality_gated=1 GROUP BY dataset_hash"
            ).fetchall()
            
            conn.close()
            
            result = {hash_val: count for hash_val, count in rows}
            return result
        except Exception as e:
            print(f"✗ Error getting per-dataset feedback counts: {e}")
            return {}
