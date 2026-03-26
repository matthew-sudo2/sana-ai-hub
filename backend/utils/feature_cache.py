"""
Feature Cache: Store extracted dataset features for feedback
Saves 8 quality features extracted by MLQualityScorer for each run.
Used when user gives feedback - features are already computed.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional


class FeatureCache:
    """Store and retrieve extracted features for datasets."""
    
    @staticmethod
    def get_dataset_hash(csv_path: str) -> str:
        """
        Generate MD5 hash of dataset for deduplication.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            MD5 hash of file contents
        """
        try:
            with open(csv_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"✗ Error hashing dataset: {e}")
            return ""
    
    @staticmethod
    def save_features(
        run_dir: str,
        features: List[float],
        dataset_hash: str = ""
    ) -> bool:
        """
        Save extracted features to run directory as JSON.
        
        Args:
            run_dir: Path to run directory (backend/runs/{run_id}/)
            features: List of 8 features from MLQualityScorer.extract_features()
            dataset_hash: MD5 hash of the dataset
        
        Returns:
            True if successful
        """
        try:
            run_path = Path(run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            
            features_file = run_path / "features.json"
            
            data = {
                "features": features,
                "dataset_hash": dataset_hash,
                "feature_names": [
                    "missing_ratio",
                    "duplicate_ratio",
                    "numeric_ratio",
                    "constant_cols",
                    "norm_variance",
                    "skewness",
                    "cardinality_ratio",
                    "mean_kurtosis"
                ]
            }
            
            with open(features_file, "w") as f:
                json.dump(data, f)
            
            print(f"[Cache] ✓ Features saved to {features_file}")
            return True
        except Exception as e:
            print(f"[Cache] ✗ Error saving features: {e}")
            return False
    
    @staticmethod
    def load_features(run_dir: str) -> Optional[List[float]]:
        """
        Load features from run directory.
        
        Args:
            run_dir: Path to run directory
        
        Returns:
            List of 8 features, or None if not found
        """
        try:
            features_file = Path(run_dir) / "features.json"
            
            if not features_file.exists():
                return None
            
            with open(features_file, "r") as f:
                data = json.load(f)
            
            return data.get("features", None)
        except Exception as e:
            print(f"[Cache] ✗ Error loading features: {e}")
            return None
    
    @staticmethod
    def save_feedback_metadata(
        run_dir: str,
        predicted_score: float,
        actual_label: Optional[int] = None
    ) -> bool:
        """
        Save feedback metadata to run directory.
        
        Args:
            run_dir: Path to run directory
            predicted_score: Model's prediction (0-100)
            actual_label: User feedback label (0-3, optional)
        
        Returns:
            True if successful
        """
        try:
            run_path = Path(run_dir)
            feedback_file = run_path / "feedback_metadata.json"
            
            data = {
                "predicted_score": predicted_score,
                "actual_label": actual_label,
                "feedback_given": actual_label is not None
            }
            
            with open(feedback_file, "w") as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            print(f"[Cache] ✗ Error saving feedback metadata: {e}")
            return False
    
    @staticmethod
    def load_feedback_metadata(run_dir: str) -> Optional[dict]:
        """
        Load feedback metadata from run directory.
        
        Args:
            run_dir: Path to run directory
        
        Returns:
            Dictionary with feedback metadata, or None
        """
        try:
            feedback_file = Path(run_dir) / "feedback_metadata.json"
            
            if not feedback_file.exists():
                return None
            
            with open(feedback_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache] ✗ Error loading feedback metadata: {e}")
            return None
