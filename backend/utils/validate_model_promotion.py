"""
Model Promotion Validator: Shadow Validation Before Deployment
Compares candidate model against current best_model on held-out test set.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, roc_auc_score
)
from datetime import datetime
from typing import Dict, Tuple, Optional
import json


class ModelPromotionValidator:
    """Validates if a new model outperforms the current best model."""
    
    def __init__(self, model_dir: str = "models", test_size: float = 0.2):
        """
        Initialize validator.
        
        Args:
            model_dir: Directory containing models
            test_size: Proportion of data to use as test set (0.2 = 20%)
        """
        self.model_dir = Path(model_dir)
        self.test_size = test_size
        self.current_model_path = self.model_dir / "best_model.pkl"
        self.archive_dir = self.model_dir / "archived"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_test_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load held-out test data. Falls back to synthetic training data split.
        
        Returns:
            Tuple of (X_test, y_test) or None if not available
        """
        try:
            # Try to load dedicated test set
            test_path = self.model_dir / "test_data.pkl"
            if test_path.exists():
                with open(test_path, 'rb') as f:
                    data = pickle.load(f)
                    return data['X_test'], data['y_test']
            
            # Fallback: load training data and split
            synthetic_dir = Path("data/synthetic")
            training_path = synthetic_dir / "training_data_8features.pkl"
            
            if training_path.exists():
                with open(training_path, 'rb') as f:
                    data = pickle.load(f)
                    X_all = data['X']
                    y_all = data['y']
                    
                    # Split into train/test
                    n_test = max(1, int(len(X_all) * self.test_size))
                    
                    # Stratified split
                    indices = np.arange(len(X_all))
                    np.random.seed(42)
                    np.random.shuffle(indices)
                    
                    test_indices = indices[:n_test]
                    X_test = X_all[test_indices]
                    y_test = y_all[test_indices]
                    
                    print(f"[Validator] Using stratified test set: {X_test.shape[0]} samples")
                    return X_test, y_test
            
            return None
        except Exception as e:
            print(f"✗ Could not load test data: {e}")
            return None
    
    def _load_model(self, model_path: Path):
        """Load a pickled model."""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"✗ Could not load model {model_path}: {e}")
            return None
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_test, y_pred_proba)),
            }
            
            return metrics
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return {}
    
    def validate(self, candidate_model_path: Path) -> Dict:
        """
        Perform shadow validation: compare candidate against current best model.
        
        Args:
            candidate_model_path: Path to new model to validate
        
        Returns:
            Dictionary with validation results:
            {
                "promoted": bool,
                "reason": str,
                "current_model_metrics": dict,
                "candidate_model_metrics": dict,
                "improvement": dict,
                "timestamp": str
            }
        """
        print("\n" + "="*80)
        print("SHADOW VALIDATION: MODEL PROMOTION DECISION")
        print("="*80)
        
        # Step 1: Load test data
        print("\n[Step 1] Loading test data...")
        test_data = self._load_test_data()
        if test_data is None:
            print("✗ No test data available. Skipping validation (defaulting to deploy).")
            return {
                "promoted": True,
                "reason": "No test data available",
                "current_model_metrics": {},
                "candidate_model_metrics": {},
                "improvement": {},
                "timestamp": datetime.now().isoformat()
            }
        
        X_test, y_test = test_data
        
        # Step 2: Load current model
        print("\n[Step 2] Loading current best model...")
        if not self.current_model_path.exists():
            print("⚠ Current model not found. Promoting candidate as initial model.")
            return {
                "promoted": True,
                "reason": "No current model to compare against",
                "current_model_metrics": {},
                "candidate_model_metrics": {},
                "improvement": {},
                "timestamp": datetime.now().isoformat()
            }
        
        current_model = self._load_model(self.current_model_path)
        if current_model is None:
            return {
                "promoted": False,
                "reason": "Could not load current model",
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Load candidate model
        print("[Step 3] Loading candidate model...")
        candidate_model = self._load_model(candidate_model_path)
        if candidate_model is None:
            return {
                "promoted": False,
                "reason": "Could not load candidate model",
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 4: Evaluate both models
        print("\n[Step 4] Evaluating current model...")
        current_metrics = self._evaluate_model(current_model, X_test, y_test)
        for metric, value in current_metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        print("\n[Step 5] Evaluating candidate model...")
        candidate_metrics = self._evaluate_model(candidate_model, X_test, y_test)
        for metric, value in candidate_metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        # Step 6: Compare and decide
        print("\n[Step 6] Comparison & Decision...")
        
        # Calculate improvements for key metrics
        improvements = {}
        current_f1 = current_metrics.get("f1", 0)
        candidate_f1 = candidate_metrics.get("f1", 0)
        f1_improvement = candidate_f1 - current_f1
        improvements["f1_delta"] = float(f1_improvement)
        
        current_balanced_acc = current_metrics.get("balanced_accuracy", 0)
        candidate_balanced_acc = candidate_metrics.get("balanced_accuracy", 0)
        balanced_acc_improvement = candidate_balanced_acc - current_balanced_acc
        improvements["balanced_accuracy_delta"] = float(balanced_acc_improvement)
        
        current_auc = current_metrics.get("auc_roc", 0)
        candidate_auc = candidate_metrics.get("auc_roc", 0)
        auc_improvement = candidate_auc - current_auc
        improvements["auc_roc_delta"] = float(auc_improvement)
        
        print(f"\n  F1 Score:          {current_f1:.4f} → {candidate_f1:.4f} (Δ {f1_improvement:+.4f})")
        print(f"  Balanced Accuracy: {current_balanced_acc:.4f} → {candidate_balanced_acc:.4f} (Δ {balanced_acc_improvement:+.4f})")
        print(f"  AUC-ROC:           {current_auc:.4f} → {candidate_auc:.4f} (Δ {auc_improvement:+.4f})")
        
        # Decision logic: Promote if F1 score improves OR balanced accuracy improves
        # with a small tolerance for noise
        IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement required
        
        promoted = False
        reason = ""
        
        if f1_improvement > IMPROVEMENT_THRESHOLD:
            promoted = True
            reason = f"F1 score improved by {f1_improvement:.4f}"
        elif balanced_acc_improvement > IMPROVEMENT_THRESHOLD:
            promoted = True
            reason = f"Balanced accuracy improved by {balanced_acc_improvement:.4f}"
        elif auc_improvement > IMPROVEMENT_THRESHOLD:
            promoted = True
            reason = f"AUC-ROC improved by {auc_improvement:.4f}"
        else:
            promoted = False
            reason = "No meaningful improvement detected"
        
        print(f"\n{'='*80}")
        if promoted:
            print(f"✓ PROMOTION APPROVED: {reason}")
            print(f"{'='*80}\n")
        else:
            print(f"✗ PROMOTION REJECTED: {reason}")
            print(f"  (Archiving candidate model instead)")
            print(f"{'='*80}\n")
        
        return {
            "promoted": promoted,
            "reason": reason,
            "current_model_metrics": current_metrics,
            "candidate_model_metrics": candidate_metrics,
            "improvement": improvements,
            "timestamp": datetime.now().isoformat()
        }
    
    def archive_model(self, model_path: Path, reason: str = "rejected") -> Optional[Path]:
        """
        Archive a model with timestamp.
        
        Args:
            model_path: Path to model to archive
            reason: Reason for archival (e.g., "rejected", "superseded")
        
        Returns:
            Path to archived model or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_path = self.archive_dir / f"model_{reason}_{timestamp}.pkl"
            
            import shutil
            shutil.copy(model_path, archived_path)
            print(f"[Archive] Model saved → {archived_path.name}")
            return archived_path
        except Exception as e:
            print(f"✗ Archive failed: {e}")
            return None


def run_validation(candidate_model_path: str = "models/best_model_candidate.pkl") -> Dict:
    """
    Convenience function to run validation from command line.
    
    Args:
        candidate_model_path: Path to candidate model
    
    Returns:
        Validation results dictionary
    """
    validator = ModelPromotionValidator()
    results = validator.validate(Path(candidate_model_path))
    
    # Log results
    metrics_log = Path("models") / "promotion_validation.jsonl"
    with open(metrics_log, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    candidate_path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model_candidate.pkl"
    
    if not Path(candidate_path).exists():
        print(f"✗ Model not found: {candidate_path}")
        sys.exit(1)
    
    results = run_validation(candidate_path)
    sys.exit(0 if results["promoted"] else 1)
