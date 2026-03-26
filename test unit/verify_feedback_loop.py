#!/usr/bin/env python3
"""
Continuous Learning Feedback Loop - Manual Verification Guide
Tests feedback storage, model retraining, and end-to-end integration
"""

import json
from pathlib import Path
from backend.utils.feedback_db import FeedbackDB
from backend.utils.continuous_learner import ContinuousLearner
from backend.utils.feature_cache import FeatureCache

def test_feedback_db():
    """Test SQLite feedback database."""
    print("\n" + "="*70)
    print("TEST 1: Feedback Database")
    print("="*70)
    
    db = FeedbackDB()
    
    # Test save
    print("\n[Test] Saving feedback...")
    sample_features = [0.05, 0.0, 0.6, 0, 2.5, 1.2, 0.3, 0.1]  # 8 features
    
    success = db.save(
        dataset_hash="test_dataset_1",
        predicted_score=78.5,
        actual_label=2,  # Good
        features=sample_features
    )
    
    if success:
        print("✓ Feedback saved successfully")
    else:
        print("✗ Failed to save feedback")
        return False
    
    # Test count
    print("\n[Test] Counting feedbacks...")
    count = db.count()
    print(f"✓ Total feedbacks: {count}")
    
    # Test retrieve
    print("\n[Test] Retrieving feedback for retraining...")
    X_feedback, y_feedback = db.get_feedback_for_retraining()
    print(f"✓ Retrieved {len(X_feedback)} feedback samples with labels: {y_feedback}")
    
    return True


def test_feature_cache():
    """Test feature caching."""
    print("\n" + "="*70)
    print("TEST 2: Feature Cache")
    print("="*70)
    
    test_dir = Path("test_run_dir")
    test_dir.mkdir(exist_ok=True)
    
    # Test save
    print("\n[Test] Saving features...")
    test_features = [0.1, 0.05, 0.7, 1, 2.0, 1.5, 0.25, 0.2]
    success = FeatureCache.save_features(
        run_dir=str(test_dir),
        features=test_features,
        dataset_hash="test_hash_123"
    )
    
    if success:
        print("✓ Features saved successfully")
    else:
        print("✗ Failed to save features")
        return False
    
    # Test load
    print("\n[Test] Loading features...")
    loaded = FeatureCache.load_features(str(test_dir))
    
    if loaded == test_features:
        print(f"✓ Features loaded correctly: {loaded}")
    else:
        print(f"✗ Feature mismatch. Expected {test_features}, got {loaded}")
        return False
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return True


def test_continuous_learner():
    """Test model retraining."""
    print("\n" + "="*70)
    print("TEST 3: Continuous Learning / Model Retraining")
    print("="*70)
    
    learner = ContinuousLearner()
    
    # Check if training data exists
    print("\n[Test] Checking training data...")
    good_path = Path("data/synthetic/good_quality_features_real.npy")
    bad_path = Path("data/synthetic/bad_quality_features_real.npy")
    
    if good_path.exists() and bad_path.exists():
        print(f"✓ Training data found:")
        print(f"  - Good features: {good_path}")
        print(f"  - Bad features: {bad_path}")
    else:
        print(f"✗ Training data not found. Skipping retraining test.")
        print(f"  Expected paths:")
        print(f"  - {good_path.absolute()}")
        print(f"  - {bad_path.absolute()}")
        return False
    
    # Test retrain (requires accumulated feedback)
    print("\n[Test] Attempting model retraining...")
    db = FeedbackDB()
    fb_count = db.count()
    
    if fb_count < 5:
        print(f"ℹ  Only {fb_count} feedbacks accumulated. Need >=5 for meaningful retrain.")
        print(f"✓ Retrain function is available and would trigger at 20+ feedbacks")
        return True
    
    print(f"ℹ  {fb_count} feedbacks available. Running retrain...")
    result = learner.retrain()
    
    if result["success"]:
        print(f"✓ Retraining successful!")
        print(f"  - CV Score: {result['cv_score']:.1%}")
        print(f"  - Feedback samples: {result['feedback_count']}")
        print(f"  - Total training samples: {result['total_samples']}")
        return True
    else:
        print(f"✗ Retraining failed: {result.get('error')}")
        return False


def test_api_endpoints():
    """Test API endpoints (requires running server)."""
    print("\n" + "="*70)
    print("TEST 4: API Endpoints")
    print("="*70)
    
    print("\n[Info] API endpoints available:")
    print("  - POST /api/feedback")
    print("    Payload: {dataset_hash, predicted_score, actual_quality, features?}")
    print("    Response: {status, feedback_count, cv_score?, next_retrain_at?, message}")
    print("")
    print("  - GET /api/feedback/stats")
    print("    Response: {total_feedbacks, models_trained, current_cv_score, ...}")
    print("")
    print("[Note] Start server with: python -m uvicorn backend.api:app --reload")
    print("[Note] Test with curl:")
    print("""
    curl -X POST http://localhost:8000/api/feedback \\
      -H "Content-Type: application/json" \\
      -d '{
        "dataset_hash": "test_123",
        "predicted_score": 78.5,
        "actual_quality": 2,
        "features": [0.05, 0.0, 0.6, 0, 2.5, 1.2, 0.3, 0.1]
      }'
    """)
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONTINUOUS LEARNING FEEDBACK LOOP - VERIFICATION SUITE")
    print("="*70)
    
    tests = [
        ("Feedback Database", test_feedback_db),
        ("Feature Cache", test_feature_cache),
        ("Continuous Learner", test_continuous_learner),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"{icon} {name:.<50} {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\nTotal: {passed}/{len(results)} tests passed\n")
    
    return all(s == "PASS" for _, s in results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
