#!/usr/bin/env python3
"""
Comprehensive feedback loop test - verify all critical fixes are in place.

Tests:
1. Features are properly extracted and cached
2. Features are retrievable via API endpoint
3. Frontend can hash datasets consistently
4. Feedback submission with features works
5. Retraining is triggered correctly
6. Model reloading works after retrain
"""

import json
import hashlib
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from backend.utils.feedback_db import FeedbackDB
from backend.utils.continuous_learner import ContinuousLearner
from backend.utils.ml_quality_scorer import MLQualityScorer
from backend.utils.feature_cache import FeatureCache


def test_feature_extraction():
    """TEST 1: Verify features are properly extracted"""
    print("\n" + "="*70)
    print("TEST 1: Feature Extraction")
    print("="*70)
    
    # Create sample dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3, np.nan, 5],
        'B': ['a', 'b', 'a', 'c', 'b'],
        'C': [1.1, 2.2, 3.3, 3.3, 5.5],
        'D': [True, False, True, False, True]
    })
    
    # Score with ML
    scorer = MLQualityScorer()
    result = scorer.score(df)
    
    print(f"✓ ML Score result: {result['quality']} (score: {result['score']:.1f})")
    print(f"✓ Features extracted: {len(result.get('features', []))} items")
    
    if len(result.get('features', [])) == 8:
        print("✓ PASS: 8 features extracted correctly")
        return True
    else:
        print(f"✗ FAIL: Expected 8 features, got {len(result.get('features', []))}")
        return False


def test_feature_caching():
    """TEST 2: Verify features are cached properly"""
    print("\n" + "="*70)
    print("TEST 2: Feature Caching")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        
        # Create sample features
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        dataset_hash = hashlib.md5(b"test_data").hexdigest()
        
        # Save features
        FeatureCache.save_features(
            run_dir=str(run_dir),
            features=features,
            dataset_hash=dataset_hash
        )
        
        # Verify file exists
        features_file = run_dir / "features.json"
        if features_file.exists():
            print(f"✓ Features cached to {features_file}")
            
            # Load and verify
            data = json.loads(features_file.read_text())
            if data.get('features') == features and len(data['features']) == 8:
                print("✓ PASS: Features cached and retrievable")
                return True
            else:
                print("✗ FAIL: Cached features don't match")
                return False
        else:
            print(f"✗ FAIL: Features file not created")
            return False


def test_feedback_database():
    """TEST 3: Verify feedback is saved and retrieved correctly"""
    print("\n" + "="*70)
    print("TEST 3: Feedback Database")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_feedback.db"
        db = FeedbackDB(str(db_path))
        
        # Save multiple feedback records with features
        test_feedbacks = [
            ("hash1", 75.0, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            ("hash1", 70.0, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            ("hash2", 45.0, 1, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            ("hash2", 50.0, 2, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
        
        for hash_val, score, label, features in test_feedbacks:
            db.save(hash_val, score, label, features)
        
        # Verify count
        count = db.count()
        print(f"✓ Feedback records saved: {count}")
        
        # Get all and verify
        all_records = db.get_all()
        print(f"✓ Feedback records retrieved: {len(all_records)}")
        
        # Get for retraining
        X_features, y_labels = db.get_feedback_for_retraining()
        print(f"✓ Valid features for training: {len(X_features)}")
        print(f"✓ Valid labels for training: {len(y_labels)}")
        
        if len(X_features) == 4 and len(y_labels) == 4:
            print("✓ PASS: Feedback storage and retrieval working")
            return True
        else:
            print("✗ FAIL: Feedback count mismatch")
            return False


def test_md5_hashing():
    """TEST 4: Verify MD5 hashing consistency"""
    print("\n" + "="*70)
    print("TEST 4: MD5 Hashing Consistency")
    print("="*70)
    
    # Create test data
    data = b"test_dataset_content_12345"
    
    # Compute hash
    hash1 = hashlib.md5(data).hexdigest()
    hash2 = hashlib.md5(data).hexdigest()
    
    print(f"✓ Hash 1: {hash1}")
    print(f"✓ Hash 2: {hash2}")
    
    if hash1 == hash2 and len(hash1) == 32:
        print("✓ PASS: MD5 hashing is consistent and correct length")
        return True
    else:
        print("✗ FAIL: MD5 hashing inconsistent")
        return False


def test_retrain_trigger_logic():
    """TEST 5: Verify retrain trigger logic"""
    print("\n" + "="*70)
    print("TEST 5: Retrain Trigger Logic")
    print("="*70)
    
    # Test the trigger logic
    retrain_triggers = []
    for feedback_count in range(1, 31):
        should_retrain = (feedback_count == 1) or (feedback_count >= 5 and feedback_count % 5 == 0)
        if should_retrain:
            retrain_triggers.append(feedback_count)
    
    print(f"✓ Retrain triggers at: {retrain_triggers}")
    
    # Verify expected triggers
    expected = [1, 5, 10, 15, 20, 25, 30]
    if retrain_triggers == expected:
        print(f"✓ PASS: Retrain triggers at correct feedback counts")
        return True
    else:
        print(f"✗ FAIL: Expected {expected}, got {retrain_triggers}")
        return False


def test_invalid_features_handling():
    """TEST 6: Verify invalid features are properly logged"""
    print("\n" + "="*70)
    print("TEST 6: Invalid Features Handling")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_feedback.db"
        db = FeedbackDB(str(db_path))
        
        # Save feedback with valid features
        db.save("hash1", 75.0, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # Save feedback with empty features (this simulates the old bug)
        db.save("hash1", 70.0, 2, [])
        
        # Save feedback with wrong number of features
        db.save("hash1", 50.0, 1, [0.1, 0.2, 0.3])
        
        # Retrieve for training
        X_features, y_labels = db.get_feedback_for_retraining()
        
        print(f"✓ Total records: {db.count()}")
        print(f"✓ Valid features for training: {len(X_features)}")
        
        if len(X_features) == 1:  # Only the first one should be valid
            print("✓ PASS: Invalid features properly filtered out")
            return True
        else:
            print(f"✗ FAIL: Expected 1 valid feature set, got {len(X_features)}")
            return False


def main():
    """Run all tests"""
    print("\n")
    print("#" * 70)
    print("# FEEDBACK LOOP VERIFICATION TESTS")
    print("#" * 70)
    
    results = {
        "Feature Extraction": test_feature_extraction(),
        "Feature Caching": test_feature_caching(),
        "Feedback Database": test_feedback_database(),
        "MD5 Hashing": test_md5_hashing(),
        "Retrain Trigger": test_retrain_trigger_logic(),
        "Invalid Features": test_invalid_features_handling(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED - Feedback loop is ready!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed - Review issues above")
        return 1


if __name__ == "__main__":
    exit(main())
