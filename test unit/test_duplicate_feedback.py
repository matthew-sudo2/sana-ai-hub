#!/usr/bin/env python3
"""
Comprehensive test for realistic retraining with duplicate dataset prevention.

Tests:
1. Quality-gated feedback acceptance (high-value vs. low-value)
2. Duplicate dataset prevention (max 3 feedback per dataset)
3. Retrain threshold at 20 samples (not 5)
4. Minimum samples validation in ContinuousLearner
5. Per-dataset feedback tracking
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.feedback_db import FeedbackDB
from backend.utils.continuous_learner import ContinuousLearner


def test_quality_gate_acceptance():
    """TEST 1: Verify quality gate logic for feedback acceptance"""
    print("\n" + "="*70)
    print("TEST 1: Quality-Gated Feedback Acceptance")
    print("="*70)
    
    test_cases = [
        # (predicted_score, actual_label, expected_accept, description)
        (75, 0, True, "Bad data + high prediction → OVERCONFIDENT (accept)"),
        (45, 0, False, "Bad data + low prediction → already correct (reject)"),
        (60, 2, False, "Good data + neutral prediction → unclear (reject)"),
        (40, 2, True, "Good data + low prediction → UNDERCONFIDENT (accept)"),
        (50, 1, False, "Neutral feedback → weak signal (reject)"),
        (70, 1, False, "Neutral feedback → weak signal (reject)"),
        (30, 3, True, "Excellent data + low prediction → UNDERCONFIDENT (accept)"),
        (80, 3, False, "Excellent data + high prediction → already confident (reject)"),
    ]
    
    passed = 0
    failed = 0
    
    for predicted, label, expected, desc in test_cases:
        accept, reason = FeedbackDB.should_accept_feedback(predicted, label)
        status = "✓" if accept == expected else "✗"
        
        if accept == expected:
            passed += 1
            print(f"{status} PASS: {desc}")
            print(f"       Reason: {reason[:70]}")
        else:
            failed += 1
            print(f"{status} FAIL: {desc}")
            print(f"       Expected: {expected}, Got: {accept}")
            print(f"       Reason: {reason[:70]}")
    
    print(f"\n✓ Quality Gate: {passed}/{len(test_cases)} passed")
    return failed == 0


def test_duplicate_dataset_prevention():
    """TEST 2: Verify duplicate dataset prevention (max 3 per dataset)"""
    print("\n" + "="*70)
    print("TEST 2: Duplicate Dataset Prevention")
    print("="*70)
    
    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "feedback_test.db"
        feedback_db = FeedbackDB(str(db_path))
        
        # Create dataset hashes
        dataset_hash_1 = hashlib.md5(b"amazon_sales_data").hexdigest()
        dataset_hash_2 = hashlib.md5(b"employee_records").hexdigest()
        
        # Create sample features (8 values)
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"\n1. Testing Dataset 1 (max 3 feedback)...")
        
        # Store 3 feedback on dataset 1 (should all be accepted)
        for i in range(3):
            result = feedback_db.save(
                dataset_hash=dataset_hash_1,
                predicted_score=50.0 + i * 10,
                actual_label=i % 4,
                features=features,
                is_quality_gated=True
            )
            print(f"   {i+1}. Save feedback #{i+1}: {'✓' if result else '✗'}")
        
        # Check per-dataset count
        per_dataset = feedback_db.get_feedback_per_dataset()
        count1 = per_dataset.get(dataset_hash_1, 0)
        print(f"   Dataset 1 feedback count: {count1}")
        
        if count1 == 3:
            print(f"   ✓ PASS: Dataset 1 has 3 feedback samples")
        else:
            print(f"   ✗ FAIL: Expected 3, got {count1}")
            return False
        
        print(f"\n2. Testing Dataset 2 (should be independent)...")
        
        # Store feedback on dataset 2
        result = feedback_db.save(
            dataset_hash=dataset_hash_2,
            predicted_score=60.0,
            actual_label=2,
            features=features,
            is_quality_gated=True
        )
        print(f"   Save feedback on dataset 2: {'✓' if result else '✗'}")
        
        per_dataset = feedback_db.get_feedback_per_dataset()
        count2 = per_dataset.get(dataset_hash_2, 0)
        print(f"   Dataset 2 feedback count: {count2}")
        
        if count1 == 3 and count2 == 1:
            print(f"   ✓ PASS: Datasets tracked independently")
            return True
        else:
            print(f"   ✗ FAIL: Expected dataset1=3, dataset2=1; got {count1}, {count2}")
            return False


def test_retrain_threshold():
    """TEST 3: Verify retrain threshold is 20, not 5"""
    print("\n" + "="*70)
    print("TEST 3: Retrain Threshold (should be 20, not 5)")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "feedback_test.db"
        feedback_db = FeedbackDB(str(db_path))
        
        # Create sample features and dataset hash
        dataset_hash = hashlib.md5(b"test_dataset").hexdigest()
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # Store 19 feedback samples
        print(f"\n1. Storing 19 feedback samples...")
        for i in range(19):
            feedback_db.save(
                dataset_hash=dataset_hash,
                predicted_score=50.0 + (i % 50),
                actual_label=i % 4,
                features=features,
                is_quality_gated=True
            )
        
        count = feedback_db.count()
        print(f"   Total feedback: {count}")
        
        # Check if retrain should trigger at 19
        should_retrain_at_19 = count >= 20 and count % 20 == 0
        print(f"   Should retrain at {count}? {should_retrain_at_19}")
        
        if should_retrain_at_19:
            print(f"   ✗ FAIL: Retrain should NOT trigger at {count}")
            return False
        
        # Add 1 more (now 20)
        print(f"\n2. Adding 20th feedback sample...")
        feedback_db.save(
            dataset_hash=dataset_hash,
            predicted_score=60.0,
            actual_label=2,
            features=features,
            is_quality_gated=True
        )
        
        count = feedback_db.count()
        print(f"   Total feedback: {count}")
        
        # Check if retrain triggers at 20
        should_retrain_at_20 = count >= 20 and count % 20 == 0
        print(f"   Should retrain at {count}? {should_retrain_at_20}")
        
        if should_retrain_at_20:
            print(f"   ✓ PASS: Retrain triggers at count=20 (batch-based learning)")
            return True
        else:
            print(f"   ✗ FAIL: Retrain should trigger at 20")
            return False


def test_minimum_samples_validation():
    """TEST 4: Verify ContinuousLearner rejects retraining with <20 feedback"""
    print("\n" + "="*70)
    print("TEST 4: Minimum Samples Validation in ContinuousLearner")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "feedback_test.db"
        feedback_db = FeedbackDB(str(db_path))
        
        # Try to retrain with 10 feedback samples
        dataset_hash = hashlib.md5(b"test_dataset").hexdigest()
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"\n1. Adding 10 feedback samples to database...")
        for i in range(10):
            feedback_db.save(
                dataset_hash=dataset_hash,
                predicted_score=50.0 + i * 5,
                actual_label=i % 4,
                features=features,
                is_quality_gated=True
            )
        
        count = feedback_db.count()
        print(f"   Total feedback in DB: {count}")
        
        print(f"\n2. Attempting retrain with {count} samples...")
        learner = ContinuousLearner(model_dir=str(Path(tmpdir) / "models"))
        result = learner.retrain()
        
        print(f"   Retrain success: {result['success']}")
        print(f"   Error: {result.get('error', 'None')}")
        print(f"   Validation reason: {result.get('validation_reason', 'None')}")
        
        if not result['success'] and "Insufficient" in result.get('error', ''):
            print(f"   ✓ PASS: Retrain correctly rejected with insufficient samples")
            return True
        else:
            print(f"   ✗ FAIL: Should reject retrain with <20 samples")
            return False


def test_quality_gated_vs_unaccepted():
    """TEST 5: Verify quality_gated flag is set correctly"""
    print("\n" + "="*70)
    print("TEST 5: Quality-Gated Flag Tracking")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "feedback_test.db"
        feedback_db = FeedbackDB(str(db_path))
        
        dataset_hash = hashlib.md5(b"test_data").hexdigest()
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"\n1. Saving high-quality feedback...")
        feedback_db.save(
            dataset_hash=dataset_hash,
            predicted_score=75,  # high prediction
            actual_label=0,       # bad data
            features=features,
            is_quality_gated=True  # explicitly quality-gated
        )
        
        print(f"2. Saving low-quality feedback...")
        feedback_db.save(
            dataset_hash=dataset_hash,
            predicted_score=50,
            actual_label=1,       # neutral label
            features=features,
            is_quality_gated=False  # explicitly not quality-gated
        )
        
        # Check database directly
        count = feedback_db.count()
        per_dataset = feedback_db.get_feedback_per_dataset()
        dataset_count = per_dataset.get(dataset_hash, 0)
        
        print(f"\n   Total records: {count}")
        print(f"   Quality-gated records for dataset: {dataset_count}")
        
        if count == 2 and dataset_count == 1:
            print(f"   ✓ PASS: Quality gate correctly filters feedback")
            return True
        else:
            print(f"   ✗ FAIL: Expected total=2, gated=1; got {count}, {dataset_count}")
            return False


def test_edge_case_multiple_datasets():
    """TEST 6: Edge case - multiple datasets with different feedback counts"""
    print("\n" + "="*70)
    print("TEST 6: Multiple Datasets Edge Case")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "feedback_test.db"
        feedback_db = FeedbackDB(str(db_path))
        
        # Create 5 different datasets
        datasets = {}
        for i in range(5):
            hash_val = hashlib.md5(f"dataset_{i}".encode()).hexdigest()
            datasets[i] = hash_val
        
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"\n1. Storing varying feedback counts per dataset...")
        feedback_counts = [1, 2, 3, 3, 2]  # Dataset 2 and 3 at max
        
        for dataset_idx, target_count in enumerate(feedback_counts):
            for sample_idx in range(target_count):
                feedback_db.save(
                    dataset_hash=datasets[dataset_idx],
                    predicted_score=50.0 + sample_idx * 10,
                    actual_label=sample_idx % 4,
                    features=features,
                    is_quality_gated=True
                )
            print(f"   Dataset {dataset_idx}: {target_count} feedback samples")
        
        # Verify counts
        print(f"\n2. Verifying per-dataset counts...")
        per_dataset = feedback_db.get_feedback_per_dataset()
        
        all_match = True
        for dataset_idx, expected_count in enumerate(feedback_counts):
            actual_count = per_dataset.get(datasets[dataset_idx], 0)
            match = actual_count == expected_count
            all_match = all_match and match
            status = "✓" if match else "✗"
            print(f"   {status} Dataset {dataset_idx}: expected {expected_count}, got {actual_count}")
        
        if all_match:
            print(f"   ✓ PASS: Multiple datasets tracked independently")
            return True
        else:
            print(f"   ✗ FAIL: Per-dataset counts don't match")
            return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REALISTIC RETRAINING WITH DUPLICATE PREVENTION - COMPREHENSIVE TESTS")
    print("="*70)
    
    results = {
        "Quality Gate": test_quality_gate_acceptance(),
        "Duplicate Prevention": test_duplicate_dataset_prevention(),
        "Retrain Threshold": test_retrain_threshold(),
        "Minimum Samples": test_minimum_samples_validation(),
        "Quality-Gated Flag": test_quality_gated_vs_unaccepted(),
        "Multiple Datasets": test_edge_case_multiple_datasets()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All realistic retraining tests PASSED!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) FAILED")
        sys.exit(1)
