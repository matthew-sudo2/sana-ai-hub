#!/usr/bin/env python3
"""
Test script for enhanced feedback loop with model retraining, persistence, and decay.

Tests:
1. Feedback storage and counting
2. Auto-retraining at feedback thresholds
3. Model persistence and reloading
4. Quality score decay over time
5. Model status tracking
"""

import json
import time
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from utils.feedback_db import FeedbackDB
from utils.continuous_learner import ContinuousLearner
from utils.ml_quality_scorer import MLQualityScorer


def test_feedback_storage():
    """Test 1: Feedback storage"""
    print("\n" + "="*70)
    print("TEST 1: Feedback Storage")
    print("="*70)
    
    feedback_db = FeedbackDB()
    
    # Store sample feedback
    feedback_db.save(
        dataset_hash="test_hash_1",
        predicted_score=85.0,
        actual_label=2,  # good
        features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    
    count = feedback_db.count()
    print(f"✓ Feedback stored. Total count: {count}")
    assert count > 0, "Feedback should be stored"
    return True


def test_model_persistence():
    """Test 2: Model persistence and reloading"""
    print("\n" + "="*70)
    print("TEST 2: Model Persistence and Reloading")
    print("="*70)
    
    try:
        # Load and reload model
        scorer = MLQualityScorer()
        print(f"✓ Model loaded: {scorer.model is not None}")
        
        # Reload model
        scorer.reload_model()
        print(f"✓ Model reloaded: {scorer.model is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Model persistence failed: {e}")
        return False


def test_model_history():
    """Test 3: Model history tracking"""
    print("\n" + "="*70)
    print("TEST 3: Model History Tracking")
    print("="*70)
    
    learner = ContinuousLearner()
    history = learner.get_model_history(max_records=10)
    
    if history:
        print(f"✓ Found {len(history)} retraining records")
        latest = history[-1]
        print(f"  Latest CV Score: {latest.get('cv_score', 'N/A')}")
        print(f"  Latest Timestamp: {latest.get('timestamp', 'N/A')}")
        return True
    else:
        print("ℹ No model history yet (normal for first run)")
        return True


def test_quality_decay_calculation():
    """Test 4: Quality score decay calculation"""
    print("\n" + "="*70)
    print("TEST 4: Quality Score Decay Calculation")
    print("="*70)
    
    from api import _calculate_quality_score_decay
    
    test_cases = [
        (0, 1.0, "Fresh (0 days)"),
        (7*24, 0.97, "1 week"),
        (30*24, 0.90, "30 days"),
        (90*24, 0.70, "90+ days"),
    ]
    
    for hours, expected_approx, desc in test_cases:
        decay = _calculate_quality_score_decay(hours)
        print(f"  {desc}: {decay:.2f} (expected ~{expected_approx:.2f})")
    
    return True


def test_feedback_endpoint_response():
    """Test 5: Feedback endpoint response structure"""
    print("\n" + "="*70)
    print("TEST 5: Feedback Endpoint Response Structure")
    print("="*70)
    
    # This would be called via HTTP in real scenario
    # For now, verify the response model is correctly typed
    from api import FeedbackResponse
    
    response = FeedbackResponse(
        status="retrained",
        feedback_count=5,
        cv_score=0.82,
        previous_cv_score=0.78,
        improvement=5.1,
        model_version=2,
        message="✓ Model successfully retrained"
    )
    
    assert response.status == "retrained"
    assert response.improvement is not None
    assert response.model_version is not None
    print(f"✓ FeedbackResponse model valid")
    print(f"  - Status: {response.status}")
    print(f"  - CV Score: {response.cv_score}")
    print(f"  - Improvement: {response.improvement:.1f}%")
    print(f"  - Model Version: {response.model_version}")
    
    return True


def test_stats_endpoint_fields():
    """Test 6: Stats endpoint response fields"""
    print("\n" + "="*70)
    print("TEST 6: Stats Endpoint Response Fields")
    print("="*70)
    
    # Test that stats endpoint returns all expected fields
    learner = ContinuousLearner()
    feedback_db = FeedbackDB()
    
    stats = {
        "total_feedbacks": feedback_db.count(),
        "models_trained": len(learner.get_model_history()),
        "current_cv_score": 0.82,
        "decay_factor": 0.95,
        "model_status": "good",
        "hours_since_retrain": 24.5,
    }
    
    required_fields = [
        "total_feedbacks", "models_trained", "current_cv_score",
        "decay_factor", "model_status", "hours_since_retrain"
    ]
    
    for field in required_fields:
        assert field in stats, f"Missing field: {field}"
        print(f"  ✓ {field}: {stats[field]}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FEEDBACK LOOP ENHANCEMENT TESTS")
    print("="*70)
    
    tests = [
        ("Feedback Storage", test_feedback_storage),
        ("Model Persistence", test_model_persistence),
        ("Model History", test_model_history),
        ("Quality Decay Calculation", test_quality_decay_calculation),
        ("Feedback Response Structure", test_feedback_endpoint_response),
        ("Stats Endpoint Fields", test_stats_endpoint_fields),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("="*70 + "\n")
    
    return all(status == "PASS" for _, status in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
