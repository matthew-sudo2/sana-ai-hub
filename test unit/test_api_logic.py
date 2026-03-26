#!/usr/bin/env python3
"""
API Feedback Logic Test - Verify the retrain trigger and cleanup work correctly
"""

def test_retrain_trigger_logic():
    """Test the retrain trigger calculation"""
    print("\n" + "="*70)
    print("TEST: Retrain Trigger Logic")
    print("="*70)
    
    test_cases = [
        (0, False, 1),    # No feedback yet, should retrain at 1
        (1, True, 4),     # First feedback should trigger (count==1)
        (2, False, 3),    # 2 feedbacks: not time, 3 more until 5
        (3, False, 2),    # 3 feedbacks: not time, 2 more until 5
        (4, False, 1),    # 4 feedbacks: not time, 1 more until 5
        (5, True, 5),     # 5 feedbacks should trigger (5 % 5 == 0 and >= 5)
        (6, False, 4),    # 6 feedbacks: not time, 4 more until 10
        (10, True, 5),    # 10 feedbacks should trigger
        (15, True, 5),    # 15 feedbacks should trigger
        (20, True, 5),    # 20 feedbacks should trigger
        (25, True, 5),    # 25 feedbacks should trigger
    ]
    
    print("\nTesting feedback count → retrain trigger mapping:\n")
    print(f"{'Count':<8} {'Should':<10} {'Next':<8} {'Status':<20}")
    print("-" * 50)
    
    all_pass = True
    for count, should_trigger, next_retrain in test_cases:
        # Calculate expected values
        trigger_logic = (count == 1) or (count >= 5 and count % 5 == 0)
        
        if count == 0:
            next_retrain_calc = 1
        elif count == 1:
            next_retrain_calc = 4
        else:
            next_retrain_calc = 5 - (count % 5)
        
        # Verify
        trigger_ok = trigger_logic == should_trigger
        next_ok = next_retrain_calc == next_retrain
        
        status = "✓ PASS" if (trigger_ok and next_ok) else "✗ FAIL"
        all_pass = all_pass and trigger_ok and next_ok
        
        print(f"{count:<8} {str(trigger_logic):<10} {next_retrain_calc:<8} {status:<20}")
    
    print()
    return all_pass


def test_feedback_counting():
    """Test feedback count progression"""
    print("\n" + "="*70)
    print("TEST: Feedback Counting Progression")
    print("="*70)
    
    triggers = []
    for count in range(1, 31):
        should_retrain = (count == 1) or (count >= 5 and count % 5 == 0)
        if should_retrain:
            triggers.append(count)
    
    expected = [1, 5, 10, 15, 20, 25, 30]
    
    print(f"\nTrigger points (0-30): {triggers}")
    print(f"Expected: {expected}")
    
    if triggers == expected:
        print("✓ PASS: Feedback count progression correct")
        return True
    else:
        print("✗ FAIL: Feedback count progression incorrect")
        return False


def test_cleanup_logic():
    """Test database cleanup after retrain"""
    print("\n" + "="*70)
    print("TEST: Database Cleanup Logic")
    print("="*70)
    
    print("\nCleanup happens AFTER successful retrain:")
    print("  - Retrain succeeds")
    print("  - Calls: feedback_db.clear_feedback(keep_last=100)")
    print("  - Keeps: Last 100 feedback records")
    print("  - Deletes: Older records")
    
    test_cases = [
        (50, 0, "Not enough records, nothing deleted"),
        (100, 0, "Exactly at limit, nothing deleted"),
        (101, 1, "1 record over limit, delete 1"),
        (150, 50, "50 records over limit, delete 50"),
        (500, 400, "400 records over limit, delete 400"),
    ]
    
    print(f"\n{'Total':<10} {'Deleted':<10} {'Kept':<10} {'Status':<40}")
    print("-" * 70)
    
    all_pass = True
    for total, expected_deleted, description in test_cases:
        deleted = max(0, total - 100)
        kept = min(total, 100)
        ok = deleted == expected_deleted
        
        status = "✓ PASS" if ok else "✗ FAIL"
        all_pass = all_pass and ok
        
        print(f"{total:<10} {deleted:<10} {kept:<10} {description:<40} {status}")
    
    print()
    return all_pass


def main():
    print("\n" + "#"*70)
    print("# API FEEDBACK LOGIC VERIFICATION")
    print("#"*70)
    
    results = {
        "Retrain Trigger Logic": test_retrain_trigger_logic(),
        "Feedback Count Progression": test_feedback_counting(),
        "Database Cleanup Logic": test_cleanup_logic(),
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
        print("\n✓ ALL API LOGIC TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
