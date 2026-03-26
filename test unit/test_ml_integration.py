#!/usr/bin/env python3
"""
Test ML model integration with validator.
Validates and scores a sample CSV file.
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.utils.ml_quality_scorer import MLQualityScorer
from backend.agents.validator import run_validation


def test_ml_scorer():
    """Test ML Scorer independently."""
    print("\n" + "="*70)
    print("TEST 1: ML Quality Scorer")
    print("="*70)
    
    try:
        scorer = MLQualityScorer()
        print("✓ Model loaded successfully")
        
        # Test with a few sample datasets
        good_file = Path("data/labeled/good/Amazon_stock_data.csv")
        bad_file = Path("data/labeled/bad/corruption_heavy_missing.csv")
        
        if good_file.exists():
            df_good = pd.read_csv(good_file)
            score_good = scorer.score(df_good)
            print(f"\n  GOOD Dataset (Amazon): {score_good['quality']} ({score_good['score']:.1f}%)")
            print(f"    P(GOOD)={score_good['probability_good']:.2%}, P(BAD)={score_good['probability_bad']:.2%}")
        
        if bad_file.exists():
            df_bad = pd.read_csv(bad_file)
            score_bad = scorer.score(df_bad)
            print(f"\n  BAD Dataset (Heavy Missing): {score_bad['quality']} ({score_bad['score']:.1f}%)")
            print(f"    P(GOOD)={score_bad['probability_good']:.2%}, P(BAD)={score_bad['probability_bad']:.2%}")
        
        return True
    
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_in_validator():
    """Test ML scorer integration in validator."""
    print("\n" + "="*70)
    print("TEST 2: ML Integration in Validator")
    print("="*70)
    
    try:
        # Create a minimal test run directory
        test_run = Path("backend/runs/test_ml_run")
        test_run.mkdir(parents=True, exist_ok=True)
        
        # Use a sample dataset
        good_file = Path("data/labeled/good/Amazon_stock_data.csv")
        if not good_file.exists():
            print("✗ Test data not found")
            return False
        
        # Create required files for validation
        df = pd.read_csv(good_file)
        
        # Write cleaned data
        df.to_csv(test_run / "cleaned_data.csv", index=False)
        
        # Create minimal scout output
        scout_output = {
            "source": "test",
            "source_type": "test",
            "sample_data": df.head().to_dict()
        }
        (test_run / "scout_input.json").write_text(json.dumps(scout_output))
        
        # Create minimal analysis output
        analysis_output = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_stats": []
        }
        (test_run / "analysis_result.json").write_text(json.dumps(analysis_output))
        
        # Create minimal viz summary
        viz_summary = {
            "charts": [{"chart_id": "test", "png_path": str(test_run / "test.png")}]
        }
        (test_run / "viz_summary.json").write_text(json.dumps(viz_summary))
        
        # Create dummy chart image
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Test Chart", ha="center", va="center")
        fig.savefig(test_run / "test.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Run validation
        result = run_validation(test_run)
        
        print(f"\n✓ Validation completed")
        # Handle both dict and object results
        status = result.get("status") if isinstance(result, dict) else result.status
        overall_confidence = result.get("overall_confidence") if isinstance(result, dict) else result.overall_confidence
        checks = result.get("checks") if isinstance(result, dict) else result.checks
        dim_scores = result.get("dimension_scores") if isinstance(result, dict) else result.dimension_scores
        
        print(f"  Status: {status}")
        print(f"  Confidence: {overall_confidence:.1%}")
        print(f"  Total checks: {len(checks)}")
        
        # Find ML assessment check
        ml_checks = [c for c in checks if (c.get("category") if isinstance(c, dict) else c.category) == "ml_assessment"]
        if ml_checks:
            print(f"\n  ML Assessment Checks: {len(ml_checks)}")
            for check in ml_checks:
                desc = check.get("description") if isinstance(check, dict) else check.description
                detail = check.get("detail") if isinstance(check, dict) else check.detail
                print(f"    {desc}")
                if detail:
                    print(f"      {detail}")
        else:
            print("  (ML assessment skipped or unavailable)")
        
        # Check dimension scores
        print(f"\n  Dimension Scores:")
        for dim in dim_scores:
            dim_name = dim.get("dimension") if isinstance(dim, dict) else dim.dimension
            score = dim.get("score") if isinstance(dim, dict) else dim.score
            checks_passed = dim.get("checks_passed") if isinstance(dim, dict) else dim.checks_passed
            checks_total = dim.get("checks_total") if isinstance(dim, dict) else dim.checks_total
            print(f"    {dim_name}: {score:.1%} ({checks_passed}/{checks_total} checks)")
        
        return True
    
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 ML MODEL INTEGRATION TESTS")
    
    test1 = test_ml_scorer()
    test2 = test_ml_in_validator()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
