"""Test ML Assessment Visualizations"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.agents.validator import run_validation
from backend.utils.ml_quality_scorer import MLQualityScorer
from backend.utils.ml_assessment_viz import MLAssessmentVisualizer


def test_ml_visualizations():
    """Test that ML visualizations generate correctly."""
    
    print("🧪 Testing ML Assessment Visualizations...")
    
    # Create test data
    test_data = {
        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'Department': ['Sales', 'IT', 'HR', 'Finance', 'Sales', 'IT', 'HR', 'Finance', 'Sales', 'IT'],
    }
    
    df = pd.DataFrame(test_data)
    
    # Test 1: ML Scorer
    print("\n📊 Test 1: ML Quality Scoring")
    try:
        scorer = MLQualityScorer()
        score_result = scorer.score(df)
        print(f"   ✓ ML Score: {score_result['quality']} ({score_result['score']:.1f}%)")
        print(f"   ✓ Features extracted: {len(score_result['features'])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Visualizer
    print("\n🎨 Test 2: Visualization Generation")
    try:
        visualizer = MLAssessmentVisualizer(
            features=score_result['features'],
            prediction=1 if score_result['quality'] == 'GOOD' else 0,
            probabilities=[score_result['probability_bad'], score_result['probability_good']],
            df=df
        )
        
        # Create temp directory for test output
        test_dir = Path(__file__).parent / "test_ml_viz"
        test_dir.mkdir(exist_ok=True)
        
        # Generate all visualizations
        print("   Generating confidence gauge...")
        visualizer.generate_confidence_gauge(str(test_dir / "gauge.png"))
        
        print("   Generating feature radar...")
        visualizer.generate_feature_radar(str(test_dir / "radar.png"))
        
        print("   Generating feature comparison...")
        visualizer.generate_feature_comparison(str(test_dir / "comparison.png"))
        
        print("   Generating probability breakdown...")
        visualizer.generate_probability_breakdown(str(test_dir / "probability.png"))
        
        # Check files exist
        files = [
            test_dir / "gauge.png",
            test_dir / "radar.png", 
            test_dir / "comparison.png",
            test_dir / "probability.png",
        ]
        
        for f in files:
            if f.exists():
                size = f.stat().st_size
                print(f"   ✓ {f.name} ({size:,} bytes)")
            else:
                print(f"   ✗ {f.name} NOT CREATED")
                return False
        
        # Generate markdown
        markdown = visualizer.generate_markdown_section()
        if markdown and len(markdown) > 100:
            print(f"   ✓ Markdown report ({len(markdown)} chars)")
        else:
            print(f"   ✗ Markdown report too short")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All ML visualization tests passed!")
    return True


if __name__ == "__main__":
    success = test_ml_visualizations()
    sys.exit(0 if success else 1)
