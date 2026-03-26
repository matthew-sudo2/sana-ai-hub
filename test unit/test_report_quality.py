"""Quick test of updated ML visualizations without defensive language"""

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.utils.ml_assessment_viz import MLAssessmentVisualizer

def test_updated_report():
    """Test that defensive language is removed from report."""
    
    print("🧪 Testing updated ML report (without defensive language)...\n")
    
    # Mock scorer output
    features = [0.05, 0.03, 0.45, 0, 1.2, 0.5, 0.8, 2.8]
    prediction = 1  # GOOD
    probabilities = [0.15, 0.85]
    
    # Create temporary DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'Age': range(1, 11),
        'Salary': range(30000, 40000, 1000),
        'Dept': ['IT'] * 10
    })
    
    visualizer = MLAssessmentVisualizer(
        features=features,
        prediction=prediction,
        probabilities=probabilities,
        df=df
    )
    
    markdown = visualizer.generate_markdown_section()
    
    # Check that defensive language is NOT present
    bad_phrases = [
        "How to Validate This",
        "Feature Reproducibility",
        "Model Transparency",
        "Ground Truth",
        "Test It Yourself",
        "Add missing values",
        "Add duplicates",
    ]
    
    found_bad = []
    for phrase in bad_phrases:
        if phrase in markdown:
            found_bad.append(phrase)
    
    if found_bad:
        print(f"❌ Found defensive phrases that should be removed:")
        for phrase in found_bad:
            print(f"   - '{phrase}'")
        print(f"\nReport content:\n{markdown}")
        return False
    
    # Check that good content IS present
    good_phrases = [
        "Machine Learning Data Quality Assessment",
        "Prediction:",
        "Model Overview",
        "Feature Analysis",
        "Model Probabilities",
    ]
    
    found_good = []
    for phrase in good_phrases:
        if phrase in markdown:
            found_good.append(phrase)
    
    print(f"✅ Verified: Removed defensive language")
    print(f"✅ Found {len(found_good)}/{len(good_phrases)} expected sections:")
    for phrase in found_good:
        print(f"   ✓ {phrase}")
    
    print(f"\n📄 Report excerpt (first 500 chars):")
    print(markdown[:500] + "...")
    
    return True

if __name__ == "__main__":
    success = test_updated_report()
    sys.exit(0 if success else 1)
