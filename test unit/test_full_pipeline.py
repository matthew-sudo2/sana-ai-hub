"""Test full validation pipeline with ML visualizations"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import pandas as pd
from backend.agents.validator import run_validation

def test_full_validation_with_images():
    """Run full validation and verify images are created."""
    
    print("🧪 Testing full validation pipeline with ML visualizations...\n")
    
    # Create test directory
    test_run_dir = Path(__file__).parent / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_run_dir.mkdir(exist_ok=True)
    print(f"📁 Test run directory: {test_run_dir}\n")
    
    # Create test CSV files
    test_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack'],
        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'Department': ['IT', 'HR', 'Finance', 'Sales', 'IT', 'HR', 'Finance', 'Sales', 'IT', 'HR'],
    }
    
    df = pd.DataFrame(test_data)
    
    # Save as raw and cleaned CSV
    raw_csv = test_run_dir / "raw_data.csv"
    cleaned_csv = test_run_dir / "cleaned_data.csv"
    
    df.to_csv(raw_csv, index=False)
    df.to_csv(cleaned_csv, index=False)
    
    print(f"✓ Created test data: {len(df)} rows, {len(df.columns)} columns\n")
    
    # Run validation
    print("Running validation...\n")
    try:
        result = run_validation(test_run_dir)
        print(f"✅ Validation complete: {result['status']}\n")
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for generated files
    print("Checking generated files:\n")
    
    expected_files = [
        "validation_result.json",
        "validation_report.md",
        "ml_confidence_gauge.png",
        "ml_feature_radar.png",
        "ml_feature_comparison.png",
        "ml_probability_breakdown.png",
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = test_run_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"   ✓ {filename} ({size:,} bytes)")
        else:
            print(f"   ✗ {filename} MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files missing!")
        print(f"Directory contents: {list(test_run_dir.iterdir())}")
        return False
    
    # Check report content
    print("\n📄 Validating report content:\n")
    
    report_path = test_run_dir / "validation_report.md"
    report_content = report_path.read_text()
    
    # Check for ML assessment section
    if "Machine Learning Data Quality Assessment" in report_content:
        print("   ✓ ML Assessment section present")
    else:
        print("   ✗ ML Assessment section MISSING")
        all_exist = False
    
    # Check for image references
    image_refs = [
        "ml_confidence_gauge.png",
        "ml_feature_radar.png", 
        "ml_feature_comparison.png",
        "ml_probability_breakdown.png",
    ]
    
    for img_ref in image_refs:
        if img_ref in report_content:
            print(f"   ✓ Image reference: {img_ref}")
        else:
            print(f"   ✗ Image reference MISSING: {img_ref}")
            all_exist = False
    
    # Check that defensive language is gone
    bad_phrases = ["How to Validate This", "Test It Yourself", "Add missing values"]
    has_bad = False
    for phrase in bad_phrases:
        if phrase in report_content:
            print(f"   ⚠️  Found defensive phrase: {phrase}")
            has_bad = True
    
    if not has_bad:
        print("   ✓ No defensive language found")
    
    # Show report excerpt
    print("\n📋 Report ML section excerpt:\n")
    ml_start = report_content.find("## Machine Learning Data Quality Assessment")
    if ml_start > 0:
        ml_end = report_content.find("## Visualizations", ml_start)
        if ml_end < 0:
            ml_end = len(report_content)
        excerpt = report_content[ml_start:min(ml_start+800, ml_end)]
        print(excerpt + "\n...")
    
    # Cleanup
    print("\n🧹 Cleaning up test directory...\n")
    shutil.rmtree(test_run_dir)
    print(f"✅ Removed {test_run_dir}")
    
    return all_exist and not has_bad

if __name__ == "__main__":
    success = test_full_validation_with_images()
    if success:
        print("\n✅ All tests passed! ML visualizations are working correctly.")
    else:
        print("\n❌ Some tests failed. See details above.")
    sys.exit(0 if success else 1)
