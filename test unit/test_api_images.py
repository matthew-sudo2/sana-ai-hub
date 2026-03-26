"""Test API endpoint for serving ML assessment images"""

import requests
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import pandas as pd
from backend.graph import create_run, execute_run
from backend.agents.validator import run_validation

def test_api_image_serving():
    """Test that API serves ML assessment images correctly."""
    
    print("🧪 Testing API image serving endpoint...\n")
    
    # Create a test run
    test_run = create_run(source="test-data.csv", source_type="file")
    run_id = test_run["run_id"]
    run_dir = Path(test_run["output_dir"])
    
    print(f"📁 Created test run: {run_id}")
    print(f"   Directory: {run_dir}\n")
    
    # Create minimal test data
    test_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45],
        'Salary': [30000, 35000, 40000, 45000, 50000],
    }
    df = pd.DataFrame(test_data)
    
    # Save test files
    raw_csv = run_dir / "raw_data.csv"
    cleaned_csv = run_dir / "cleaned_data.csv"
    df.to_csv(raw_csv, index=False)
    df.to_csv(cleaned_csv, index=False)
    
    # Run validation to generate images
    print("Running validation to generate images...\n")
    result = run_validation(run_dir)
    
    # Check files were created
    images = [
        "ml_confidence_gauge.png",
        "ml_feature_radar.png",
        "ml_feature_comparison.png",
        "ml_probability_breakdown.png",
    ]
    
    print("Checking generated files:\n")
    for img in images:
        img_path = run_dir / img
        if img_path.exists():
            print(f"   ✓ {img} ({img_path.stat().st_size:,} bytes)")
        else:
            print(f"   ✗ {img} NOT FOUND")
            return False
    
    # Test API endpoints
    print("\n\n🌐 Testing API endpoints:\n")
    
    base_url = "http://localhost:8000"
    
    for img in images:
        url = f"{base_url}/runs/{run_id}/ml-assessment/{img}"
        print(f"   GET {url}")
        
        try:
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', 'unknown')
                content_length = len(response.content)
                print(f"      ✓ 200 OK ({content_type}, {content_length:,} bytes)\n")
            else:
                print(f"      ✗ {response.status_code} {response.reason}")
                print(f"      Response: {response.text}\n")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"      ✗ Connection refused (is server running?)\n")
            return False
        except Exception as e:
            print(f"      ✗ Error: {e}\n")
            return False
    
    # Test validation report endpoint
    print("Testing validation report endpoint:\n")
    url = f"{base_url}/runs/{run_id}/validation-report"
    print(f"   GET {url}\n")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '')
            
            if 'Machine Learning Data Quality Assessment' in content:
                print(f"   ✓ 200 OK - ML Assessment section found\n")
            else:
                print(f"   ⚠️  200 OK but ML section not in response\n")
                
            if 'ml_confidence_gauge.png' in content:
                print(f"   ✓ Image references found in report\n")
            else:
                print(f"   ⚠️  No image references found in report\n")
        else:
            print(f"   ✗ {response.status_code} {response.reason}\n")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False
    
    print("✅ All API endpoint tests passed!\n")
    return True

if __name__ == "__main__":
    success = test_api_image_serving()
    if not success:
        print("❌ API test failed")
    sys.exit(0 if success else 1)
