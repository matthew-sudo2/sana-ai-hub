"""
Submit additional feedback to trigger model retraining.
"""

import requests
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

API_BASE_URL = "http://localhost:8000"


def get_dataset_hash(csv_path):
    """Compute MD5 hash of CSV file."""
    with open(csv_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def extract_8_features(df):
    """Extract 8 features matching MLQualityScorer."""
    from scipy.stats import skew, kurtosis
    
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns)) if (len(df) * len(df.columns)) > 0 else 0
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    cv_list = []
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if abs(mean_val) > 1e-10:
                cv = min(abs(std_val / mean_val), 100)
                cv_list.append(cv)
        except:
            pass
    norm_variance = np.mean(cv_list) if cv_list else 0
    
    skewness_list = []
    for col in numeric_cols:
        try:
            s = skew(df[col].dropna())
            skewness_list.append(abs(s))
        except:
            pass
    skewness = np.mean(skewness_list) if skewness_list else 0
    
    cardinalities = [df[col].nunique() for col in df.columns] if len(df.columns) > 0 else []
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    cardinality_ratio = (avg_cardinality / len(df)) if len(df) > 0 else 0
    cardinality_ratio = min(cardinality_ratio, 1.0)
    
    kurtosis_list = []
    for col in numeric_cols:
        try:
            kurt = kurtosis(df[col].dropna())
            if not np.isnan(kurt):
                kurtosis_list.append(abs(kurt))
        except:
            pass
    mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
    mean_kurtosis = min(mean_kurtosis / 10, 1.0)
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols,
            norm_variance, skewness, cardinality_ratio, mean_kurtosis]


print("=" * 70)
print("SUBMITTING FEEDBACK TO TRIGGER MODEL RETRAINING")
print("=" * 70)

# Test with multiple datasets to accumulate feedbacks
test_datasets = [
    "data/labeled/good/Spotify.csv",
    "data/labeled/bad/corruption_heavy_missing.csv",
    "data/labeled/good/employee_records.csv"
]

for i, csv_path in enumerate(test_datasets, 1):
    test_csv = Path(csv_path)
    if not test_csv.exists():
        print(f"✗ File not found: {test_csv}")
        continue
    
    print(f"\n[Feedback {i}] {test_csv.name}")
    df = pd.read_csv(test_csv)
    dataset_hash = get_dataset_hash(test_csv)
    features = extract_8_features(df)
    
    # Determine quality label based on path
    actual_quality = 3 if "good" in str(test_csv) else 0
    
    feedback_request = {
        "dataset_hash": dataset_hash,
        "predicted_score": 75.0,
        "actual_quality": actual_quality,
        "features": features
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/feedback",
            json=feedback_request,
            timeout=30
        )
        
        feedback_response = response.json()
        status = feedback_response.get('status')
        message = feedback_response.get('message', 'No message')
        
        print(f"  Status: {status}")
        print(f"  Message: {message}")
        
        if status == 'retrained':
            print(f"  ✓ MODEL RETRAINED!")
            print(f"    CV Score: {feedback_response.get('cv_score', 'N/A')}")
            break
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
