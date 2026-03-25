"""
Multi-level synthetic corruption: simplified and robust approach
Severity levels: Light (5%), Medium (12%), Severe (25%)
Expected: 11 datasets × 5 corruption types × 3 levels = 165 corrupted samples
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("MULTI-LEVEL SYNTHETIC CORRUPTION (Light, Medium, Severe)")
print("=" * 80)

# Load good datasets
print("\n1. LOADING GOOD DATASETS")
print("-" * 80)

good_dir = Path("traindata/raw")
good_datasets = {}

good_files = ["Accenture_stock_history.csv", "adult.csv", "data (1).csv", 
              "diabetes.csv", "olist_customers_dataset.csv", "pokemon.csv", 
              "submission.csv", "survey.csv", "Titanic-Dataset.csv",
              "WA_Fn-UseC_-Telco-Customer-Churn.csv", 
              "disuguaglianza-economica-globale-e-povert-1980-2024.csv"]

for fname in good_files:
    csv_file = good_dir / fname
    try:
        df = pd.read_csv(csv_file, nrows=1000)
        if df.shape[0] >= 10:
            good_datasets[fname] = df
            print(f"  OK {fname:<50} {df.shape[0]:>5} rows")
    except:
        pass

print(f"\nLoaded {len(good_datasets)} good datasets")

# Corruption functions - robust implementations
def corrupt_missing_pct(df, pct):
    """Introduce missing values randomly"""
    df = df.copy()
    mask = np.random.random(df.shape) < (pct / 100.0)
    df = df.mask(mask)
    return df

def corrupt_duplicates_pct(df, pct):
    """Add duplicate rows"""
    df = df.copy()
    n_dup = max(1, int(len(df) * (pct / 100.0)))
    dup_idx = np.random.choice(len(df), n_dup, replace=False)
    dup_rows = df.iloc[dup_idx]
    return pd.concat([df, dup_rows], ignore_index=True)

def corrupt_outliers_pct(df, pct):
    """Introduce outliers in numeric columns"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        n_outliers = max(1, int(len(df) * (pct / 100.0)))
        outlier_idx = np.random.choice(len(df), n_outliers, replace=False)
        for idx in outlier_idx:
            if pd.notna(df.loc[idx, col]):
                df.at[idx, col] = float(df.loc[idx, col] * np.random.choice([10, -10, 100]))
    
    return df

def corrupt_inconsistency_pct(df, pct):
    """Nullify entire columns partially"""
    df = df.copy()
    cols = list(df.columns)
    n_affected_cols = max(1, int(len(cols) * (pct / 100.0)))
    affected_cols = np.random.choice(cols, min(n_affected_cols, len(cols)), replace=False)
    
    for col in affected_cols:
        n_rows = max(1, int(len(df) * (pct / 100.0)))
        rows_to_null = np.random.choice(len(df), min(n_rows, len(df)), replace=False)
        for row_idx in rows_to_null:
            df.at[row_idx, col] = np.nan
    
    return df

def corrupt_mixed_pct(df, pct):
    """Apply 2-3 corruption types"""
    df = df.copy()
    n_types = np.random.randint(2, 4)
    funcs = [corrupt_missing_pct, corrupt_duplicates_pct, 
             corrupt_outliers_pct, corrupt_inconsistency_pct]
    
    selected = np.random.choice(funcs, n_types, replace=False)
    for func in selected:
        df = func(df, pct=pct)
    
    return df

def extract_features(df):
    """Extract 6 quality features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Missing ratio
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    
    # Duplicate ratio
    duplicate_ratio = (len(df) - len(df.drop_duplicates())) / max(1, len(df))
    
    # Numeric ratio
    numeric_ratio = len(numeric_cols) / max(1, len(df.columns))
    
    # Constant columns
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    # Variance and skewness
    variance = np.nanmean([df[col].var() for col in numeric_cols]) if len(numeric_cols) > 0 else 0
    skewness = np.nanmean([df[col].skew() for col in numeric_cols]) if len(numeric_cols) > 0 else 0
    
    # Fix for NumPy 2.x: avoid nan_to_num with copy parameter
    variance = float('inf') if np.isnan(variance) else variance
    skewness = float('inf') if np.isnan(skewness) else skewness
    variance = min(variance, 1e6) if variance != float('inf') else 0
    skewness = min(abs(skewness), 1e6) if skewness != float('inf') else 0
    
    return np.array([missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, variance, skewness])

# Generate multi-level corrupted datasets
print("\n2. GENERATING MULTI-LEVEL CORRUPTED DATASETS")
print("-" * 80)

severity_config = {
    "light": 5,
    "medium": 12,
    "severe": 25
}

corruption_types = [
    ("missing", corrupt_missing_pct),
    ("duplicates", corrupt_duplicates_pct),
    ("outliers", corrupt_outliers_pct),
    ("inconsistency", corrupt_inconsistency_pct),
    ("mixed", corrupt_mixed_pct),
]

raw_dir = Path("traindata/raw")
corrupted_data = []
total = 0

for dataset_name, dataset_df in good_datasets.items():
    stem = Path(dataset_name).stem
    
    for ctype, cfunc in corruption_types:
        for severity_name, severity_pct in severity_config.items():
            try:
                cdf = cfunc(dataset_df, severity_pct)
                
                if cdf.shape[0] < 2:
                    continue
                
                fname = f"{stem}_corrupt_{ctype}_{severity_name}.csv"
                cdf.to_csv(raw_dir / fname, index=False)
                
                features = extract_features(cdf)
                corrupted_data.append({"file": fname, "features": features, "label": 0})
                total += 1
                print(f"  OK {fname:<65}")
            except:
                pass

print(f"\nGenerated {total} multi-level corrupted datasets")

# Create augmented dataset
print("\n3. CREATING MULTI-LEVEL AUGMENTED DATASET")
print("-" * 80)

try:
    with open("data/synthetic/combined_real_data.pkl", "rb") as f:
        original = pickle.load(f)
    
    X_orig = original["X"]
    y_orig = original["y"]
    
    if len(corrupted_data) > 0:
        X_corr = np.vstack([d["features"].flatten() for d in corrupted_data])
        y_corr = np.zeros(len(corrupted_data))
        
        X_aug = np.vstack([X_orig, X_corr])
        y_aug = np.hstack([y_orig, y_corr])
        
        print(f"Original: {X_orig.shape[0]} samples")
        print(f"Corrupted (multi-level): {X_corr.shape[0]} samples")
        print(f"Augmented: {X_aug.shape[0]} samples")
        print(f"Increase: {X_aug.shape[0]/X_orig.shape[0]:.1f}x")
        
        good = np.sum(y_aug == 1)
        bad = np.sum(y_aug == 0)
        print(f"\nGood: {good} ({good/len(y_aug)*100:.1f}%)")
        print(f"Bad: {bad} ({bad/len(y_aug)*100:.1f}%)")
        
        # Breakdown by severity
        light_count = sum(1 for d in corrupted_data if "light" in d["file"])
        medium_count = sum(1 for d in corrupted_data if "medium" in d["file"])
        severe_count = sum(1 for d in corrupted_data if "severe" in d["file"])
        print(f"\nSeverity Breakdown:")
        print(f"  Light: {light_count} samples")
        print(f"  Medium: {medium_count} samples")
        print(f"  Severe: {severe_count} samples")
        
        aug_data = {
            "X": X_aug,
            "y": y_aug,
            "feature_names": original["feature_names"],
            "original_count": X_orig.shape[0],
            "corrupted_count": X_corr.shape[0],
            "improvement": "multi-level realistic patterns (light/medium/severe)"
        }
        
        with open("data/synthetic/augmented_data_multilevel.pkl", "wb") as f:
            pickle.dump(aug_data, f)
        
        print(f"\nSaved: data/synthetic/augmented_data_multilevel.pkl")
        print("=" * 80)
    else:
        print("NO CORRUPTED DATA GENERATED")
        
except FileNotFoundError:
    print("ERROR: combined_real_data.pkl not found. Skipping augmentation.")
except Exception as e:
    print(f"ERROR: {e}")
