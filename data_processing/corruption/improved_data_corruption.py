import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from features.quality_features import extract_quality_features

def corrupt_missing_cluster(df, severity=0.1):
    """Missing values appear in clusters of related columns"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    all_cols = numeric_cols + text_cols
    n_cluster_cols = max(1, int(len(all_cols) * 0.3))
    cluster_cols = np.random.choice(all_cols, size=min(n_cluster_cols, len(all_cols)), replace=False)
    
    n_rows_to_corrupt = max(1, int(df.shape[0] * severity))
    corrupt_rows = np.random.choice(df.shape[0], size=n_rows_to_corrupt, replace=False)
    
    for row in corrupt_rows:
        for col in cluster_cols:
            df.loc[row, col] = np.nan
    
    return df

def corrupt_duplicate_patterns(df, severity=0.05):
    """Duplicates with slight variations (realistic)"""
    df = df.copy()
    n_dup = max(1, int(df.shape[0] * severity))
    
    dup_indices = np.random.choice(df.shape[0], size=n_dup, replace=True)
    
    for idx in dup_indices:
        dup_row = df.iloc[idx].copy()
        
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            if np.random.rand() > 0.7 and isinstance(dup_row[col], str):
                dup_row[col] = dup_row[col][:len(dup_row[col])//2]
        
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
    
    return df

def corrupt_type_mismatch(df, severity=0.05):
    """Type inconsistencies (e.g., string in numeric column)"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return df
    
    n_corrupted = max(1, int(df.shape[0] * severity))
    
    for _ in range(n_corrupted):
        row = np.random.randint(0, df.shape[0])
        col = np.random.choice(numeric_cols)
        df.loc[row, col] = "INVALID"
    
    return df

def corrupt_outlier_cluster(df, severity=0.05):
    """Multiple outliers in same row (realistic)"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df
    
    n_rows = max(1, int(df.shape[0] * severity))
    
    for _ in range(n_rows):
        row = np.random.randint(0, df.shape[0])
        n_outlier_cols = np.random.randint(2, len(numeric_cols))
        cols = np.random.choice(numeric_cols, size=n_outlier_cols, replace=False)
        
        for col in cols:
            val = df.loc[row, col]
            if pd.notna(val):
                df.loc[row, col] = val * np.random.choice([-50, 50, 100])
    
    return df

def corrupt_logical_inconsistency(df, severity=0.08):
    """Inconsistent values that violate logic"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df
    
    n_rows = max(1, int(df.shape[0] * severity))
    
    for _ in range(n_rows):
        row = np.random.randint(0, df.shape[0])
        col = np.random.choice(numeric_cols)
        
        val = df.loc[row, col]
        if pd.notna(val) and val != 0:
            df.loc[row, col] = -abs(val) if val > 0 else abs(val)
    
    return df

def corrupt_mixed_realistic(df, severity=0.12):
    """Realistic mix of multiple corruption types"""
    df = df.copy()
    
    if np.random.rand() > 0.5: df = corrupt_missing_cluster(df, severity * 0.6)
    if np.random.rand() > 0.5: df = corrupt_duplicate_patterns(df, severity * 0.3)
    if np.random.rand() > 0.4: df = corrupt_type_mismatch(df, severity * 0.2)
    if np.random.rand() > 0.5: df = corrupt_outlier_cluster(df, severity * 0.4)
    if np.random.rand() > 0.6: df = corrupt_logical_inconsistency(df, severity * 0.3)
    
    return df

print("\n" + "="*80)
print("OPTION A: IMPROVED SYNTHETIC CORRUPTION (Realistic Patterns)")
print("="*80)

raw_dir = Path("traindata/raw")
good_files = [
    "Accenture_stock_history.csv", "adult.csv", "data (1).csv", "diabetes.csv",
    "olist_customers_dataset.csv", "pokemon.csv", "submission.csv", "survey.csv",
    "Titanic-Dataset.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "disuguaglianza-economica-globale-e-povert-1980-2024.csv"
]

good_datasets = {}
print("\n1️⃣ LOADING GOOD DATASETS")
print("-"*80)
for fname in good_files:
    fpath = raw_dir / fname
    if fpath.exists():
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip", encoding="latin-1")
            if df.shape[0] > 0:
                good_datasets[fname] = df.copy()
                print(f"  ✓ {fname:<40} {df.shape[0]:>6} rows")
        except:
            pass

print(f"\n✓ Loaded {len(good_datasets)} good datasets")

print("\n2️⃣ GENERATING IMPROVED CORRUPTED DATASETS")
print("-"*80)

corruptions = [
    ("cluster_missing", corrupt_missing_cluster, 0.15),
    ("cluster_heavy", corrupt_missing_cluster, 0.30),
    ("dup_pattern", corrupt_duplicate_patterns, 0.08),
    ("dup_heavy", corrupt_duplicate_patterns, 0.15),
    ("type_mismatch", corrupt_type_mismatch, 0.08),
    ("outlier_cluster", corrupt_outlier_cluster, 0.10),
    ("logical_incons", corrupt_logical_inconsistency, 0.12),
    ("mixed_realistic", corrupt_mixed_realistic, 0.12),
]

corrupted_data = []
total = 0

for dataset_name, dataset_df in good_datasets.items():
    stem = Path(dataset_name).stem
    for ctype, cfunc, severity in corruptions:
        try:
            cdf = cfunc(dataset_df, severity=severity)
            if cdf.shape[0] < 2: 
                continue
            
            fname = f"{stem}_improved_{ctype}.csv"
            cdf.to_csv(raw_dir / fname, index=False)
            
            features = extract_quality_features(cdf)
            if features is not None:
                corrupted_data.append({"file": fname, "features": features, "label": 0})
                total += 1
                print(f"  ✓ {fname:<50}")
        except:
            pass

print(f"\n✓ Generated {total} improved corrupted datasets")

print("\n3️⃣ CREATING IMPROVED AUGMENTED DATASET")
print("-"*80)

with open("data/synthetic/combined_real_data.pkl", "rb") as f:
    original = pickle.load(f)

X_orig = original["X"]
y_orig = original["y"]

if len(corrupted_data) > 0:
    X_corr = np.vstack([d["features"].flatten() for d in corrupted_data])
    y_corr = np.zeros(len(corrupted_data))
    
    X_aug = np.vstack([X_orig, X_corr])
    y_aug = np.hstack([y_orig, y_corr])
    
    print(f"✓ Original: {X_orig.shape[0]} samples")
    print(f"✓ Corrupted (improved): {X_corr.shape[0]} samples")
    print(f"✓ Augmented: {X_aug.shape[0]} samples")
    print(f"   Increase: {X_aug.shape[0]/X_orig.shape[0]:.1f}x")
    
    good = np.sum(y_aug == 1)
    bad = np.sum(y_aug == 0)
    print(f"\n  Good: {good} ({good/len(y_aug)*100:.1f}%)")
    print(f"  Bad: {bad} ({bad/len(y_aug)*100:.1f}%)")
    
    aug_data = {
        "X": X_aug,
        "y": y_aug,
        "feature_names": original["feature_names"],
        "original_count": X_orig.shape[0],
        "corrupted_count": X_corr.shape[0],
        "improvement": "realistic patterns"
    }
    
    with open("data/synthetic/augmented_data_improved.pkl", "wb") as f:
        pickle.dump(aug_data, f)
    
    print(f"\n✅ Saved: data/synthetic/augmented_data_improved.pkl")

print("="*80 + "\n")
