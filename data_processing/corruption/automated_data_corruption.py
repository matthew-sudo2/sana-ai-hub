import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from features.quality_features import extract_quality_features

def corrupt_missing_values(df, severity=0.1):
    df = df.copy()
    n_cells = df.shape[0] * df.shape[1]
    n_missing = max(1, int(n_cells * severity))
    for _ in range(n_missing):
        df.iloc[np.random.randint(0, df.shape[0]), np.random.randint(0, df.shape[1])] = np.nan
    return df

def corrupt_duplicates(df, severity=0.05):
    df = df.copy()
    n_dup = max(1, int(df.shape[0] * severity))
    dup_indices = np.random.choice(df.shape[0], size=n_dup, replace=True)
    return pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

def corrupt_text_encoding(df, severity=0.05):
    df = df.copy()
    text_cols = df.select_dtypes(include=["object"]).columns
    if len(text_cols) > 0:
        for _ in range(max(1, int(df.shape[0] * severity))):
            row = np.random.randint(0, df.shape[0])
            col = np.random.choice(text_cols)
            val = df.loc[row, col]
            if isinstance(val, str) and len(val) > 0:
                df.loc[row, col] = val[:len(val)//2] if np.random.rand() > 0.5 else val + "###"
    return df

def corrupt_outliers(df, severity=0.05):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for _ in range(max(1, int(df.shape[0] * severity))):
            row = np.random.randint(0, df.shape[0])
            col = np.random.choice(numeric_cols)
            val = df.loc[row, col]
            if pd.notna(val):
                df.loc[row, col] = val * np.random.choice([-100, 50, 100])
    return df

def corrupt_inconsistency(df, severity=0.05):
    df = df.copy()
    n_cols = max(1, int(df.shape[1] * severity))
    cols_to_corrupt = np.random.choice(df.columns, size=min(n_cols, len(df.columns)), replace=False)
    for col in cols_to_corrupt:
        if np.random.rand() > 0.5:
            df[col] = np.nan
        else:
            n_mixed = max(1, int(df.shape[0] * 0.1))
            for _ in range(n_mixed):
                df.iloc[np.random.randint(0, df.shape[0]), df.columns.get_loc(col)] = np.nan
    return df

def corrupt_mixed(df, severity=0.1):
    df = df.copy()
    if np.random.rand() > 0.5: df = corrupt_missing_values(df, severity * 0.8)
    if np.random.rand() > 0.5: df = corrupt_duplicates(df, severity * 0.3)
    if np.random.rand() > 0.5: df = corrupt_text_encoding(df, severity * 0.5)
    if np.random.rand() > 0.5: df = corrupt_outliers(df, severity * 0.5)
    if np.random.rand() > 0.5: df = corrupt_inconsistency(df, severity * 0.3)
    return df

print("\n" + "="*80)
print("PHASE 1: AUTOMATED DATA CORRUPTION PIPELINE")
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

print("\n2️⃣ GENERATING CORRUPTED DATASETS")
print("-"*80)

corruptions = [
    ("missing", corrupt_missing_values, 0.10),
    ("missing_heavy", corrupt_missing_values, 0.25),
    ("duplicates", corrupt_duplicates, 0.10),
    ("duplicates_heavy", corrupt_duplicates, 0.25),
    ("encoding", corrupt_text_encoding, 0.10),
    ("outliers", corrupt_outliers, 0.10),
    ("inconsistency", corrupt_inconsistency, 0.08),
    ("mixed", corrupt_mixed, 0.12),
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
            
            fname = f"{stem}_corrupt_{ctype}.csv"
            cdf.to_csv(raw_dir / fname, index=False)
            
            features = extract_quality_features(cdf)
            if features is not None:
                corrupted_data.append({"file": fname, "features": features, "label": 0})
                total += 1
                print(f"  ✓ {fname:<50}")
        except:
            pass

print(f"\n✓ Generated {total} corrupted datasets")

print("\n3️⃣ AUGMENTING TRAINING DATASET")
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
    print(f"✓ Corrupted: {X_corr.shape[0]} samples")
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
    }
    
    with open("data/synthetic/augmented_data.pkl", "wb") as f:
        pickle.dump(aug_data, f)
    
    print(f"\n✅ Saved: data/synthetic/augmented_data.pkl")

print("="*80 + "\n")
