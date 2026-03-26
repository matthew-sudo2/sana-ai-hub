"""
Retrain model with real labeled datasets.

This script:
1. Loads augmented synthetic data (188 samples)
2. Loads real labeled datasets from data/labeled/good and data/labeled/bad
3. Combines them
4. Retrains Random Forest
5. Compares CV scores: synthetic vs real
6. Saves improved model

Usage:
    python models/train/retrain_with_real_data.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.agents.scout import (
    _compute_completeness,
    _compute_consistency,
    _compute_accuracy_score,
    _compute_duplicate_score,
    _compute_outlier_score,
)


def extract_features(df):
    """Extract 6 base features from a dataset - matching original synthetic data format."""
    
    # Compute quality metrics
    completeness = _compute_completeness(df)
    consistency = _compute_consistency(df)
    accuracy = _compute_accuracy_score(df)
    duplicates = _compute_duplicate_score(df)
    outliers = _compute_outlier_score(df)
    
    # Base 6 features only
    missing_ratio = 1 - completeness
    duplicate_ratio = 1 - duplicates  # Inverse of uniqueness
    numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    numeric_cols = df.select_dtypes(include=[np.number])
    variance = numeric_cols.var().mean() if not numeric_cols.empty else 0.0
    variance = 0.0 if np.isnan(variance) else variance
    
    skewness = numeric_cols.skew().abs().mean() if not numeric_cols.empty else 0.0
    skewness = 0.0 if np.isnan(skewness) else skewness
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols,
            variance, skewness]


def load_real_labeled_data():
    """
    Auto-load all datasets from data/labeled/good and data/labeled/bad
    Returns: X_real (features), y_real (labels where 1=GOOD, 0=BAD)
    """
    X_real = []
    y_real = []
    dataset_names = []
    
    # Load GOOD datasets (label = 1)
    good_dir = Path('data/labeled/good')
    if good_dir.exists():
        for csv_file in sorted(good_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:  # Min size check
                    features = extract_features(df)
                    X_real.append(features)
                    y_real.append(1)
                    dataset_names.append((csv_file.name, 'GOOD'))
                    print(f"  ✓ {csv_file.name:<40} (GOOD)")
            except Exception as e:
                print(f"  ✗ {csv_file.name} - Error: {e}")
    
    # Load BAD datasets (label = 0)
    bad_dir = Path('data/labeled/bad')
    if bad_dir.exists():
        for csv_file in sorted(bad_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:  # Min size check
                    features = extract_features(df)
                    X_real.append(features)
                    y_real.append(0)
                    dataset_names.append((csv_file.name, 'BAD'))
                    print(f"  ✗ {csv_file.name:<40} (BAD)")
            except Exception as e:
                print(f"  ✗ {csv_file.name} - Error: {e}")
    
    return np.array(X_real), np.array(y_real), dataset_names


def load_augmented_data():
    """Load original augmented synthetic data (188 samples)"""
    try:
        with open('data/synthetic/augmented_data_multilevel.pkl', 'rb') as f:
            data = pickle.load(f)
            X_synthetic, y_synthetic = data['X'], data['y']
            return X_synthetic, y_synthetic
    except FileNotFoundError:
        print("⚠️  augmented_data_multilevel.pkl not found. Using synthetic fallback.")
        # Generate minimal synthetic data if file missing
        return None, None


def main():
    print("=" * 70)
    print("RETRAINING MODEL WITH REAL LABELED DATASETS")
    print("=" * 70)
    
    # Step 1: Load augmented synthetic data
    print("\n[1] Loading augmented synthetic data (188 samples)...")
    X_synthetic, y_synthetic = load_augmented_data()
    
    if X_synthetic is None:
        print("⚠️  No synthetic data found. Will train only on real data.")
        X_combined = None
        y_combined = None
    else:
        print(f"    ✓ Loaded {X_synthetic.shape[0]} synthetic samples")
    
    # Step 2: Load real labeled datasets
    print("\n[2] Loading real labeled datasets...")
    X_real, y_real, dataset_names = load_real_labeled_data()
    
    if len(X_real) == 0:
        print("⚠️  No real datasets found in data/labeled/good or data/labeled/bad")
        print("    Please add CSV files to these directories.")
        return
    
    print(f"    ✓ Loaded {len(X_real)} real datasets")
    
    # Step 3: Combine data
    print("\n[3] Combining synthetic + real data...")
    if X_synthetic is not None:
        X_combined = np.vstack([X_synthetic, X_real])
        y_combined = np.hstack([y_synthetic, y_real])
        print(f"    ✓ Combined: {X_synthetic.shape[0]} synthetic + {len(X_real)} real = {X_combined.shape[0]} total")
    else:
        X_combined = X_real
        y_combined = y_real
        print(f"    ✓ Using real data only: {len(X_real)} samples")
    
    # Step 4: Train model
    print("\n[4] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validate on SYNTHETIC data (to compare with 95.2%)
    if X_synthetic is not None:
        print("    Computing CV on synthetic data...")
        cv_synthetic = cross_val_score(
            model, X_synthetic, y_synthetic,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        )
        print(f"    Synthetic CV: {cv_synthetic.mean():.3f} ± {cv_synthetic.std():.3f}")
    else:
        print("    (Skipping synthetic CV - no synthetic data)")
    
    # Cross-validate on REAL data
    print("    Computing CV on real data...")
    n_splits = min(5, len(X_real) // 2)  # Handle small real dataset
    if n_splits < 2:
        n_splits = 2
    
    cv_real = cross_val_score(
        model, X_real, y_real,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    )
    print(f"    Real data CV: {cv_real.mean():.3f} ± {cv_real.std():.3f}")
    
    # Fit on all combined data
    print("\n[5] Fitting model on all combined data...")
    model.fit(X_combined, y_combined)
    print(f"    ✓ Model trained")
    
    # Step 5: Display feature importance
    print("\n[6] Feature importance:")
    feature_names = [
        'missing_ratio', 'duplicate_ratio', 'numeric_ratio', 'constant_cols',
        'variance', 'skewness', 'M×D', 'M×N', 'V×S', 'log(V)', 'log(|S|)', 'V/S', 'S/V'
    ]
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"    {name:<20} {importance:.4f}")
    
    # Step 6: Save model
    print("\n[7] Saving model...")
    model_path = Path('models/best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"    ✓ Model saved to {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training samples: {X_combined.shape[0]} (synthetic + real)")
    print(f"Features: 13 engineered features")
    if X_synthetic is not None:
        print(f"Synthetic CV: {cv_synthetic.mean():.3f} ± {cv_synthetic.std():.3f} (baseline: 95.2%)")
    print(f"Real data CV: {cv_real.mean():.3f} ± {cv_real.std():.3f}")
    print(f"\n✓ Model ready for testing on real datasets!")
    print("=" * 70)


if __name__ == '__main__':
    main()
