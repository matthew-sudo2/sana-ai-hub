#!/usr/bin/env python3
"""
Enhanced retraining with:
1. Synthetic GOOD datasets for balance
2. New features for better signals
3. Tuned hyperparameters
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def extract_features_enhanced(df):
    """Extract enhanced features (original 6 + 4 new)"""
    
    # Original 6 features
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    cv_list = []
    skewness_list = []
    
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if abs(mean_val) > 1e-10:
                cv = min(abs(std_val / mean_val), 100)
                cv_list.append(cv)
        except:
            pass
        
        try:
            skew = df[col].skew()
            if not pd.isna(skew):
                skewness_list.append(abs(skew))
        except:
            pass
    
    norm_variance = np.mean(cv_list) if cv_list else 0
    skewness = np.mean(skewness_list) if skewness_list else 0
    
    # NEW FEATURES
    # Feature 7: Type consistency (good data is mostly numeric or mostly strings)
    type_consistency = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # Feature 8: Cardinality ratio (good data has balanced cardinality)
    cardinalities = [df[col].nunique() for col in df.columns]
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    cardinality_ratio = (avg_cardinality / len(df)) * 100 if len(df) > 0 else 0
    cardinality_ratio = min(cardinality_ratio, 100)  # Cap at 100
    
    # Feature 9: Distribution shape variance (good data has consistent distributions)
    if len(skewness_list) > 1:
        skewness_variance = np.var(skewness_list)
    else:
        skewness_variance = 0
    
    # Feature 10: Kurtosis (good data has normal-ish distributions)
    kurtosis_list = []
    for col in numeric_cols:
        try:
            kurt = df[col].kurtosis()
            if not pd.isna(kurt):
                kurtosis_list.append(abs(kurt))
        except:
            pass
    
    mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
    
    return [
        missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
        norm_variance, skewness,
        type_consistency, cardinality_ratio, skewness_variance, mean_kurtosis
    ]

def main():
    print("\n" + "="*80)
    print("ENHANCED RETRAINING WITH SYNTHETIC GOOD + NEW FEATURES")
    print("="*80)
    
    X = []
    y = []
    dataset_names = []
    
    # Load GOOD datasets (both real and synthetic)
    good_dir = 'data/labeled/good'
    print(f"\nLoading GOOD datasets from {good_dir}...")
    good_count = 0
    
    if os.path.exists(good_dir):
        for filename in sorted(os.listdir(good_dir)):
            if filename.endswith('.csv'):
                filepath = os.path.join(good_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_enhanced(df)
                    X.append(features)
                    y.append(1)  # GOOD = 1
                    good_count += 1
                    tag = "[SYNTH]" if filename.startswith('synthetic') else "[REAL]"
                    print(f"✓ {tag} {filename:45}")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
    
    # Load BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"\nLoading BAD datasets from {bad_dir}...")
    bad_count = 0
    
    if os.path.exists(bad_dir):
        for filename in sorted(os.listdir(bad_dir)):
            if filename.endswith('.csv'):
                filepath = os.path.join(bad_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_enhanced(df)
                    X.append(features)
                    y.append(0)  # BAD = 0
                    bad_count += 1
                    print(f"✓ {filename:45}")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n" + "="*80)
    print(f"TRAINING DATA SUMMARY")
    print(f"="*80)
    print(f"├─ GOOD datasets:  {good_count}")
    print(f"├─ BAD datasets:   {bad_count}")
    print(f"├─ TOTAL samples:  {len(X)}")
    print(f"├─ Features:       {X.shape[1]} (original 6 + 4 new)")
    print(f"└─ Ratio:          {good_count:.0f}:{bad_count:.0f}")
    
    # Train model with tuned hyperparameters
    print(f"\nTraining Random Forest with tuned hyperparameters...")
    print(f"├─ n_estimators: 120 (↑ from 50)")
    print(f"├─ max_depth: 5 (↑ from 3)")
    print(f"├─ min_samples_split: 2 (↑ from 5)")
    print(f"└─ Features: 10 (↑ from 6, normalized)")
    
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_results = cross_validate(
        model,
        X, y,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=True
    )
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"├─ Accuracy:  {cv_results['test_accuracy'].mean():.1%} (±{cv_results['test_accuracy'].std():.1%})")
    print(f"├─ Precision: {cv_results['test_precision'].mean():.1%} (±{cv_results['test_precision'].std():.1%})")
    print(f"├─ Recall:    {cv_results['test_recall'].mean():.1%} (±{cv_results['test_recall'].std():.1%})")
    print(f"└─ F1:        {cv_results['test_f1'].mean():.1%} (±{cv_results['test_f1'].std():.1%})")
    
    # Train on full data
    model.fit(X, y)
    
    # Feature importance
    print(f"\nFeature Importance:")
    feature_names = [
        'missing_ratio', 'duplicate_ratio', 'numeric_ratio', 'constant_cols', 
        'norm_variance', 'skewness',
        'type_consistency', 'cardinality_ratio', 'skewness_variance', 'mean_kurtosis'
    ]
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"├─ {name:20} {imp*100:5.1f}%")
    
    # Save model and scaler
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/feature_scaler.pkl'
    os.makedirs('models', exist_ok=True)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return
    
    print(f"\n" + "="*80)
    print(f"✓ ENHANCED MODEL TRAINED")
    print(f"="*80)
    
    acc = cv_results['test_accuracy'].mean()
    print(f"\nAccuracy Improvements:")
    print(f"  • Baseline (real only, 6 features):       84.0%")
    print(f"  • Enhanced (syn GOOD + 10 features):      {acc*100:.1f}%")
    print(f"  • Improvement:                            +{(acc*100 - 84.0):.1f}%")
    
    if acc >= 0.867:
        print(f"\n✅ TARGET REACHED! (≥86.7%)")
    else:
        gap = 86.7 - (acc * 100)
        print(f"\n⚠️  Gap to target: {gap:.1f} points")

if __name__ == '__main__':
    main()
