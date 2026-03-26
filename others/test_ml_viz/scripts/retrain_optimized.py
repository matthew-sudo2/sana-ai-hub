#!/usr/bin/env python3
"""
Optimized retraining:
1. Use only REAL data + selected high-quality synthetic variants
2. Add only 2 best new features (cardinality_ratio, mean_kurtosis)
3. Keep original feature extraction (no normalization of all features)
4. Tune hyperparameters moderately
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def extract_features_optimized(df):
    """Extract original 6 features + 2 best new features"""
    
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
    
    # 2 NEW FEATURES (best ones)
    # Feature 7: Cardinality ratio (good data has balanced cardinality)
    cardinalities = [df[col].nunique() for col in df.columns]
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    cardinality_ratio = (avg_cardinality / len(df)) if len(df) > 0 else 0
    cardinality_ratio = min(cardinality_ratio, 1.0)  # Normalize to [0, 1]
    
    # Feature 8: Mean Kurtosis (good data has normal distributions)
    kurtosis_list = []
    for col in numeric_cols:
        try:
            kurt = df[col].kurtosis()
            if not pd.isna(kurt):
                kurtosis_list.append(abs(kurt))
        except:
            pass
    
    mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
    mean_kurtosis = min(mean_kurtosis / 10, 1.0)  # Normalize to ~[0, 1]
    
    return [
        missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
        norm_variance, skewness,
        cardinality_ratio, mean_kurtosis
    ]

def main():
    print("\n" + "="*80)
    print("OPTIMIZED RETRAINING: REAL + SELECTED SYNTHETIC")
    print("="*80)
    
    X = []
    y = []
    
    # Load REAL GOOD datasets
    good_dir = 'data/labeled/good'
    print(f"\nLoading REAL GOOD datasets...")
    real_good_count = 0
    
    real_good_files = [
        'AI_Impact_on_Jobs_2030.csv',
        'Amazon_stock_data.csv',
        'bank_account_transactions.csv',
        'employee_records.csv',
        'Exam_Score_Prediction.csv',
        'games.csv',
        'Sample - Superstore.csv',
        'Spotify.csv',
        'student_exam_scores.csv',
        'student_grades.csv',
        'synthetic_clean_transactions.csv'
    ]
    
    for filename in real_good_files:
        filepath = os.path.join(good_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X.append(features)
                y.append(1)
                real_good_count += 1
                print(f"✓ [REAL] {filename:45}")
            except Exception as e:
                print(f"❌ {filename}: {e}")
    
    # Add SELECTED SYNTHETIC GOOD (only bootstrap variants, not interpolate)
    print(f"\nLoading SELECTED SYNTHETIC GOOD datasets...")
    syn_good_count = 0
    
    for filename in os.listdir(good_dir):
        # Only include bootstrap variants (more stable than interpolate)
        if 'synthetic_good_bootstrap' in filename:
            filepath = os.path.join(good_dir, filename)
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X.append(features)
                y.append(1)
                syn_good_count += 1
                print(f"✓ [SYNTH] {filename:45}")
            except Exception as e:
                print(f"❌ {filename}: {e}")
    
    # Load BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"\nLoading BAD datasets...")
    bad_count = 0
    
    for filename in sorted(os.listdir(bad_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(bad_dir, filename)
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X.append(features)
                y.append(0)
                bad_count += 1
                print(f"✓ {filename:45}")
            except Exception as e:
                print(f"❌ {filename}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*80)
    print(f"TRAINING DATA SUMMARY")
    print(f"="*80)
    print(f"├─ REAL GOOD:        {real_good_count}")
    print(f"├─ SYNTHETIC GOOD:   {syn_good_count}")
    print(f"├─ BAD datasets:     {bad_count}")
    print(f"├─ TOTAL samples:    {len(X)}")
    print(f"├─ Ratio:            {real_good_count + syn_good_count}:{bad_count}")
    print(f"└─ Features:         {X.shape[1]} (6 original + 2 new)")
    
    # Train model
    print(f"\nTraining Random Forest (optimized tuning)...")
    print(f"├─ n_estimators: 100")
    print(f"├─ max_depth: 4")
    print(f"└─ min_samples_split: 3")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        max_features='sqrt',
        min_samples_split=3,
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
        'cardinality_ratio', 'mean_kurtosis'
    ]
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"├─ {name:20} {imp*100:5.1f}%")
    
    # Save model
    model_path = 'models/best_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return
    
    print(f"\n" + "="*80)
    print(f"✓ OPTIMIZED MODEL TRAINED")
    print(f"="*80)
    
    acc = cv_results['test_accuracy'].mean()
    print(f"\nAccuracy trajectory:")
    print(f"  • Baseline (real 24, 6 feat):           84.0%")
    print(f"  • Enhanced (real 11 + syn 11, 8 feat):  {acc*100:.1f}%")
    
    if acc >= 0.867:
        print(f"\n✅ TARGET REACHED! (≥86.7%)")
    else:
        gap = 86.7 - (acc * 100)
        print(f"\n⚠️  Gap to target: {gap:.1f} points")

if __name__ == '__main__':
    main()
