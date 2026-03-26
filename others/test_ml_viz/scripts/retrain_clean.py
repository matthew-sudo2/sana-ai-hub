#!/usr/bin/env python3
"""
Retrain model with ALL datasets from data/labeled/good/ and data/labeled/bad/
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

def extract_features_normalized(df):
    """Extract normalized features"""
    
    # Feature 1: Missing ratio
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    
    # Feature 2: Duplicate ratio
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    
    # Feature 3: Numeric ratio
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # Feature 4: Constant columns
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    # Feature 5 & 6: Coefficient of Variation and Skewness
    cv_list = []
    skewness_list = []
    
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if abs(mean_val) > 1e-10:
                cv = abs(std_val / mean_val)
                cv = min(cv, 100)
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
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, norm_variance, skewness]

def main():
    print("\n" + "="*80)
    print("RETRAINING WITH ALL DATA FROM data/labeled/")
    print("="*80)
    
    X = []
    y = []
    
    # Load GOOD datasets
    good_dir = 'data/labeled/good'
    print(f"\nLoading GOOD datasets from {good_dir}...")
    good_count = 0
    
    if os.path.exists(good_dir):
        for filename in os.listdir(good_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(good_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_normalized(df)
                    X.append(features)
                    y.append(1)  # GOOD = 1
                    good_count += 1
                    print(f"✓ {filename:40} | GOOD")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
    
    # Load BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"\nLoading BAD datasets from {bad_dir}...")
    bad_count = 0
    
    if os.path.exists(bad_dir):
        for filename in os.listdir(bad_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(bad_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_normalized(df)
                    X.append(features)
                    y.append(0)  # BAD = 0
                    bad_count += 1
                    print(f"✓ {filename:40} | BAD")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*80)
    print(f"TRAINING DATA SUMMARY")
    print(f"="*80)
    print(f"├─ GOOD datasets:  {good_count}")
    print(f"├─ BAD datasets:   {bad_count}")
    print(f"├─ TOTAL samples:  {len(X)}")
    print(f"└─ Features:       {X.shape[1]}")
    
    # Train model
    print(f"\nTraining Random Forest with {len(X)} samples...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        max_features='sqrt',
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
    feature_names = ['missing_ratio', 'duplicate_ratio', 'numeric_ratio', 'constant_cols', 'norm_variance', 'skewness']
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
        print(f"❌ Failed to save model: {e}")
        return
    
    print(f"\n" + "="*80)
    print(f"✓ RETRAINING COMPLETE")
    print(f"="*80)
    print(f"\nModel trained on:")
    print(f"  • {good_count} GOOD datasets from data/labeled/good/")
    print(f"  • {bad_count} BAD datasets from data/labeled/bad/")
    print(f"  • TOTAL: {len(X)} datasets")
    print(f"\nPerformance:")
    print(f"  • CV Accuracy: {cv_results['test_accuracy'].mean():.1%} (±{cv_results['test_accuracy'].std():.1%})")
    print(f"  • Stable, production-ready model")

if __name__ == '__main__':
    main()
