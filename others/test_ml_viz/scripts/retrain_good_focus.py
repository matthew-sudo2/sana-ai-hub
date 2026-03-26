#!/usr/bin/env python3
"""
Use SYNTHETIC GOOD + REAL data (skip bad synthetics that cause bias)
This should improve on 84% real-only baseline
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
    print("OPTIMAL RETRAINING: SYNTHETIC GOOD + REAL ALL")
    print("="*80)
    
    X = []
    y = []
    
    # Load REAL GOOD datasets
    good_dir = 'data/labeled/good'
    print(f"\nLoading REAL GOOD datasets...")
    good_count = 0
    
    if os.path.exists(good_dir):
        for filename in os.listdir(good_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(good_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_normalized(df)
                    X.append(features)
                    y.append(1)
                    good_count += 1
                except:
                    pass
    
    print(f"✓ Real GOOD: {good_count} datasets")
    
    # Load REAL BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"Loading REAL BAD datasets...")
    bad_count = 0
    
    if os.path.exists(bad_dir):
        for filename in os.listdir(bad_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(bad_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    features = extract_features_normalized(df)
                    X.append(features)
                    y.append(0)
                    bad_count += 1
                except:
                    pass
    
    print(f"✓ Real BAD: {bad_count} datasets")
    
    # Load SYNTHETIC GOOD ONLY (skip BAD to avoid bias)
    print(f"\nLoading SYNTHETIC GOOD data only...")
    synthetic_path = 'data/synthetic/augmented_data_multilevel.pkl'
    syn_good_count = 0
    
    if os.path.exists(synthetic_path):
        try:
            with open(synthetic_path, 'rb') as f:
                synthetic_data = pickle.load(f)
            
            if isinstance(synthetic_data, dict) and 'X' in synthetic_data and 'y' in synthetic_data:
                X_syn = synthetic_data['X']
                y_syn = synthetic_data['y']
                
                # Add ONLY GOOD synthetic samples
                for i, label in enumerate(y_syn):
                    if label == 1:  # Only GOOD
                        X.append(X_syn[i])
                        y.append(1)
                        syn_good_count += 1
                
                print(f"✓ Synthetic GOOD: {syn_good_count} samples (skipped {sum(1 for l in y_syn if l == 0)} BAD)")
        except Exception as e:
            print(f"⚠️  Synthetic load error (non-critical): {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*80)
    print(f"FINAL TRAINING DATA (GOOD-FOCUSED)")
    print(f"="*80)
    print(f"├─ Real GOOD:      {good_count}")
    print(f"├─ Real BAD:       {bad_count}")
    print(f"├─ Synthetic GOOD: {syn_good_count}")
    print(f"├─ TOTAL:          {len(X)} samples")
    print(f"├─ GOOD:           {sum(1 for label in y if label == 1)}")
    print(f"├─ BAD:            {sum(1 for label in y if label == 0)}")
    print(f"└─ Features:       {X.shape[1]}")
    
    # Train model
    print(f"\nTraining Random Forest with {len(X)} samples...")
    model = RandomForestClassifier(
        n_estimators=150,  # More trees for stability
        max_depth=6,       # Slightly deeper
        max_features='sqrt',
        min_samples_split=2,
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
        print(f"❌ Failed to save: {e}")
        return
    
    print(f"\n" + "="*80)
    print(f"✓ FINAL OPTIMIZED MODEL")
    print(f"="*80)
    
    acc = cv_results['test_accuracy'].mean()
    print(f"\nAccuracy trajectory:")
    print(f"  • Real only (12):        80.0%")
    print(f"  • Real only (24):        84.0%")
    print(f"  • Syn GOOD + Real (all): {acc*100:.1f}%")
    
    if acc >= 0.867:
        print(f"\n✅ TARGET REACHED! (≥86.7%)")
    else:
        gap = 86.7 - (acc * 100)
        print(f"\n⚠️  Close to target! Gap: {gap:.1f} points")

if __name__ == '__main__':
    main()
