#!/usr/bin/env python3
"""
Smartly balance synthetic + real data to avoid bias
Use class weights and balanced sampling
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils.class_weight import compute_class_weight

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
    print("SMART RETRAINING: BALANCED SYNTHETIC + REAL DATA")
    print("="*80)
    
    X_real = []
    y_real = []
    
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
                    X_real.append(features)
                    y_real.append(1)
                    good_count += 1
                except:
                    pass
    
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
                    X_real.append(features)
                    y_real.append(0)
                    bad_count += 1
                except:
                    pass
    
    print(f"✓ Real data: {good_count} GOOD, {bad_count} BAD")
    
    # Load SYNTHETIC data
    print(f"\nLoading SYNTHETIC data...")
    synthetic_path = 'data/synthetic/augmented_data_multilevel.pkl'
    X_synthetic = []
    y_synthetic = []
    
    if os.path.exists(synthetic_path):
        try:
            with open(synthetic_path, 'rb') as f:
                synthetic_data = pickle.load(f)
            
            if isinstance(synthetic_data, dict) and 'X' in synthetic_data and 'y' in synthetic_data:
                X_synthetic = synthetic_data['X']
                y_synthetic = synthetic_data['y']
                
                good_syn = sum(1 for label in y_synthetic if label == 1)
                bad_syn = sum(1 for label in y_synthetic if label == 0)
                print(f"✓ Synthetic data: {good_syn} GOOD, {bad_syn} BAD (raw)")
                
                # SMART BALANCING: Downsample BAD synthetics to match GOOD ratio
                # Target ratio should be similar to real data or balanced
                good_indices = [i for i, label in enumerate(y_synthetic) if label == 1]
                bad_indices = [i for i, label in enumerate(y_synthetic) if label == 0]
                
                # Keep all GOOD synthetic, but downsample BAD to 1.5x GOOD ratio
                keep_bad = min(len(bad_indices), int(len(good_indices) * 1.5))
                bad_indices = np.random.choice(bad_indices, size=keep_bad, replace=False)
                
                selected_indices = good_indices + list(bad_indices)
                X_synthetic_balanced = [X_synthetic[i] for i in selected_indices]
                y_synthetic_balanced = [y_synthetic[i] for i in selected_indices]
                
                good_balanced = sum(1 for label in y_synthetic_balanced if label == 1)
                bad_balanced = sum(1 for label in y_synthetic_balanced if label == 0)
                print(f"✓ Balanced synthetic: {good_balanced} GOOD, {bad_balanced} BAD (downsampled)")
                
                X_synthetic = X_synthetic_balanced
                y_synthetic = y_synthetic_balanced
        except Exception as e:
            print(f"❌ Synthetic error: {e}")
    
    # Combine REAL + BALANCED SYNTHETIC
    X = np.array(X_real + X_synthetic)
    y = np.array(y_real + y_synthetic)
    
    print(f"\n" + "="*80)
    print(f"FINAL BALANCED TRAINING DATA")
    print(f"="*80)
    print(f"├─ REAL:      {good_count} GOOD, {bad_count} BAD")
    print(f"├─ SYNTHETIC: {sum(1 for label in y_synthetic if label == 1)} GOOD, {sum(1 for label in y_synthetic if label == 0)} BAD")
    print(f"├─ TOTAL:     {len(X)} samples")
    print(f"├─ GOOD:      {sum(1 for label in y if label == 1)}")
    print(f"├─ BAD:       {sum(1 for label in y if label == 0)}")
    print(f"└─ Ratio:     {sum(1 for label in y if label == 1):.0f}:{sum(1 for label in y if label == 0):.0f}")
    
    # Train model
    print(f"\nTraining Random Forest (balanced) with {len(X)} samples...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
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
    print(f"✓ BALANCED MODEL TRAINED")
    print(f"="*80)
    
    acc = cv_results['test_accuracy'].mean()
    target = 0.867
    
    print(f"\nAccuracy vs Target:")
    print(f"  • Target:      86.7%")
    print(f"  • Achieved:    {acc*100:.1f}%")
    
    if acc >= target:
        print(f"  ✅ TARGET REACHED!")
    else:
        print(f"  ⚠️ Gap: {(target - acc)*100:.1f} points")

if __name__ == '__main__':
    main()
