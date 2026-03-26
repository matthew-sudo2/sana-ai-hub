#!/usr/bin/env python3
"""
Proper K-Fold Cross Validation
Train/test split to reveal true generalization accuracy
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def extract_features_optimized(df):
    """Extract 8 features (6 original + 2 new)"""
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    cv_list, skewness_list = [], []
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if abs(mean_val) > 1e-10:
                cv = min(abs(std_val / mean_val), 100)
                cv_list.append(cv)
        except: pass
        try:
            skew = df[col].skew()
            if not pd.isna(skew):
                skewness_list.append(abs(skew))
        except: pass
    
    norm_variance = np.mean(cv_list) if cv_list else 0
    skewness = np.mean(skewness_list) if skewness_list else 0
    
    cardinalities = [df[col].nunique() for col in df.columns]
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    cardinality_ratio = (avg_cardinality / len(df)) if len(df) > 0 else 0
    cardinality_ratio = min(cardinality_ratio, 1.0)
    
    kurtosis_list = []
    for col in numeric_cols:
        try:
            kurt = df[col].kurtosis()
            if not pd.isna(kurt):
                kurtosis_list.append(abs(kurt))
        except: pass
    
    mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
    mean_kurtosis = min(mean_kurtosis / 10, 1.0)
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
            norm_variance, skewness, cardinality_ratio, mean_kurtosis]

def main():
    print('\n' + '='*90)
    print('K-FOLD CROSS VALIDATION - TRUE GENERALIZATION TEST')
    print('='*90)
    
    X = []
    y = []
    
    # Load GOOD datasets (both real and synthetic)
    good_dir = 'data/labeled/good'
    print(f"\nLoading GOOD datasets...")
    good_count = 0
    
    for filename in sorted(os.listdir(good_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(good_dir, filename)
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X.append(features)
                y.append(1)
                good_count += 1
            except:
                pass
    
    # Load BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"Loading BAD datasets...")
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
            except:
                pass
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*90)
    print("DATA SUMMARY")
    print("="*90)
    print(f"├─ GOOD datasets: {good_count}")
    print(f"├─ BAD datasets:  {bad_count}")
    print(f"├─ Total:        {len(X)}")
    print(f"└─ Features:     {X.shape[1]}")
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        max_features='sqrt',
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 5-Fold Stratified Cross Validation
    print(f"\n" + "="*90)
    print("5-FOLD STRATIFIED CROSS VALIDATION")
    print("="*90)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train on this fold
        model.fit(X_train, y_train)
        
        # Test on held-out fold
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fold_results.append({
            'fold': fold_idx,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        print(f"\nFold {fold_idx}:")
        print(f"  ├─ Train size: {len(X_train)} | Test size: {len(X_test)}")
        print(f"  ├─ Accuracy:  {acc*100:6.1f}%")
        print(f"  ├─ Precision: {prec*100:6.1f}%")
        print(f"  ├─ Recall:    {rec*100:6.1f}%")
        print(f"  └─ F1 Score:  {f1*100:6.1f}%")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1_scores = [r['f1'] for r in fold_results]
    
    print(f"\n" + "="*90)
    print("CROSS-VALIDATION SUMMARY STATISTICS")
    print("="*90)
    print(f"├─ Accuracy:  {np.mean(accuracies)*100:6.1f}% (±{np.std(accuracies)*100:5.1f}%)")
    print(f"├─ Precision: {np.mean(precisions)*100:6.1f}% (±{np.std(precisions)*100:5.1f}%)")
    print(f"├─ Recall:    {np.mean(recalls)*100:6.1f}% (±{np.std(recalls)*100:5.1f}%)")
    print(f"└─ F1 Score:  {np.mean(f1_scores)*100:6.1f}% (±{np.std(f1_scores)*100:5.1f}%)")
    
    print(f"\n" + "="*90)
    print("COMPARISON")
    print("="*90)
    print(f"├─ Training on all data (no CV):     100.0% (overfitting on training set)")
    print(f"├─ K-Fold CV (proper validation):   {np.mean(accuracies)*100:6.1f}% (±{np.std(accuracies)*100:5.1f}%)")
    print(f"├─ Loss from overfitting:           {(1 - np.mean(accuracies))*100:6.1f}% drop")
    print(f"└─ Real generalization ability:     ~{np.mean(accuracies)*100:.1f}%")
    
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    
    print(f"\n" + "="*90)
    print("FOLD VARIANCE ANALYSIS")
    print("="*90)
    print(f"├─ Best fold:    {max_acc*100:6.1f}%")
    print(f"├─ Worst fold:   {min_acc*100:6.1f}%")
    print(f"├─ Spread:       {(max_acc - min_acc)*100:6.1f}%")
    print(f"└─ Stability:    {'Poor (high variance)' if np.std(accuracies) > 0.1 else 'Good (low variance)'}")
    
    print(f"\n" + "="*90)
    print("✓ K-FOLD CROSS VALIDATION TESTING COMPLETE")
    print("="*90)

if __name__ == '__main__':
    main()
