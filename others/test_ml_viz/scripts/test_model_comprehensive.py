#!/usr/bin/env python3
"""
Comprehensive testing of the 88.6% model
Tests on ALL datasets and reports detailed metrics
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    # Load model
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print('\n' + '='*90)
    print('COMPREHENSIVE MODEL TESTING - 88.6% ENHANCED MODEL')
    print('='*90)
    
    X_test = []
    y_test = []
    test_labels = []
    
    # Test on GOOD datasets
    good_dir = 'data/labeled/good'
    print(f"\nTesting GOOD datasets from {good_dir}...")
    good_tested = 0
    
    for filename in sorted(os.listdir(good_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(good_dir, filename)
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X_test.append(features)
                y_test.append(1)
                test_labels.append(('GOOD', filename))
                good_tested += 1
            except Exception as e:
                pass
    
    # Test on BAD datasets
    bad_dir = 'data/labeled/bad'
    print(f"Testing BAD datasets from {bad_dir}...")
    bad_tested = 0
    
    for filename in sorted(os.listdir(bad_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(bad_dir, filename)
            try:
                df = pd.read_csv(filepath)
                features = extract_features_optimized(df)
                X_test.append(features)
                y_test.append(0)
                test_labels.append(('BAD', filename))
                bad_tested += 1
            except Exception as e:
                pass
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n" + "="*90)
    print("TEST DATA SUMMARY")
    print("="*90)
    print(f"├─ GOOD datasets tested: {good_tested}")
    print(f"├─ BAD datasets tested:  {bad_tested}")
    print(f"├─ Total test samples:   {len(X_test)}")
    print(f"└─ Features:             8")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n" + "="*90)
    print("OVERALL PERFORMANCE METRICS")
    print("="*90)
    print(f"├─ Accuracy:  {accuracy*100:6.1f}%")
    print(f"├─ Precision: {precision*100:6.1f}%  (TP / (TP + FP))")
    print(f"├─ Recall:    {recall*100:6.1f}%  (TP / (TP + FN))")
    print(f"├─ F1 Score:  {f1*100:6.1f}%")
    print(f"└─ Support:   {len(y_test)} samples")
    
    print(f"\n" + "="*90)
    print("CONFUSION MATRIX")
    print("="*90)
    print(f"├─ True Negatives (TN):  {tn:3d}  (correctly predicted BAD)")
    print(f"├─ False Positives (FP): {fp:3d}  (incorrectly predicted GOOD when BAD)")
    print(f"├─ False Negatives (FN): {fn:3d}  (incorrectly predicted BAD when GOOD)")
    print(f"└─ True Positives (TP):  {tp:3d}  (correctly predicted GOOD)")
    
    # Detailed predictions
    print(f"\n" + "="*90)
    print("DETAILED PREDICTIONS")
    print("="*90)
    
    print(f"\n✓ GOOD DATASETS (Label=1):")
    print(f"  {'Status':<4} {'Filename':<50} {'Pred':<6} {'Conf %':<8}")
    print(f"  {'-'*4} {'-'*50} {'-'*6} {'-'*8}")
    
    good_correct = 0
    for i, (true_label, filename) in enumerate(test_labels):
        if true_label == 'GOOD':
            pred = y_pred[i]
            conf = y_pred_proba[i][int(pred)] * 100
            pred_label = 'GOOD' if pred == 1 else 'BAD'
            status = '✓' if pred == 1 else '✗'
            if pred == 1:
                good_correct += 1
            fname = filename[:46] + '..' if len(filename) > 48 else filename
            print(f"  {status:<4} {fname:<50} {pred_label:<6} {conf:6.1f}%")
    
    print(f"\n✗ BAD DATASETS (Label=0):")
    print(f"  {'Status':<4} {'Filename':<50} {'Pred':<6} {'Conf %':<8}")
    print(f"  {'-'*4} {'-'*50} {'-'*6} {'-'*8}")
    
    bad_correct = 0
    for i, (true_label, filename) in enumerate(test_labels):
        if true_label == 'BAD':
            pred = y_pred[i]
            conf = y_pred_proba[i][int(pred)] * 100
            pred_label = 'GOOD' if pred == 1 else 'BAD'
            status = '✓' if pred == 0 else '✗'
            if pred == 0:
                bad_correct += 1
            fname = filename[:46] + '..' if len(filename) > 48 else filename
            print(f"  {status:<4} {fname:<50} {pred_label:<6} {conf:6.1f}%")
    
    print(f"\n" + "="*90)
    print("PREDICTION ACCURACY BY CLASS")
    print("="*90)
    good_class_accuracy = good_correct / good_tested * 100 if good_tested > 0 else 0
    bad_class_accuracy = bad_correct / bad_tested * 100 if bad_tested > 0 else 0
    
    print(f"├─ GOOD class accuracy: {good_class_accuracy:6.1f}% ({good_correct}/{good_tested} correct)")
    print(f"├─ BAD class accuracy:  {bad_class_accuracy:6.1f}% ({bad_correct}/{bad_tested} correct)")
    print(f"└─ Overall accuracy:    {accuracy*100:6.1f}%")
    
    print(f"\n" + "="*90)
    print("✓ TESTING COMPLETE")
    print("="*90)
    
    # Summary
    print(f"\nModel Status:")
    if accuracy >= 0.867:
        print(f"  ✅ Exceeds target (≥86.7%): {accuracy*100:.1f}%")
    else:
        print(f"  ⚠️  Below target: {accuracy*100:.1f}%")
    
    if good_class_accuracy >= 80 and bad_class_accuracy >= 80:
        print(f"  ✅ Both classes well-detected (>80%)")
    else:
        print(f"  ⚠️  Unbalanced class performance")

if __name__ == '__main__':
    main()
