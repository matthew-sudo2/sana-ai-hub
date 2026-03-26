#!/usr/bin/env python3
"""
Detailed feature analysis for user datasets
Shows what features the model is seeing
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def extract_features_detailed(df, name):
    """Extract and display all 6 features in detail"""
    
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"{'='*70}")
    print(f"Shape: {len(df)} rows × {len(df.columns)} columns")
    
    # Feature 1: Missing ratio
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / (len(df) * len(df.columns))
    print(f"├─ Missing Ratio:       {missing_ratio:.6f} ({missing_count} cells / {len(df) * len(df.columns)} total)")
    
    # Feature 2: Duplicate ratio
    duplicate_count = len(df) - len(df.drop_duplicates())
    duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
    print(f"├─ Duplicate Ratio:     {duplicate_ratio:.6f} ({duplicate_count} rows / {len(df)} total)")
    
    # Feature 3: Numeric ratio
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    print(f"├─ Numeric Ratio:       {numeric_ratio:.6f} ({len(numeric_cols)} numeric / {len(df.columns)} total columns)")
    
    # Feature 4: Constant columns
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    print(f"├─ Constant Columns:    {constant_cols} columns (unique values ≤ 1)")
    
    # Feature 5: Variance
    variance_list = []
    for col in numeric_cols:
        try:
            col_var = df[col].var()
            if not pd.isna(col_var) and col_var > 0:
                variance_list.append(col_var)
        except:
            pass
    variance = np.mean(variance_list) if variance_list else 0
    print(f"├─ Mean Variance:       {variance:.6f} (from {len(variance_list)} numeric columns)")
    
    # Feature 6: Skewness
    skewness_list = []
    for col in numeric_cols:
        try:
            col_skew = df[col].skew()
            if not pd.isna(col_skew):
                skewness_list.append(abs(col_skew))
        except:
            pass
    skewness = np.mean(skewness_list) if skewness_list else 0
    print(f"└─ Mean |Skewness|:    {skewness:.6f} (from {len(skewness_list)} numeric columns)")
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, variance, skewness]

def main():
    datasets = [
        'AI_Impact_on_Jobs_2030.csv',
        'student_exam_scores.csv',
        'Exam_Score_Prediction.csv'
    ]
    
    # Load model
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("\n" + "="*70)
    print("DETAILED FEATURE ANALYSIS - User's Clean Datasets")
    print("="*70)
    
    predictions = []
    
    for dataset_name in datasets:
        if not os.path.exists(dataset_name):
            print(f"\n⚠️  {dataset_name} not found")
            continue
        
        df = pd.read_csv(dataset_name)
        features = extract_features_detailed(df, dataset_name)
        
        # Make prediction
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]
        pred_label = 'GOOD' if pred == 1 else 'BAD'
        confidence = max(prob) * 100
        
        predictions.append({
            'dataset': dataset_name,
            'pred': pred_label,
            'conf': confidence,
            'features': features
        })
        
        print(f"\n→ Model Prediction: {pred_label} ({confidence:.1f}% confidence)")
    
    # Summary comparison
    print(f"\n\n{'='*70}")
    print("COMPARISON WITH TRAINING DATA PATTERNS")
    print(f"{'='*70}")
    
    # Show bounds from training data
    print("\nTraining data feature ranges (from 11 real dataset + 188 synthetic):")
    print("  Missing Ratio:     0.0  - 0.5")
    print("  Duplicate Ratio:   0.0  - 0.2")
    print("  Numeric Ratio:     0.1  - 1.0")
    print("  Constant Cols:     0    - 3")
    print("  Variance:          0.1  - 100.0")
    print("  Skewness:          0.0  - 3.0")
    
    print("\nYour datasets feature ranges:")
    if predictions:
        for pred in predictions:
            print(f"\n{pred['dataset']}:")
            features_names = ['Missing', 'Duplicate', 'Numeric', 'Constant', 'Variance', 'Skewness']
            for name, val in zip(features_names, pred['features']):
                print(f"  {name:12} = {val:.6f}")

if __name__ == '__main__':
    main()
