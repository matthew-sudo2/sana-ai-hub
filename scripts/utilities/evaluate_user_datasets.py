#!/usr/bin/env python3
"""
Evaluate user-provided real-world datasets
Tests if model correctly predicts GOOD quality datasets
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def extract_features(df):
    """Extract 6 quality features from dataframe (with normalized variance)"""
    try:
        # Feature 1: Missing ratio
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Feature 2: Duplicate ratio
        duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
        
        # Feature 3: Numeric ratio
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
        
        # Feature 4: Constant columns
        constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
        
        # Feature 5 & 6: Coefficient of Variation (normalized variance) and skewness
        # CV = std_dev / mean (independent of data scale)
        cv_list = []
        skewness_list = []
        
        for col in numeric_cols:
            try:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Only compute CV if mean is not zero
                if abs(mean_val) > 1e-10:
                    cv = abs(std_val / mean_val)
                    # Clip extreme CV values
                    cv = min(cv, 100)
                    cv_list.append(cv)
            except:
                pass
            
            try:
                col_skewness = df[col].skew()
                if not pd.isna(col_skewness):
                    skewness_list.append(abs(col_skewness))
            except:
                pass
        
        # Mean coefficient of variation (normalized variance)
        norm_variance = np.mean(cv_list) if cv_list else 0
        skewness = np.mean(skewness_list) if skewness_list else 0
        
        return {
            'missing_ratio': missing_ratio,
            'duplicate_ratio': duplicate_ratio,
            'numeric_ratio': numeric_ratio,
            'constant_cols': constant_cols,
            'norm_variance': norm_variance,
            'skewness': skewness
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def main():
    # Define datasets to test
    datasets = [
        'AI_Impact_on_Jobs_2030.csv',
        'student_exam_scores.csv',
        'Exam_Score_Prediction.csv'
    ]
    
    # Load trained model
    model_path = 'models/best_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test each dataset
    results = []
    print("=" * 80)
    print("Real-World Dataset Evaluation - User's Clean Datasets")
    print("=" * 80)
    
    for dataset_name in datasets:
        dataset_path = dataset_name
        
        if not os.path.exists(dataset_path):
            print(f"⚠️  {dataset_name:40} | Not found")
            continue
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Extract features
            features_dict = extract_features(df)
            if features_dict is None:
                print(f"⚠️  {dataset_name:40} | Failed to extract features")
                continue
            
            # Prepare features for model
            features = [
                features_dict['missing_ratio'],
                features_dict['duplicate_ratio'],
                features_dict['numeric_ratio'],
                features_dict['constant_cols'],
                features_dict['norm_variance'],
                features_dict['skewness']
            ]
            
            # Predict
            prediction = model.predict([features])[0]
            confidence = model.predict_proba([features])[0]
            
            pred_label = 'GOOD' if prediction == 1 else 'BAD'
            pred_confidence = max(confidence) * 100
            
            # Prepare output
            missing_pct = features_dict['missing_ratio'] * 100
            duplicates_pct = features_dict['duplicate_ratio'] * 100
            
            results.append({
                'dataset': dataset_name,
                'prediction': pred_label,
                'confidence': pred_confidence,
                'rows': len(df),
                'cols': len(df.columns),
                'missing': missing_pct
            })
            
            # Print result
            status = "✓" if pred_label == "GOOD" else "?"
            print(f"{status} {dataset_name:35} | Pred: {pred_label:4} ({pred_confidence:5.1f}%) | "
                  f"Rows: {len(df):5} | Cols: {len(df.columns):3} | Missing: {missing_pct:5.1f}%")
        
        except Exception as e:
            print(f"❌ {dataset_name:35} | Error: {str(e)[:40]}")
    
    # Summary
    print("\n" + "=" * 80)
    if results:
        good_count = sum(1 for r in results if r['prediction'] == 'GOOD')
        total_count = len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"Summary: {good_count}/{total_count} predicted as GOOD")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        print("=" * 80)
        
        if good_count == total_count:
            print("✓ All datasets correctly identified as GOOD!")
        else:
            print(f"⚠️  {total_count - good_count} dataset(s) predicted as BAD - may need investigation")
    else:
        print("No datasets could be evaluated")

if __name__ == '__main__':
    main()
