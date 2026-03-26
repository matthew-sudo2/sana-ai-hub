#!/usr/bin/env python3
"""
Test Amazon stock data with the retrained model
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

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
    
    return {
        'missing_ratio': missing_ratio,
        'duplicate_ratio': duplicate_ratio,
        'numeric_ratio': numeric_ratio,
        'constant_cols': constant_cols,
        'norm_variance': norm_variance,
        'skewness': skewness
    }

def main():
    print("\n" + "="*80)
    print("Amazon Stock Data - Quality Assessment")
    print("="*80)
    
    dataset_path = 'data_processing/quality/Amazon_stock_data.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}")
        print(f"\nSearching for the file...")
        import glob
        matches = glob.glob('**/Amazon_stock_data.csv', recursive=True)
        if matches:
            print(f"Found at: {matches[0]}")
            dataset_path = matches[0]
        else:
            print("Could not find Amazon_stock_data.csv")
            return
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"\n✓ Dataset loaded: {dataset_path}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Display dataset info
    print(f"\nDataset Info:")
    print(f"├─ Rows: {len(df)}")
    print(f"├─ Columns: {len(df.columns)}")
    print(f"├─ Column names: {', '.join(df.columns)}")
    print(f"├─ Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Check data quality
    missing_count = df.isnull().sum().sum()
    duplicate_count = len(df) - len(df.drop_duplicates())
    
    print(f"\nData Quality Overview:")
    print(f"├─ Missing values: {missing_count} ({missing_count/(len(df)*len(df.columns))*100:.2f}%)")
    print(f"├─ Duplicate rows: {duplicate_count}")
    print(f"├─ Data types:")
    for dtype in df.dtypes.unique():
        count = len(df.columns[df.dtypes == dtype])
        print(f"│  └─ {dtype}: {count} columns")
    
    # Extract features
    print(f"\nExtracting quality features...")
    features_dict = extract_features_normalized(df)
    
    print(f"\nQuality Features:")
    print(f"├─ Missing ratio:    {features_dict['missing_ratio']:.6f}")
    print(f"├─ Duplicate ratio:  {features_dict['duplicate_ratio']:.6f}")
    print(f"├─ Numeric ratio:    {features_dict['numeric_ratio']:.6f}")
    print(f"├─ Constant cols:    {features_dict['constant_cols']}")
    print(f"├─ Norm variance:    {features_dict['norm_variance']:.6f}")
    print(f"└─ Skewness:         {features_dict['skewness']:.6f}")
    
    # Load model
    print(f"\nLoading trained model...")
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Prepare features for model
    features = [
        features_dict['missing_ratio'],
        features_dict['duplicate_ratio'],
        features_dict['numeric_ratio'],
        features_dict['constant_cols'],
        features_dict['norm_variance'],
        features_dict['skewness']
    ]
    
    # Make prediction
    print(f"\nMaking prediction...")
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0]
    
    pred_label = 'GOOD' if prediction == 1 else 'BAD'
    pred_confidence = max(confidence) * 100
    
    print(f"\n" + "="*80)
    print(f"PREDICTION RESULT")
    print(f"="*80)
    print(f"\nStatus: {pred_label.upper()}")
    print(f"Confidence: {pred_confidence:.1f}%")
    print(f"Probability GOOD: {confidence[1]*100:.1f}%")
    print(f"Probability BAD: {confidence[0]*100:.1f}%")
    
    if pred_label == "GOOD":
        print(f"\n✓ Dataset quality looks good!")
        print(f"  This dataset appears to be clean with proper structure.")
    else:
        print(f"\n⚠️  Dataset may have quality issues")
        print(f"  Consider investigating:")
        if features_dict['missing_ratio'] > 0.1:
            print(f"  • High missing data ratio: {features_dict['missing_ratio']*100:.1f}%")
        if features_dict['duplicate_ratio'] > 0.1:
            print(f"  • High duplicate ratio: {features_dict['duplicate_ratio']*100:.1f}%")
        if features_dict['norm_variance'] > 5:
            print(f"  • High variance: {features_dict['norm_variance']:.2f}")
    
    print(f"\n" + "="*80)

if __name__ == '__main__':
    main()
