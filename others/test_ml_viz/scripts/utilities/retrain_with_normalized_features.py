#!/usr/bin/env python3
"""
Retrain model with normalized features (coefficient of variation for variance)
Fixes the scale-dependent variance issue that caused false BAD predictions
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def extract_features_normalized(df):
    """
    Extract 6 quality features with NORMALIZED VARIANCE
    Uses coefficient of variation instead of raw variance to handle different data scales
    """
    
    # Feature 1: Missing ratio
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    
    # Feature 2: Duplicate ratio
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    
    # Feature 3: Numeric ratio
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # Feature 4: Constant columns
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    # Feature 5 & 6: Coefficient of Variation (CV) and Skewness
    # CV = std_dev / mean (normalized by scale)
    cv_list = []
    skewness_list = []
    
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Only compute CV if mean is not zero
            if abs(mean_val) > 1e-10:
                cv = abs(std_val / mean_val)
                # Clip extreme CV values (data with very small mean can have huge CV)
                cv = min(cv, 100)  # Cap at 100 to prevent outliers
                cv_list.append(cv)
        except:
            pass
        
        try:
            skew = df[col].skew()
            if not pd.isna(skew):
                skewness_list.append(abs(skew))
        except:
            pass
    
    # Mean coefficient of variation (normalized variance)
    norm_variance = np.mean(cv_list) if cv_list else 0
    
    # Mean absolute skewness
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
    print("RETRAINING MODEL WITH NORMALIZED FEATURES")
    print("="*80)
    
    # Load all training data (11 real + 188 synthetic)
    real_datasets = [
        'data/labeled/good/bank_account_transactions.csv',
        'data/labeled/good/employee_records.csv',
        'data/labeled/good/games.csv',
        'data/labeled/good/Spotify.csv',
        'data/labeled/good/student_grades.csv',
        'data/labeled/good/synthetic_clean_transactions.csv',
        'data/labeled/bad/corruption_extreme_outliers.csv',
        'data/labeled/bad/corruption_heavy_missing.csv',
        'data/labeled/bad/corruption_inconsistent_columns.csv',
        'data/labeled/bad/corruption_many_duplicates.csv',
        'data/labeled/bad/corruption_mixed_issues.csv',
    ]
    
    X = []
    y = []
    
    # Load real datasets
    print("\nLoading real datasets...")
    for dataset_path in real_datasets:
        if not os.path.exists(dataset_path):
            print(f"⚠️  Missing: {dataset_path}")
            continue
        
        try:
            df = pd.read_csv(dataset_path)
            features_dict = extract_features_normalized(df)
            features = [
                features_dict['missing_ratio'],
                features_dict['duplicate_ratio'],
                features_dict['numeric_ratio'],
                features_dict['constant_cols'],
                features_dict['norm_variance'],
                features_dict['skewness']
            ]
            X.append(features)
            
            # Label: 1=GOOD, 0=BAD
            label = 1 if 'good' in dataset_path else 0
            y.append(label)
            print(f"✓ {Path(dataset_path).name:40} | Label: {'GOOD' if label==1 else 'BAD':4} | CV={features[4]:.4f}")
        except Exception as e:
            print(f"❌ {dataset_path}: {e}")
    
    # Load synthetic augmented data
    print("\nLoading synthetic augmented data...")
    synthetic_path = 'data/synthetic/augmented_data_multilevel.pkl'
    if os.path.exists(synthetic_path):
        try:
            with open(synthetic_path, 'rb') as f:
                aug_data = pickle.load(f)
            
            good_count = 0
            bad_count = 0
            
            for synthetic_df, label_str in aug_data:
                features_dict = extract_features_normalized(synthetic_df)
                features = [
                    features_dict['missing_ratio'],
                    features_dict['duplicate_ratio'],
                    features_dict['numeric_ratio'],
                    features_dict['constant_cols'],
                    features_dict['norm_variance'],
                    features_dict['skewness']
                ]
                X.append(features)
                
                label = 1 if label_str == 'good' else 0
                y.append(label)
                
                if label == 1:
                    good_count += 1
                else:
                    bad_count += 1
            
            print(f"✓ Synthetic data loaded: {good_count} good + {bad_count} bad = {good_count + bad_count} total")
        except Exception as e:
            print(f"⚠️  Could not load synthetic: {e}")
    
    # Prepare training data
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTraining Data Summary:")
    print(f"├─ Total samples: {len(X)}")
    print(f"├─ GOOD samples: {sum(y)}")
    print(f"├─ BAD samples: {len(y) - sum(y)}")
    print(f"└─ Features: {X.shape[1]}")
    
    # Train new model
    print(f"\nTraining Random Forest (normalized features)...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    print("✓ Model trained")
    
    # Cross-validation on real data only
    real_X = X[:11]
    real_y = y[:11]
    
    cv_scores = cross_val_score(model, real_X, real_y, cv=5)
    print(f"\nReal Data 5-Fold Cross-Validation:")
    print(f"├─ Scores: {[f'{s:.1%}' for s in cv_scores]}")
    print(f"└─ Mean: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    
    # Test on real data
    real_predictions = model.predict(real_X)
    real_accuracy = (real_predictions == real_y).mean()
    print(f"\nReal Data Test Accuracy: {real_accuracy:.1%}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    feature_names = ['missing_ratio', 'duplicate_ratio', 'numeric_ratio', 'constant_cols', 'norm_variance', 'skewness']
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"├─ {name:20} {imp*100:5.1f}%")
    
    # Save model
    model_path = 'models/best_model_normalized.pkl'
    os.makedirs('models', exist_ok=True)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return
    
    # Also overwrite the main model
    try:
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Updated main model: models/best_model.pkl (with normalized features)")
    except Exception as e:
        print(f"⚠️  Could not update main model: {e}")
    
    print("\n" + "="*80)
    print("✓ RETRAINING COMPLETE")
    print("="*80)
    print("\nNext: Run evaluate_user_datasets.py to test on your 3 clean datasets")

if __name__ == '__main__':
    main()
