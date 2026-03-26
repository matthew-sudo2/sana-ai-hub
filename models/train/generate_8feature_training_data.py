"""
Generate 8-feature training data for ML quality model.
Creates training data compatible with MLQualityScorer's 8-feature extraction.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_8_features(df):
    """Extract 8 features matching MLQualityScorer exactly."""
    # Feature 1: Missing data ratio
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns)) if (len(df) * len(df.columns)) > 0 else 0
    
    # Feature 2: Duplicate row ratio
    duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
    
    # Feature 3: Numeric column ratio
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # Feature 4: Constant columns (only 1 unique value)
    constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    # Feature 5: Normalized variance (Coefficient of Variation)
    cv_list = []
    for col in numeric_cols:
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if abs(mean_val) > 1e-10:
                cv = min(abs(std_val / mean_val), 100)
                cv_list.append(cv)
        except:
            pass
    norm_variance = np.mean(cv_list) if cv_list else 0
    
    # Feature 6: Skewness
    skewness_list = []
    for col in numeric_cols:
        try:
            s = skew(df[col].dropna())
            skewness_list.append(abs(s))
        except:
            pass
    skewness = np.mean(skewness_list) if skewness_list else 0
    
    # Feature 7: Cardinality ratio (avg unique values / rows)
    cardinalities = [df[col].nunique() for col in df.columns] if len(df.columns) > 0 else []
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    cardinality_ratio = (avg_cardinality / len(df)) if len(df) > 0 else 0
    cardinality_ratio = min(cardinality_ratio, 1.0)
    
    # Feature 8: Mean Kurtosis (distribution shape)
    kurtosis_list = []
    for col in numeric_cols:
        try:
            kurt = kurtosis(df[col].dropna())
            if not np.isnan(kurt):
                kurtosis_list.append(abs(kurt))
        except:
            pass
    mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
    mean_kurtosis = min(mean_kurtosis / 10, 1.0)
    
    return [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols,
            norm_variance, skewness, cardinality_ratio, mean_kurtosis]


def augment_data(X, y, n_augmented=100):
    """
    Augment feature data by adding noise and perturbations.
    Creates synthetic variations of real feature vectors.
    """
    X_augmented = [X]  # Keep original
    y_augmented = [y]
    
    for _ in range(n_augmented // len(X)):
        # Small random perturbations (±5% noise)
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = np.clip(X + noise, 0, 100)  # Keep values in reasonable range
        X_augmented.append(X_noisy)
        y_augmented.append(y)
    
    return np.vstack(X_augmented), np.hstack(y_augmented)


def load_datasets():
    """Load good and bad datasets and extract 8 features."""
    X_features = []
    y_labels = []
    dataset_names = []
    
    print("\n[1] Loading real datasets and extracting 8 features...")
    
    # Load GOOD datasets (label = 1)
    good_dir = Path('data/labeled/good')
    good_count = 0
    if good_dir.exists():
        for csv_file in sorted(good_dir.glob('*.csv'))[:20]:  # Limit to 20 per category for speed
            try:
                df = pd.read_csv(csv_file, nrows=1000)  # Limit rows for speed
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_8_features(df)
                    X_features.append(features)
                    y_labels.append(1)
                    dataset_names.append((csv_file.name, 'GOOD'))
                    good_count += 1
            except Exception as e:
                pass
        print(f"  ✓ Loaded {good_count} GOOD datasets")
    
    # Load BAD datasets (label = 0)
    bad_dir = Path('data/labeled/bad')
    bad_count = 0
    if bad_dir.exists():
        for csv_file in sorted(bad_dir.glob('*.csv'))[:20]:
            try:
                df = pd.read_csv(csv_file, nrows=1000)
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_8_features(df)
                    X_features.append(features)
                    y_labels.append(0)
                    dataset_names.append((csv_file.name, 'BAD'))
                    bad_count += 1
            except Exception as e:
                pass
        print(f"  ✓ Loaded {bad_count} BAD datasets")
    
    return np.array(X_features), np.array(y_labels), dataset_names


def main():
    print("=" * 70)
    print("GENERATING 8-FEATURE TRAINING DATA FOR ML QUALITY MODEL")
    print("=" * 70)
    
    # Load real datasets
    X_real, y_real, dataset_names = load_datasets()
    
    if len(X_real) == 0:
        print("\n✗ No datasets found!")
        return
    
    print(f"  Loaded {len(X_real)} total datasets")
    print(f"    GOOD: {sum(y_real == 1)}")
    print(f"    BAD: {sum(y_real == 0)}")
    
    # Augment data to get enough training samples
    print("\n[2] Augmenting data with perturbations...")
    X_augmented, y_augmented = augment_data(X_real, y_real, n_augmented=200)
    print(f"  ✓ Augmented to {len(X_augmented)} samples (8 features each)")
    print(f"    Feature shape: {X_augmented.shape}")
    
    # Train model
    print("\n[3] Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validate
    cv_scores = cross_val_score(
        model, X_augmented, y_augmented,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    print(f"  K-fold CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    # Fit final model
    model.fit(X_augmented, y_augmented)
    
    # Feature importance
    print("\n[4] Feature importance:")
    feature_names = [
        'missing_ratio', 'duplicate_ratio', 'numeric_ratio', 'constant_cols',
        'norm_variance', 'skewness', 'cardinality_ratio', 'mean_kurtosis'
    ]
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"    {name:<20} {importance:.4f}")
    
    # Save model
    print("\n[5] Saving model...")
    model_path = Path('models/best_model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save augmented training data for reference
    print("\n[6] Saving training data...")
    training_data = {
        'X': X_augmented,
        'y': y_augmented,
        'feature_names': feature_names,
        'cv_score': float(cv_scores.mean())
    }
    training_data_path = Path('data/synthetic/training_data_8features.pkl')
    training_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"  ✓ Training data saved to {training_data_path}")
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS: Model trained and saved with 8-feature compatibility!")
    print("=" * 70)
    print(f"\nModel is now compatible with:")
    print(f"  - MLQualityScorer.extract_features() (8 features)")
    print(f"  - Feedback submission with 8-feature vectors")
    print(f"  - Continuous learning retraining pipeline")


if __name__ == "__main__":
    main()
