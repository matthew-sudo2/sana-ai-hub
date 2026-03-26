"""
Extract quality features from REAL datasets in traindata/raw
This replaces synthetic-only training with real-world data.

Analyzes: Titanic, Airbnb NYC, Telecom Churn, Adult Census, Pokemon, 
          COVID Vaccinations, E-commerce, Accenture Stock, Diabetes, Survey, etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from features.quality_features import extract_quality_features


def analyze_dataset(csv_path: Path) -> dict:
    """Analyze a single dataset and return quality metrics."""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='latin-1')
        
        if df.shape[0] == 0:
            return None
            
        features = extract_quality_features(df)
        
        return {
            'file': csv_path.name,
            'rows': df.shape[0],
            'cols': df.shape[1],
            'missing_ratio': features[0, 0],
            'duplicate_ratio': features[0, 1],
            'numeric_ratio': features[0, 2],
            'constant_columns': features[0, 3],
            'avg_variance': features[0, 4],
            'avg_skewness': features[0, 5],
            'features': features
        }
    except Exception as e:
        print(f"  ⚠️  {csv_path.name}: {str(e)[:50]}")
        return None


def label_dataset_quality(metrics: dict) -> int:
    """
    Intelligently label dataset as good (1) or bad (0).
    
    Good: Low missing %, low duplicates, well-structured
    Bad: High missing %, duplicates, inconsistent formatting
    """
    
    filename = metrics['file'].lower()
    
    # Check for known bad keywords - automatically mark as bad
    bad_keywords = ['messy', 'dirty', 'corrupt', 'real_world']
    if any(keyword in filename for keyword in bad_keywords):
        return 0  # Bad quality
    
    score = 100  # Start with perfect score
    
    # Missing values penalty (high = bad)
    if metrics['missing_ratio'] > 0.2:
        score -= 30  # High missing data
    elif metrics['missing_ratio'] > 0.1:
        score -= 15
    elif metrics['missing_ratio'] > 0.05:
        score -= 5
    
    # Duplicates penalty (high = bad)
    if metrics['duplicate_ratio'] > 0.1:
        score -= 20
    elif metrics['duplicate_ratio'] > 0.05:
        score -= 10
    
    # Consistency: too many constant columns = bad
    if metrics['constant_columns'] > metrics['cols'] * 0.2:
        score -= 15
    
    # Known problematic datasets (manually verified)
    bad_datasets = {
        'AB_NYC_2019.csv': 0,  # High missing values, ~30% missing
        'country_vaccinations.csv': 0,  # Missing dates, inconsistent records
        'fifa21 raw data v2.csv': 0,  # Raw/incomplete data with inconsistencies
        'restaurant_sales_data.csv': 0,  # Missing Item names, empty Price/Quantity, incomplete Payment Method
        'WA_Fn-UseC_-Telco-Customer-Churn.csv': 1,  # Actually well-maintained, slight issues
    }
    
    if metrics['file'] in bad_datasets:
        return bad_datasets[metrics['file']]
    
    # Decision: score >= 70 = good, < 70 = bad
    return 1 if score >= 70 else 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXTRACTING QUALITY FEATURES FROM REAL DATASETS")
    print("="*80)

    raw_dir = Path("traindata/raw")
    print(f"\n📂 Loading from: {raw_dir.absolute()}")

    # Find all CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    print(f"📊 Found {len(csv_files)} datasets\n")

    # Analyze all datasets
    all_metrics = []
    print("Analyzing datasets:")
    print(f"{'File':<40} {'Rows':<8} {'Cols':<6} {'Missing%':<10} {'Quality':<8}")
    print("-" * 92)

    for csv_file in sorted(csv_files):
        metrics = analyze_dataset(csv_file)
        
        if metrics:
            all_metrics.append(metrics)
            quality_label = label_dataset_quality(metrics)
            quality_str = "✓ GOOD" if quality_label == 1 else "✗ BAD"
            missing_pct = metrics['missing_ratio'] * 100
            
            print(f"{csv_file.name:<40} {metrics['rows']:<8} {metrics['cols']:<6} {missing_pct:<10.1f} {quality_str:<8}")

    print(f"\n✓ Analyzed {len(all_metrics)} datasets successfully\n")

    # Extract features and labels
    X_list = []
    y_list = []

    print("Creating training data:")
    print(f"{'Dataset':<40} {'Label':<10} {'Feature Vector':<50}")
    print("-" * 100)

    for metrics in all_metrics:
        label = label_dataset_quality(metrics)
        y_list.append(label)
        X_list.append(metrics['features'].flatten())
        
        label_str = "Good (1)" if label == 1 else "Bad (0)"
        feature_str = f"[{metrics['features'][0, 0]:.3f}, {metrics['features'][0, 1]:.3f}, {metrics['features'][0, 2]:.3f}, ...]"
        print(f"{metrics['file']:<40} {label_str:<10} {feature_str:<50}")

    X_all = np.vstack(X_list)
    y_all = np.array(y_list)

    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(y_all)}")
    print(f"   Good quality: {np.sum(y_all==1)} datasets")
    print(f"   Bad quality:  {np.sum(y_all==0)} datasets")
    print(f"   Feature matrix shape: {X_all.shape}")

    # Save to disk
    data_dir = Path("data/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)

    good_idx = y_all == 1
    bad_idx = y_all == 0

    if np.sum(good_idx) > 0:
        np.save(data_dir / "good_quality_features_real.npy", X_all[good_idx])
        print(f"\n✓ Saved good quality features: {X_all[good_idx].shape}")

    if np.sum(bad_idx) > 0:
        np.save(data_dir / "bad_quality_features_real.npy", X_all[bad_idx])
        print(f"✓ Saved bad quality features: {X_all[bad_idx].shape}")

    # Save combined dataset
    combined_data = {
        'X': X_all,
        'y': y_all,
        'feature_names': ['missing_ratio', 'duplicate_ratio', 'numeric_ratio', 
                          'constant_columns', 'avg_variance', 'avg_skewness']
    }

    import pickle
    with open(data_dir / "combined_real_data.pkl", 'wb') as f:
        pickle.dump(combined_data, f)
        print(f"✓ Saved combined dataset: {data_dir / 'combined_real_data.pkl'}")

    print("\n" + "="*80)
    print("✅ Real data preparation complete!")
    print("="*80)
    print(f"\nNext: Run models/train_quality_model.ipynb with real data")
    print(f"Expected improvement: 80-85% K-fold CV accuracy (no 100% overfitting)")
