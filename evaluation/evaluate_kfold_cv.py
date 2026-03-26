"""
K-Fold Cross-Validation on Real Datasets.
Test if 90% accuracy is reliable or just lucky.

Usage:
    python evaluate_kfold_cv.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))

from models.train.retrain_with_real_data import extract_features


def load_all_real_data():
    """Load all GOOD and BAD datasets for k-fold evaluation"""
    
    X = []
    y = []
    dataset_names = []
    
    print("[1] Loading real labeled datasets...\n")
    
    # Load GOOD datasets
    good_dir = Path('data/labeled/good')
    good_files = [f for f in good_dir.glob('*.csv') if f.name != 'Sample - Superstore.csv']
    
    for csv_file in sorted(good_files):
        try:
            df = pd.read_csv(csv_file)
            if len(df) >= 3 and len(df.columns) >= 2:
                features = extract_features(df)
                X.append(features)
                y.append(1)
                dataset_names.append((csv_file.name, 'GOOD'))
                print(f"  ✓ {csv_file.name:<40} (GOOD)")
        except Exception as e:
            print(f"  ✗ {csv_file.name} - Error: {e}")
    
    # Load BAD datasets
    bad_dir = Path('data/labeled/bad')
    for csv_file in sorted(bad_dir.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file)
            if len(df) >= 3 and len(df.columns) >= 2:
                features = extract_features(df)
                X.append(features)
                y.append(0)
                dataset_names.append((csv_file.name, 'BAD'))
                print(f"  ✓ {csv_file.name:<40} (BAD)")
        except Exception as e:
            print(f"  ✗ {csv_file.name} - Error: {e}")
    
    print(f"\nLoaded {len(X)} datasets:")
    print(f"  GOOD: {sum(1 for label in y if label == 1)}")
    print(f"  BAD:  {sum(1 for label in y if label == 0)}")
    
    return np.array(X), np.array(y), dataset_names


def load_synthetic_data():
    """Load original synthetic training data"""
    try:
        with open('data/synthetic/augmented_data_multilevel.pkl', 'rb') as f:
            data = pickle.load(f)
            X_synthetic, y_synthetic = data['X'], data['y']
        print(f"Loaded {len(X_synthetic)} synthetic samples")
        return X_synthetic, y_synthetic
    except FileNotFoundError:
        print("Error: augmented_data_multilevel.pkl not found")
        return None, None


def kfold_cross_validate():
    """Perform k-fold cross-validation on real data"""
    
    print("=" * 80)
    print("K-FOLD CROSS-VALIDATION: Real Dataset Evaluation")
    print("=" * 80)
    
    # Load real data
    X_real, y_real, dataset_names = load_all_real_data()
    
    if len(X_real) < 5:
        print(f"\n⚠️  Only {len(X_real)} datasets - need at least 5 for meaningful k-fold")
        return
    
    # Load synthetic data for training
    print("\n[2] Loading synthetic training data...")
    X_synthetic, y_synthetic = load_synthetic_data()
    
    if X_synthetic is None:
        return
    
    # K-Fold setup
    k = min(5, len(X_real))  # Use k=5 or less if fewer datasets
    print(f"\n[3] Setting up {k}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_details = []
    
    # Perform k-fold
    print(f"\nRunning {k}-fold validation:\n")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real), 1):
        print(f"Fold {fold_idx}/{k}:")
        
        # Split real data
        X_train_real = X_real[train_idx]
        y_train_real = y_real[train_idx]
        X_test = X_real[test_idx]
        y_test = y_real[test_idx]
        
        # Combine synthetic + real training data
        X_train_combined = np.vstack([X_synthetic, X_train_real])
        y_train_combined = np.hstack([y_synthetic, y_train_real])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_train_combined, y_train_combined)
        
        # Evaluate
        score = model.score(X_test, y_test)
        fold_scores.append(score)
        
        # Get predictions for details
        y_pred = model.predict(X_test)
        correct = sum(y_pred == y_test)
        
        test_names = [dataset_names[i] for i in test_idx]
        fold_details.append({
            'fold': fold_idx,
            'score': score,
            'correct': correct,
            'total': len(y_test),
            'test_datasets': test_names,
            'predictions': list(zip([n[0] for n in test_names], y_test, y_pred))
        })
        
        print(f"  Training: {len(X_train_combined)} samples (synthetic + real)")
        print(f"  Testing:  {len(X_test)} datasets")
        print(f"  Accuracy: {score*100:.1f}% ({correct}/{len(y_test)} correct)")
        
        for name, true_y, pred_y in fold_details[-1]['predictions']:
            symbol = '✓' if true_y == pred_y else '✗'
            true_class = 'GOOD' if true_y == 1 else 'BAD'
            pred_class = 'GOOD' if pred_y == 1 else 'BAD'
            print(f"    {symbol} {name:<35} Pred: {pred_class:<5} True: {true_class}")
        print()
    
    # Summary
    print("=" * 80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nFold Scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score*100:6.1f}%")
    
    print(f"\nOverall Results:")
    print(f"  Mean Accuracy: {mean_score*100:.1f}%")
    print(f"  Std Dev:       {std_score*100:.1f}%")
    print(f"  Min Accuracy:  {min(fold_scores)*100:.1f}%")
    print(f"  Max Accuracy:  {max(fold_scores)*100:.1f}%")
    
    if mean_score >= 0.90:
        print(f"\n✅ CONFIRMED: 90% accuracy is REAL (not lucky!)")
        print(f"   Model reliably achieves {mean_score*100:.1f}% ± {std_score*100:.1f}%")
    elif mean_score >= 0.80:
        print(f"\n⚠️  Borderline: {mean_score*100:.1f}% is close to 90%")
        print(f"   Model is stable but slightly below target")
    else:
        print(f"\n❌ Below target: {mean_score*100:.1f}% is below 90%")
    
    print("=" * 80)


if __name__ == '__main__':
    kfold_cross_validate()
