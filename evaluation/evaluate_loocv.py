"""
Leave-One-Out Cross-Validation on Real Datasets.
More suitable for small datasets (11 samples).
Tests if model generalizes without synthetic data.

Usage:
    python evaluate_loocv.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))

from models.train.retrain_with_real_data import extract_features


def load_all_real_data():
    """Load all GOOD and BAD datasets"""
    
    X = []
    y = []
    dataset_names = []
    
    print("[1] Loading real labeled datasets...\n")
    
    # Load GOOD
    good_dir = Path('data/labeled/good')
    good_files = [f for f in good_dir.glob('*.csv') if f.name != 'Sample - Superstore.csv']
    
    for csv_file in sorted(good_files):
        try:
            df = pd.read_csv(csv_file)
            if len(df) >= 3 and len(df.columns) >= 2:
                features = extract_features(df)
                X.append(features)
                y.append(1)
                dataset_names.append(csv_file.name)
                print(f"  ✓ {csv_file.name:<40} GOOD")
        except Exception as e:
            print(f"  ✗ {csv_file.name} - {type(e).__name__}")
    
    # Load BAD
    bad_dir = Path('data/labeled/bad')
    for csv_file in sorted(bad_dir.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file)
            if len(df) >= 3 and len(df.columns) >= 2:
                features = extract_features(df)
                X.append(features)
                y.append(0)
                dataset_names.append(csv_file.name)
                print(f"  ✓ {csv_file.name:<40} BAD")
        except Exception as e:
            print(f"  ✗ {csv_file.name} - {type(e).__name__}")
    
    print(f"\nLoaded {len(X)} datasets: {sum(1 for l in y if l==1)} GOOD, {sum(1 for l in y if l==0)} BAD")
    
    return np.array(X), np.array(y), dataset_names


def loocv_on_real_data():
    """Leave-One-Out Cross-Validation using ONLY real data"""
    
    print("=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION: Real Data Only")
    print("=" * 80)
    
    X, y, names = load_all_real_data()
    n = len(X)
    
    print(f"\n[2] Running LOOCV ({n} iterations)...")
    print("(Train on N-1 real datasets, test on 1 held-out dataset)\n")
    
    correct = 0
    results = []
    
    for i in range(n):
        # Leave one out
        X_train = np.vstack([X[:i], X[i+1:]])
        y_train = np.hstack([y[:i], y[i+1:]])
        X_test = X[i:i+1]
        y_test = y[i]
        test_name = names[i]
        
        # Train on real data only
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)[0]
        confidence = model.predict_proba(X_test)[0].max()
        
        is_correct = (y_pred == y_test)
        if is_correct:
            correct += 1
        
        true_class = 'GOOD' if y_test == 1 else 'BAD'
        pred_class = 'GOOD' if y_pred == 1 else 'BAD'
        symbol = '✓' if is_correct else '✗'
        
        results.append({
            'name': test_name,
            'true': true_class,
            'pred': pred_class,
            'confidence': confidence,
            'correct': is_correct
        })
        
        print(f"{symbol} Iter {i+1:2d}: {test_name:<40} Pred: {pred_class:<5} ({confidence*100:5.1f}%) True: {true_class}")
    
    print("\n" + "=" * 80)
    print("LOOCV SUMMARY (Real Data Only)")
    print("=" * 80)
    
    accuracy = correct / n
    correct_good = sum(1 for r in results if r['true']=='GOOD' and r['correct'])
    correct_bad = sum(1 for r in results if r['true']=='BAD' and r['correct'])
    total_good = sum(1 for r in results if r['true']=='GOOD')
    total_bad = sum(1 for r in results if r['true']=='BAD')
    
    print(f"\nOverall Accuracy: {correct}/{n} ({accuracy*100:.1f}%)")
    print(f"\nPer-Class Performance:")
    print(f"  GOOD: {correct_good}/{total_good} ({correct_good/total_good*100:.1f}%)")
    print(f"  BAD:  {correct_bad}/{total_bad} ({correct_bad/total_bad*100:.1f}%)")
    
    if accuracy >= 0.90:
        print(f"\n✅ VALIDATED: {accuracy*100:.1f}% accuracy confirmed!")
    elif accuracy >= 0.80:
        print(f"\n⚠️  Borderline: {accuracy*100:.1f}% is close to 90%")
    else:
        print(f"\n❌ Real data only: {accuracy*100:.1f}% (below 90%)")
    
    print("=" * 80)
    
    # Detailed breakdown
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"  {status} {r['name']:<40} Pred: {r['pred']:<5} ({r['confidence']*100:5.1f}%) True: {r['true']}")


if __name__ == '__main__':
    loocv_on_real_data()
