"""
Test model on all real labeled datasets.

This script:
1. Loads the best model
2. Tests on all datasets in data/labeled/good and data/labeled/bad
3. Computes accuracy on real data
4. Shows which predictions are correct/incorrect

Usage:
    python evaluation/test_all_real_datasets.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train.retrain_with_real_data import extract_features


def test_all_datasets():
    """Test model on all real labeled datasets"""
    
    print("=" * 80)
    print("REAL-WORLD VALIDATION: Testing on All Labeled Datasets")
    print("=" * 80)
    
    # Load model
    print("\n[1] Loading model...")
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("    ✓ Model loaded")
    except FileNotFoundError:
        print("    ✗ Model not found! Run retrain_with_real_data.py first.")
        return
    
    # Test on all datasets
    print("\n[2] Testing on all datasets...\n")
    
    results = []
    
    # Test GOOD datasets
    good_dir = Path('data/labeled/good')
    if good_dir.exists():
        for csv_file in sorted(good_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_features(df)
                    
                    pred = model.predict([features])[0]
                    conf = model.predict_proba([features])[0].max()
                    
                    true_label = 1  # GOOD
                    pred_label = pred
                    correct = pred_label == true_label
                    symbol = '✓' if correct else '✗'
                    
                    pred_class = 'GOOD' if pred == 1 else 'BAD'
                    
                    results.append({
                        'name': csv_file.name,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': conf,
                        'correct': correct
                    })
                    
                    print(f"{symbol} {csv_file.name:<40} | "
                          f"Pred: {pred_class:<5} ({conf*100:5.1f}%) | True: GOOD")
            except Exception as e:
                print(f"✗ {csv_file.name:<40} | Error: {e}")
    
    # Test BAD datasets
    bad_dir = Path('data/labeled/bad')
    if bad_dir.exists():
        for csv_file in sorted(bad_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_features(df)
                    
                    pred = model.predict([features])[0]
                    conf = model.predict_proba([features])[0].max()
                    
                    true_label = 0  # BAD
                    pred_label = pred
                    correct = pred_label == true_label
                    symbol = '✓' if correct else '✗'
                    
                    pred_class = 'GOOD' if pred == 1 else 'BAD'
                    
                    results.append({
                        'name': csv_file.name,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': conf,
                        'correct': correct
                    })
                    
                    print(f"{symbol} {csv_file.name:<40} | "
                          f"Pred: {pred_class:<5} ({conf*100:5.1f}%) | True: BAD")
            except Exception as e:
                print(f"✗ {csv_file.name:<40} | Error: {e}")
    
    if len(results) == 0:
        print("\n⚠️  No datasets found! Add CSV files to:")
        print("    - data/labeled/good/")
        print("    - data/labeled/bad/")
        return
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    avg_confidence = np.mean([r['confidence'] for r in results])
    good_correct = sum(1 for r in results if r['true_label'] == 1 and r['correct'])
    bad_correct = sum(1 for r in results if r['true_label'] == 0 and r['correct'])
    good_total = sum(1 for r in results if r['true_label'] == 1)
    bad_total = sum(1 for r in results if r['true_label'] == 0)
    
    print(f"Total datasets tested: {len(results)}")
    print(f"Correct predictions: {sum(1 for r in results if r['correct'])}/{len(results)} ({accuracy*100:.1f}%)")
    print(f"Average confidence: {avg_confidence*100:.1f}%")
    print(f"\nGOOD datasets: {good_correct}/{good_total} correct ({good_correct/good_total*100:.1f}%)" if good_total > 0 else "")
    print(f"BAD datasets:  {bad_correct}/{bad_total} correct ({bad_correct/bad_total*100:.1f}%)" if bad_total > 0 else "")
    print("=" * 80)


if __name__ == '__main__':
    test_all_datasets()
