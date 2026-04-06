"""
Compare accuracy of all models in models/ folder on real labeled datasets.
Tests each model file and ranks them by performance.

Usage:
    python evaluation/compare_all_models.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train.retrain_with_real_data import extract_features


def load_all_models():
    """Load all models from models/ folder"""
    models_dir = Path("models")
    model_files = sorted(models_dir.glob("*.pkl"))
    
    models = {}
    for model_file in model_files:
        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            models[model_file.name] = model
            print(f"✓ Loaded: {model_file.name}")
        except Exception as e:
            print(f"✗ Failed to load {model_file.name}: {e}")
    
    return models


def load_labeled_datasets():
    """Load all labeled GOOD and BAD datasets"""
    X = []
    y = []
    dataset_names = []
    
    # Load GOOD datasets
    good_dir = Path("data/labeled/good")
    if good_dir.exists():
        for csv_file in sorted(good_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_features(df)
                    X.append(features)
                    y.append(1)
                    dataset_names.append(csv_file.name)
            except Exception as e:
                print(f"  ⚠ Could not load {csv_file.name}: {e}")
    
    # Load BAD datasets
    bad_dir = Path("data/labeled/bad")
    if bad_dir.exists():
        for csv_file in sorted(bad_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 3 and len(df.columns) >= 2:
                    features = extract_features(df)
                    X.append(features)
                    y.append(0)
                    dataset_names.append(csv_file.name)
            except Exception as e:
                print(f"  ⚠ Could not load {csv_file.name}: {e}")
    
    return np.array(X), np.array(y), dataset_names


def test_model(model, X, y, model_name):
    """Test a model and return metrics"""
    try:
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        correct = sum(y_pred == y)
        total = len(y)
        
        return {
            "name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "total": total,
            "error": None
        }
    except Exception as e:
        return {
            "name": model_name,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "correct": 0,
            "total": 0,
            "error": str(e)
        }


def main():
    print("=" * 80)
    print("MODEL COMPARISON: Testing all models in models/ folder")
    print("=" * 80)
    
    # Load models
    print("\n[1] Loading all models from models/ folder...\n")
    models = load_all_models()
    
    if not models:
        print("✗ No models found!")
        return
    
    print(f"\n✓ Loaded {len(models)} models\n")
    
    # Load datasets
    print("[2] Loading labeled datasets...\n")
    X, y, dataset_names = load_labeled_datasets()
    
    print(f"✓ Loaded {len(X)} datasets")
    print(f"  GOOD: {sum(1 for label in y if label == 1)}")
    print(f"  BAD:  {sum(1 for label in y if label == 0)}\n")
    
    # Test each model
    print("[3] Testing each model...\n")
    results = []
    
    for model_name, model in models.items():
        print(f"Testing: {model_name}...", end=" ")
        result = test_model(model, X, y, model_name)
        results.append(result)
        
        if result["error"]:
            print(f"✗ ERROR: {result['error']}")
        else:
            print(f"✓ {result['correct']}/{result['total']} ({result['accuracy']*100:.1f}%)")
    
    # Sort by accuracy
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS (sorted by accuracy)")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 100)
    
    for rank, result in enumerate(results, 1):
        if result["error"]:
            print(f"{rank:<6} {result['name']:<40} ERROR: {result['error'][:40]}")
        else:
            print(f"{rank:<6} {result['name']:<40} {result['accuracy']*100:>10.1f}%  "
                  f"{result['precision']*100:>10.1f}%  {result['recall']*100:>10.1f}%  "
                  f"{result['f1']*100:>10.1f}%")
    
    # Best model
    print("\n" + "=" * 80)
    best = results[0]
    if not best["error"]:
        print(f"🏆 BEST MODEL: {best['name']}")
        print(f"   Accuracy:  {best['accuracy']*100:.1f}% ({best['correct']}/{best['total']})")
        print(f"   Precision: {best['precision']*100:.1f}%")
        print(f"   Recall:    {best['recall']*100:.1f}%")
        print(f"   F1 Score:  {best['f1']*100:.1f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
