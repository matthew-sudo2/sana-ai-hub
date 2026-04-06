#!/usr/bin/env python3
"""
Test current best_model.pkl using K-Fold cross-validation.
Provides robust performance estimate across multiple data splits.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)

print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION: CURRENT best_model.pkl")
print("="*80)

# Load model
print("\n[1] Loading best_model.pkl...")
model_path = Path("models/best_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"  ✓ Model loaded: {type(model).__name__}")

# Load training data
print("\n[2] Loading training data...")
training_data_path = Path("data/synthetic/training_data_8features.pkl")
with open(training_data_path, 'rb') as f:
    data = pickle.load(f)
    X = data['X']
    y = data['y']

print(f"  ✓ Loaded {len(X)} samples with {X.shape[1]} features")
print(f"    Class distribution: {np.bincount(y.astype(int))}")

# K-Fold Cross-Validation
print("\n[3] Running 5-fold cross-validation...")
print("-" * 80)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 1

cv_accuracies = []
cv_f1s = []
cv_balanced_accs = []
cv_aucs = []

for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train on fold
    model_fold = type(model)(**model.get_params())  # Create new model with same params
    model_fold.fit(X_train, y_train)
    
    # Evaluate on test fold
    y_pred = model_fold.predict(X_test)
    y_pred_proba = model_fold.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    cv_accuracies.append(acc)
    cv_f1s.append(f1)
    cv_balanced_accs.append(bal_acc)
    cv_aucs.append(auc)
    
    print(f"\nFold {fold_num}:")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  Balanced Acc:  {bal_acc:.4f}")
    print(f"  AUC-ROC:       {auc:.4f}")
    print(f"  Test samples:  {len(X_test)}")
    
    fold_num += 1

# Aggregate results
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("="*80)

print(f"\nAccuracy:")
print(f"  Mean:  {np.mean(cv_accuracies):.4f}")
print(f"  Std:   {np.std(cv_accuracies):.4f}")
print(f"  Range: {np.min(cv_accuracies):.4f} - {np.max(cv_accuracies):.4f}")

print(f"\nF1 Score:")
print(f"  Mean:  {np.mean(cv_f1s):.4f}")
print(f"  Std:   {np.std(cv_f1s):.4f}")
print(f"  Range: {np.min(cv_f1s):.4f} - {np.max(cv_f1s):.4f}")

print(f"\nBalanced Accuracy:")
print(f"  Mean:  {np.mean(cv_balanced_accs):.4f}")
print(f"  Std:   {np.std(cv_balanced_accs):.4f}")
print(f"  Range: {np.min(cv_balanced_accs):.4f} - {np.max(cv_balanced_accs):.4f}")

print(f"\nAUC-ROC:")
print(f"  Mean:  {np.mean(cv_aucs):.4f}")
print(f"  Std:   {np.std(cv_aucs):.4f}")
print(f"  Range: {np.min(cv_aucs):.4f} - {np.max(cv_aucs):.4f}")

# Also do full model evaluation on all data for reference
print("\n" + "="*80)
print("FULL DATASET EVALUATION (Fitted on all data)")
print("="*80)

model.fit(X, y)
y_pred_all = model.predict(X)
y_pred_proba_all = model.predict_proba(X)[:, 1]

print(f"\nAccuracy:      {accuracy_score(y, y_pred_all):.4f}")
print(f"F1 Score:      {f1_score(y, y_pred_all):.4f}")
print(f"Balanced Acc:  {balanced_accuracy_score(y, y_pred_all):.4f}")
print(f"AUC-ROC:       {roc_auc_score(y, y_pred_proba_all):.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"""
K-Fold results show how well the model generalizes:
  - Low std deviation (< 0.05) = Stable, consistent performance
  - High std deviation (> 0.10) = Unstable, variable performance
  
Your model K-Fold accuracy std: {np.std(cv_accuracies):.4f}
  -> This indicates {'STABLE' if np.std(cv_accuracies) < 0.05 else 'MODERATE' if np.std(cv_accuracies) < 0.10 else 'VARIABLE'} generalization

The full dataset evaluation (fitted on all data) represents upper bound
of performance but can overfit. K-Fold is more realistic.
""")
print("="*80 + "\n")
