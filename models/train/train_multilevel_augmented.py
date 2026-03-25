"""
Train on multi-level augmented data with 3 severity levels
Compare baseline (79%) vs Option A single-level (88.4%) vs Multi-level (TBD)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING ON MULTI-LEVEL AUGMENTED DATA")
print("=" * 80)

# Load multi-level augmented dataset
with open('data/synthetic/augmented_data_multilevel.pkl', 'rb') as f:
    aug_dict = pickle.load(f)

X = aug_dict['X']
y = aug_dict['y']

print(f"\n+Loaded multi-level augmented data: {len(X)} samples")
print(f"  Good: {sum(y == 1)} | Bad: {sum(y == 0)}")
print(f"  Shape: {X.shape}")

# Convert to numpy if needed
if isinstance(X, pd.DataFrame):
    X = X.values

# Engineer 13 features
def engineer_features(X_base):
    """Apply 13 features: 6 base + 7 engineered"""
    X_base = np.asarray(X_base)
    n_samples = X_base.shape[0]
    X_engineered = np.zeros((n_samples, 13))
    
    # First 6 features (base)
    X_engineered[:, :6] = X_base[:, :6]
    
    # Additional engineered features
    X_engineered[:, 6] = X_base[:, 0] * X_base[:, 1]   # F7: M*D
    X_engineered[:, 7] = X_base[:, 0] * X_base[:, 2]   # F8: M*N
    X_engineered[:, 8] = X_base[:, 4] * X_base[:, 5]   # F9: V*S
    X_engineered[:, 9] = np.log1p(X_base[:, 4])        # F10
    X_engineered[:, 10] = np.log1p(np.abs(X_base[:, 5]))# F11
    X_engineered[:, 11] = X_base[:, 4] / (np.abs(X_base[:, 5]) + 1e-10)  # F12
    X_engineered[:, 12] = np.abs(X_base[:, 5]) / (X_base[:, 4] + 1e-10)  # F13
    
    return X_engineered

X_eng = engineer_features(X)
print(f"\nEngineered 13 features: {X_eng.shape}")

# Train with cross-validation
print("\nCross-validation evaluation (5-fold StratifiedKFold)...")
rf = RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_eng, y, cv=cv, scoring='accuracy')

print(f"K-fold CV Scores: {[f'{s:.1%}' for s in scores]}")
print(f"K-fold CV: {scores.mean():.1%} ± {scores.std():.1%}")

# Compare vs baselines
baseline_cv = 0.790        # Original 23 samples, 13 features
single_level_cv = 0.884    # Option A: 95 samples (88.4%)
multilevel_cv = scores.mean()

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Baseline (23 original):              79.0%")
print(f"Option A (95 single-level):          88.4%  (+9.4% vs baseline)")
print(f"Multi-Level (188 varied severity):   {multilevel_cv:.1%}  ({multilevel_cv - baseline_cv:+.1%} vs baseline)")
print(f"                                               ({multilevel_cv - single_level_cv:+.1%} vs single-level)")
print("=" * 80)

# Decision logic
if multilevel_cv > single_level_cv:
    improvement = multilevel_cv - single_level_cv
    print(f"\n++ IMPROVEMENT: Multi-level corruption adds {improvement:.1%} more accuracy!")
    print(f"   Larger dataset with better diversity helps generalization")
else:
    degradation = single_level_cv - multilevel_cv
    print(f"\n-- TRADEOFF: -{degradation:.1%} but still above baseline (+{multilevel_cv - baseline_cv:.1%})")
    print(f"   Single-level approach is more optimal for this task")

print("\n+ Training complete.")
