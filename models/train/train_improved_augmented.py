"""
Train on IMPROVED augmented data with realistic corruption patterns (Option A)
Uses 13 engineered features to evaluate if realistic patterns improve accuracy
Compare: Phase 1B (74.7%) vs Option A (improved) = TBD
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Load improved augmented dataset
print("=" * 80)
print("OPTION A: TRAINING ON IMPROVED AUGMENTED DATA")
print("=" * 80)

with open('data/synthetic/augmented_data_improved.pkl', 'rb') as f:
    aug_dict = pickle.load(f)

X_improved = aug_dict['X']
y_improved = aug_dict['y']

print(f"\n✓ Loaded improved augmented data: {len(X_improved)} samples")
print(f"  Good: {sum(y_improved == 1)} | Bad: {sum(y_improved == 0)}")
print(f"  Features shape: {X_improved.shape}")
print(f"  Type: {type(X_improved)}")

# Convert to numpy array if needed
if isinstance(X_improved, pd.DataFrame):
    X_improved = X_improved.values

# Feature engineering (13 features - same as baseline)
print("\n📊 Engineering 13 features...")

def engineer_features(X_base):
    """Apply 13 features: 6 base + 7 engineered"""
    X_base = np.asarray(X_base)
    n_samples = X_base.shape[0]
    X_engineered = np.zeros((n_samples, 13))
    
    # First 6 features (base)
    X_engineered[:, :6] = X_base[:, :6]
    
    # Additional engineered features
    X_engineered[:, 6] = X_base[:, 0] * X_base[:, 1]  # F7: M×D
    X_engineered[:, 7] = X_base[:, 0] * X_base[:, 2]  # F8: M×N
    X_engineered[:, 8] = X_base[:, 4] * X_base[:, 5]  # F9: V×S
    X_engineered[:, 9] = np.log1p(X_base[:, 4])  # F10
    X_engineered[:, 10] = np.log1p(np.abs(X_base[:, 5]))  # F11
    X_engineered[:, 11] = X_base[:, 4] / (np.abs(X_base[:, 5]) + 1e-10)  # F12
    X_engineered[:, 12] = np.abs(X_base[:, 5]) / (X_base[:, 4] + 1e-10)  # F13
    
    return X_engineered

X_improved_eng = engineer_features(X_improved)
print(f"✓ Engineered features shape: {X_improved_eng.shape}")

# Train with cross-validation
print("\n🔄 Cross-validation evaluation (5-fold)...")
rf = RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_improved_eng, y_improved, cv=cv, scoring='accuracy')

print(f"K-fold CV Scores: {[f'{s:.1%}' for s in scores]}")
print(f"K-fold CV: {scores.mean():.1%} ± {scores.std():.1%}")

# Compare vs baselines
baseline_cv = 0.790  # Original 23 samples, 13 features
phase1b_cv = 0.747   # Phase 1B: 111 samples (88 basic corrupted), 13 features
improved_cv = scores.mean()

print("\n📈 COMPARISON")
print("=" * 80)
print(f"Baseline (23 original samples):      {baseline_cv:.1%}")
print(f"Phase 1B (basic corruption):          {phase1b_cv:.1%}  (degraded -4.3%)")
print(f"Option A (improved corruption):       {improved_cv:.1%}")
print(f"  vs Phase 1B:                        {improved_cv - phase1b_cv:+.1%}")
print(f"  vs Baseline:                        {improved_cv - baseline_cv:+.1%}")
print("=" * 80)

# Decision logic
if improved_cv >= 0.77:
    print("\n✅ SUCCESS: Realistic patterns significantly improved accuracy!")
    print(f"   Improvement: {improved_cv - phase1b_cv:+.1%} vs Phase 1B")
    print("   Recommendation: Option A is viable, more stable than baseline")
elif improved_cv >= 0.75:
    print("\n⚠️  MARGINAL: Realistic patterns provide modest improvement")
    print(f"   Improvement: {improved_cv - phase1b_cv:+.1%} vs Phase 1B")
    print("   Recommendation: Consider cost-benefit, baseline accuracy still better")
else:
    print("\n❌ INSUFFICIENT: Realistic patterns still degrading accuracy")
    print(f"   Gap to baseline: {baseline_cv - improved_cv:+.1%}")
    print("   Recommendation: Revert to baseline or try Option C (find real data)")

print("\n✓ Complete. Training script finished.")
