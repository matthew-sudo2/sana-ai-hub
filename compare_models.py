"""
Random Forest vs Logistic Regression comparison
"""
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

# Load data
good_features = np.load('data/synthetic/good_quality_features_real.npy')
bad_features = np.load('data/synthetic/bad_quality_features_real.npy')

X_all = np.vstack([good_features, bad_features])
y_all = np.array([1]*len(good_features) + [0]*len(bad_features))

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
)

print("="*80)
print("RANDOM FOREST vs LOGISTIC REGRESSION")
print("="*80)

print(f"\nDataset: {len(X_train)} training, {len(X_test)} test")
print(f"Classes: {np.sum(y_train==1)} good, {np.sum(y_train==0)} bad")

# ====== RANDOM FOREST ======
print("\n" + "="*80)
print("TESTING RANDOM FOREST (Different max_depth values)")
print("="*80)

best_acc = 0
best_depth = 0
rf_results = {}

for depth in [3, 4, 5, 6, 7, 8]:
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=depth,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    acc = (rf.predict(X_test) == y_test).mean()
    rf_results[depth] = rf
    
    print(f"  max_depth={depth}: Test Accuracy = {acc:.1%}")
    
    if acc > best_acc:
        best_acc = acc
        best_depth = depth

# Train best RF
print(f"\n✓ Best Random Forest: max_depth={best_depth} with {best_acc:.1%} test accuracy")

rf_best = rf_results[best_depth]
rf_test_acc = (rf_best.predict(X_test) == y_test).mean()
rf_test_auc = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])

# ====== LOGISTIC REGRESSION ======
print("\n" + "="*80)
print("LOGISTIC REGRESSION")
print("="*80)

# Handle NaN values with mean imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_all_imputed = imputer.fit_transform(X_all)

lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_imputed, y_train)

lr_test_acc = (lr.predict(X_test_imputed) == y_test).mean()
lr_test_auc = roc_auc_score(y_test, lr.predict_proba(X_test_imputed)[:, 1])

print(f"\n✓ Logistic Regression:")
print(f"  Test Accuracy = {lr_test_acc:.1%}")
print(f"  ROC-AUC = {lr_test_auc:.1%}")

# ====== K-FOLD CV COMPARISON ======
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION (5-fold)")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RF CV
rf_cv = cross_validate(rf_best, X_all, y_all, cv=skf, scoring=['accuracy', 'roc_auc'])
rf_cv_mean = rf_cv['test_accuracy'].mean()
rf_cv_std = rf_cv['test_accuracy'].std()
rf_cv_auc = rf_cv['test_roc_auc'].mean()

print(f"\n🌲 Random Forest:")
print(f"  Accuracy: {rf_cv_mean:.1%} ± {rf_cv_std:.1%}")
print(f"  ROC-AUC:  {rf_cv_auc:.1%}")

# LR CV
lr_cv = cross_validate(lr, X_all_imputed, y_all, cv=skf, scoring=['accuracy', 'roc_auc'])
lr_cv_mean = lr_cv['test_accuracy'].mean()
lr_cv_std = lr_cv['test_accuracy'].std()
lr_cv_auc = lr_cv['test_roc_auc'].mean()

print(f"\n🔵 Logistic Regression:")
print(f"  Accuracy: {lr_cv_mean:.1%} ± {lr_cv_std:.1%}")
print(f"  ROC-AUC:  {lr_cv_auc:.1%}")

# ====== FINAL COMPARISON ======
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

print(f"\n{'Model':<25} {'Test Acc':<15} {'K-fold CV':<20} {'ROC-AUC':<15}")
print("-"*75)
print(f"{'Random Forest':<25} {rf_test_acc:.1%} {rf_cv_mean:.1%} ± {rf_cv_std:.1%} {rf_cv_auc:.1%}")
print(f"{'Logistic Regression':<25} {lr_test_acc:.1%} {lr_cv_mean:.1%} ± {lr_cv_std:.1%} {lr_cv_auc:.1%}")

# Determine winner
print("\n" + "="*80)
print("🏆 WINNER")
print("="*80)

if rf_cv_mean > lr_cv_mean:
    improvement = (rf_cv_mean - lr_cv_mean) * 100
    print(f"\n🌲 Random Forest WINS!")
    print(f"   K-fold CV: {rf_cv_mean:.1%} (vs {lr_cv_mean:.1%})")
    print(f"   Improvement: +{improvement:.1f}%")
    
    # Save RF model
    with open('models/quality_model_rf.pkl', 'wb') as f:
        pickle.dump(rf_best, f)
    print(f"\n✓ Saved to: models/quality_model_rf.pkl")
else:
    improvement = (lr_cv_mean - rf_cv_mean) * 100
    print(f"\n🔵 Logistic Regression WINS!")
    print(f"   K-fold CV: {lr_cv_mean:.1%} (vs {rf_cv_mean:.1%})")
    print(f"   Improvement: +{improvement:.1f}%")
    
    # Save LR model
    with open('models/quality_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    print(f"\n✓ Saved to: models/quality_model.pkl")

# Feature importance
print("\n" + "="*80)
print("🎯 FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_names = ['missing_ratio', 'duplicate_ratio', 'numeric_ratio', 
                 'constant_columns', 'avg_variance', 'avg_skewness']

print("\n🌲 Random Forest Feature Importance:")
rf_importance = rf_best.feature_importances_
for name, imp in sorted(zip(feature_names, rf_importance), key=lambda x: x[1], reverse=True):
    print(f"   {name:<20} {imp:.1%}")

print("\n🔵 Logistic Regression Coefficients (absolute):")
lr_coef = np.abs(lr.coef_[0])
for name, coef in sorted(zip(feature_names, lr_coef), key=lambda x: x[1], reverse=True):
    print(f"   {name:<20} {coef:.4f}")

print("\n" + "="*80)
