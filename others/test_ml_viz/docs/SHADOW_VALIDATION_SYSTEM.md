# Model Promotion & Shadow Validation System

## Overview

The Sana AI Hub now implements **shadow validation** before deploying retrained models. This ensures new models only reach production if they genuinely outperform the current model on a held-out test set.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RETRAINING WORKFLOW                      │
└─────────────────────────────────────────────────────────────┘

  User Feedback              
        ↓
  Feedback Database (≥N samples)
        ↓
  ┌─ Retrain Triggered ────────────────────────┐
  │                                            │
  │  1. Load training data + feedback          │
  │  2. Combine datasets                       │
  │  3. Train new RandomForest                 │
  │  4. K-fold CV validation                   │
  │  5. Save candidate model                   │
  │                                            │
  │  ┌► SHADOW VALIDATION ◄─────────────────┐ │
  │  │                                       │ │
  │  │  Load test set (20% holdout)         │ │
  │  │  Evaluate CURRENT model on test     │ │
  │  │  Evaluate CANDIDATE model on test   │ │
  │  │  Compare metrics:                    │ │
  │  │    - F1 Score                        │ │
  │  │    - Balanced Accuracy               │ │
  │  │    - AUC-ROC                         │ │
  │  │                                       │ │
  │  │  Decision:                           │ │
  │  │  ✓ PROMOTE (if improved)            │ │
  │  │  ✗ REJECT (if no improvement)      │ │
  │  │                                       │ │
  │  └───────────────────────────────────────┘ │
  │                                            │
  │  6. If PROMOTED:                          │
  │     - Backup current model                │
  │     - Deploy candidate to production      │
  │     - Log success                         │
  │                                            │
  │  7. If REJECTED:                          │
  │     - Archive candidate model             │
  │     - Keep current model in production    │
  │     - Log rejection reason                │
  └────────────────────────────────────────────┘
        ↓
  ┌─────────────────────────────┐
  │ models/best_model.pkl       │ (Active)
  │ models/archived/            │ (Rejected candidates)
  │ models/best_model_bak_*.pkl │ (Old versions)
  └─────────────────────────────┘
```

## Components

### 1. **Continuous Learner** (`backend/utils/continuous_learner.py`)

**Updated Retraining Flow:**
- Steps 1-6: Train and validate model as before
- **NEW Step 7-8**: Save as candidate + run shadow validation
- **NEW Step 9**: Promote OR archive based on validation results
- Step 10: Log metrics with promotion status

**Returns:**
```python
{
    "success": bool,           # Training succeeded
    "promoted": bool,          # Model was promoted to production
    "cv_score": float,         # K-fold CV score
    "feedback_count": int,     # Number of feedback samples
    "total_samples": int,      # Total training data
    "validation_reason": str   # Why it was promoted/rejected
}
```

### 2. **Model Promotion Validator** (`backend/utils/validate_model_promotion.py`)

**Performs Shadow Validation:**
- Loads test data (20% holdout from training set)
- Evaluates current model on test set
- Evaluates candidate model on test set
- Compares metrics (F1, Balanced Accuracy, AUC-ROC)
- Decides: Promote if any metric improves by >0.1%

**Test Metrics Evaluated:**
- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Per-class accuracy (handles imbalance)
- **Precision**: True positives / all positives
- **Recall**: True positives / all actual positives
- **F1 Score**: Harmonic mean of precision & recall
- **AUC-ROC**: Area under ROC curve

**Promotion Criteria:**
Model is promoted if **ANY** of these improve by >0.1%:
- ✓ F1 Score improves
- ✓ Balanced Accuracy improves
- ✓ AUC-ROC improves

Otherwise:
- ✗ Model is archived (not deployed)
- ✗ Current model continues in production

### 3. **CLI Validation Script** (`validate_model.py`)

**Clear Pass/Fail Output:**

Run from root directory:
```bash
# Validate default candidate model
python validate_model.py

# Validate specific model
python validate_model.py path/to/model.pkl
```

**Output Example:**
```
================================================================================
                    🔍 MODEL PROMOTION VALIDATION
================================================================================

Candidate Model: best_model_candidate.pkl

================================================================================
              SHADOW VALIDATION: MODEL PROMOTION DECISION
================================================================================

[Step 1] Loading test data...
[Step 2] Loading current best model...
[Step 3] Loading candidate model...

[Step 4] Evaluating current model...
  accuracy            : 0.8500
  balanced_accuracy   : 0.8200
  precision           : 0.8100
  recall              : 0.7900
  f1                  : 0.8000
  auc_roc             : 0.8800

[Step 5] Evaluating candidate model...
  accuracy            : 0.8650
  balanced_accuracy   : 0.8450
  precision           : 0.8350
  recall              : 0.8200
  f1                  : 0.8275
  auc_roc             : 0.8950

[Step 6] Comparison & Decision...

  F1 Score:          0.8000 → 0.8275 (Δ +0.0275)
  Balanced Accuracy: 0.8200 → 0.8450 (Δ +0.0250)
  AUC-ROC:           0.8800 → 0.8950 (Δ +0.0150)

================================================================================
✓ PROMOTION APPROVED: F1 score improved by 0.0275
================================================================================

✅ MODEL PROMOTION APPROVED

Reason: F1 score improved by 0.0275

--------------------------------------------------------------------------------
Metric Improvements:

  ✓ f1_delta                   : +0.0275
  ✓ balanced_accuracy_delta    : +0.0250
  ✓ auc_roc_delta              : +0.0150

--------------------------------------------------------------------------------

Decision: APPROVED
Timestamp: 2026-03-27T12:34:56.789000
```

**Rejection Example:**
```
================================================================================
✗ PROMOTION REJECTED: No meaningful improvement detected
  (Archiving candidate model instead)
================================================================================

❌ MODEL PROMOTION REJECTED

Reason: No meaningful improvement detected

The candidate model has been archived for reference.
The current production model remains in use.
```

## File Structure

```
models/
├── best_model.pkl                    # Current production model
├── best_model_candidate.pkl          # Candidate during validation (temp)
├── best_model_bak_20260327_061432.pkl  # Backup of previous version
├── model_metrics.jsonl               # Metrics log (includes promotion status)
├── promotion_validation.jsonl        # Validation results
└── archived/
    ├── model_rejected_validation_20260327_061621.pkl
    ├── model_rejected_validation_20260327_062407.pkl
    └── model_rejected_validation_20260327_063042.pkl
```

## Metrics Log

### `model_metrics.jsonl` (updated format)
```json
{
  "timestamp": "2026-03-27T12:34:56.789000",
  "cv_score": 0.8265,
  "feedback_samples": 12,
  "total_training_samples": 249,
  "model_type": "RandomForestClassifier",
  "action": "retrain",
  "promoted": true,
  "validation_reason": "F1 score improved by 0.0275",
  "improvement": {
    "f1_delta": 0.0275,
    "balanced_accuracy_delta": 0.0250,
    "auc_roc_delta": 0.0150
  }
}
```

### `promotion_validation.jsonl` (new)
```json
{
  "promoted": true,
  "reason": "F1 score improved by 0.0275",
  "current_model_metrics": {
    "accuracy": 0.85,
    "f1": 0.8000,
    ...
  },
  "candidate_model_metrics": {
    "accuracy": 0.8650,
    "f1": 0.8275,
    ...
  },
  "improvement": {
    "f1_delta": 0.0275,
    "balanced_accuracy_delta": 0.0250,
    "auc_roc_delta": 0.0150
  },
  "timestamp": "2026-03-27T12:34:56.789000"
}
```

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Validation** | None - auto deploy | Shadow validation on test set |
| **Decision** | Always promote | Only if outperforms |
| **Safety** | High risk of worse model | Protected by comparison |
| **Archival** | No rejected models | Archived for audit trail |
| **Visibility** | Limited outputs | Clear pass/fail in terminal |
| **Metrics** | CV score only | Full test set metrics |

## Usage Examples

### Automatic Retraining (triggered by feedback threshold)
```python
from backend.utils.continuous_learner import retrain_model

result = retrain_model()

if result["promoted"]:
    print(f"✓ Model promoted! CV: {result['cv_score']:.1%}")
else:
    print(f"✗ Model rejected: {result['validation_reason']}")
    print(f"  Candidate archived. Current model retained.")
```

### Manual Validation
```bash
# From root directory
cd /path/to/sana-ai-hub
python validate_model.py models/my_model.pkl
```

### Check Retraining History
```bash
# View last 10 retraining operations
python -c "
import json
with open('models/model_metrics.jsonl', 'r') as f:
    lines = f.readlines()
    for line in lines[-10:]:
        data = json.loads(line)
        status = '✓' if data.get('promoted') else '✗'
        print(f\"{status} {data['timestamp']}: CV={data['cv_score']:.1%}, Promoted={data['promoted']}\")
"
```

## Safety Features

✅ **Current Model Protection**: Always kept except after successful promotion
✅ **Audit Trail**: Rejected models archived with timestamp
✅ **Test Set**: 20% holdout ensures unbiased evaluation
✅ **Multiple Metrics**: Decision based on F1, Balanced Acc, AUC-ROC
✅ **Threshold Protection**: Only >0.1% improvement counts
✅ **Clear Logging**: Every decision recorded to JSONL

## Troubleshooting

**Q: Model always rejected?**
- Check if test data has enough samples
- Verify feedback data has actual quality variation
- May need more feedback samples to improve (currently at threshold)

**Q: I want to force deploy a candidate?**
- Manually run: `shutil.copy('models/best_model_candidate.pkl', 'models/best_model.pkl')`
- Then update `model_metrics.jsonl` with manual entry
- (Not recommended - breaks the validation contract)

**Q: How many feedback samples needed for promotion?**
- Minimum 5-10 samples for retraining to trigger
- More feedback = more confidence in new patterns
- Typically 20+ for meaningful improvement

**Q: Can I adjust the improvement threshold?**
- Edit line in `validate_model_promotion.py`:
  ```python
  IMPROVEMENT_THRESHOLD = 0.001  # Currently 0.1%
  ```
- Rebuild to 0.005 (0.5%) for stricter checking
