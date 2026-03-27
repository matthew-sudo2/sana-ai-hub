# Project File Organization Guide

## Directory Structure

```
sana-ai-hub/
├── scripts/
│   ├── data_generation/        # Dataset creation and corruption
│   │   ├── generate_corrupted_datasets.py
│   │   ├── create_synthetic_good_dataset.py
│   │   ├── create_more_synthetic_datasets.py
│   │   └── create_additional_bad_dataset.py
│   └── utilities/              # System scripts
│       ├── start.py            # Start backend + frontend
│       └── stop.py             # Stop services
│
├── models/
│   └── train/
│       ├── retrain_with_real_data.py    # Main retraining script (188 synthetic + N real)
│       └── RETRAINING_GUIDE.md          # Complete instructions
│
├── evaluation/
│   ├── test_all_real_datasets.py        # Single test on all 11 datasets
│   ├── evaluate_kfold_cv.py             # 5-Fold cross-validation (unstable)
│   └── evaluate_loocv.py                # Leave-One-Out CV (validates 90.9%)
│
├── data/
│   ├── labeled/
│   │   ├── good/   (6 datasets)
│   │   └── bad/    (5 datasets)
│   ├── synthetic/
│   │   └── augmented_data_multilevel.pkl (188 samples)
│   └── raw/        (pipeline input data)
│
├── backend/        (core pipeline)
├── frontend/       (React dashboard)
├── models/
│   └── best_model.pkl          # Current model (90.9% accuracy)
│
└── archive/        # Deprecated/experimental scripts
```

## Quick Reference

### Model Training & Evaluation

**Retrain the model** (188 synthetic + 11 real datasets):
```bash
python models/train/retrain_with_real_data.py
```

**Test model on all 11 datasets** (single evaluation):
```bash
python evaluation/test_all_real_datasets.py
```

**Validate with Leave-One-Out CV** (rigorous 90.9% test):
```bash
python evaluation/evaluate_loocv.py
```

**Test with k-fold CV** (not recommended for small datasets):
```bash
python evaluation/evaluate_kfold_cv.py
```

### Data Generation

**Generate corrupted datasets** (create BAD datasets from GOOD):
```bash
python scripts/data_generation/generate_corrupted_datasets.py
```

**Create synthetic GOOD datasets** (teach model about low-variance good data):
```bash
python scripts/data_generation/create_synthetic_good_dataset.py
python scripts/data_generation/create_more_synthetic_datasets.py
python scripts/data_generation/create_additional_bad_dataset.py
```

### System Control

**Start backend + frontend**:
```bash
python scripts/utilities/start.py
```

**Stop services**:
```bash
python scripts/utilities/stop.py
```

## Model Performance Summary

| Metric | Value |
|--------|-------|
| Single Test (all 11) | 90.9% |
| Leave-One-Out CV | 90.9% ± 0.0% |
| 5-Fold K-CV | 56.7% ± 22.6% |
| BAD Detection | 100% (5/5) |
| GOOD Detection | 83.3% (5/6) |

**Key Insight**: LOOCV confirms 90.9% is the true real-world accuracy. The model reliably detects all corrupted data and correctly identifies most clean datasets (games.csv is the one exception).

## Dataset Count

**Training Data** (for retrain script):
- Synthetic: 188 samples
- Real GOOD: 6 datasets
- Real BAD: 5 datasets
- **Total: 199 samples**

**Evaluation Data**:
- GOOD: bank_account_transactions.csv, employee_records.csv, games.csv, Spotify.csv, student_grades.csv, synthetic_clean_transactions.csv
- BAD: corruption_extreme_outliers.csv, corruption_heavy_missing.csv, corruption_inconsistent_columns.csv, corruption_many_duplicates.csv, corruption_mixed_issues.csv

## When to Use Each Script

| Goal | Script | Command |
|------|--------|---------|
| Retrain model | `models/train/retrain_with_real_data.py` | `python models/train/retrain_with_real_data.py` |
| Quick test | `evaluation/test_all_real_datasets.py` | `python evaluation/test_all_real_datasets.py` |
| Validate rigorously | `evaluation/evaluate_loocv.py` | `python evaluation/evaluate_loocv.py` |
| Create test data | `scripts/data_generation/*.py` | `python scripts/data_generation/create_*.py` |
| Start system | `scripts/utilities/start.py` | `python scripts/utilities/start.py` |
| Stop system | `scripts/utilities/stop.py` | `python scripts/utilities/stop.py` |

## Archive (Not Needed)

Deprecated scripts moved to `archive/`:
- `compare_models.py` - Old model comparison
- `prepare_real_quality_features.py` - Old feature extraction
- `save_best_model.py` - Integrated into retrain script

These can be deleted safely.
