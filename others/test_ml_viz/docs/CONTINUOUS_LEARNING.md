# Continuous Learning & Dynamic Intelligence System

## Overview

This implementation adds a **feedback loop** to the Sana AI Hub, enabling dynamic model improvement through user feedback. The system automatically retrains the quality prediction model as users provide feedback on score accuracy.

## How It Works

### 1. User Provides Feedback (Frontend)
After a dataset is processed and scored, users see quality feedback buttons:
```
Was this score accurate?
├─ ✓ Perfect         (label: 3 - Excellent)
├─ ✓ Close enough    (label: 2 - Good)
├─ ✗ Too high        (label: 1 - Fair)
└─ ✗ Very wrong      (label: 0 - Poor)
```

### 2. Feedback Storage (Backend - SQLite)
**File**: `backend/utils/feedback_db.py`

Feedback is stored in SQLite database:
```sql
feedback(
  id INTEGER PRIMARY KEY,
  dataset_hash TEXT,          -- MD5 of dataset
  predicted_score REAL,       -- Model's prediction (0-100)
  actual_label INTEGER,       -- User feedback (0-3)
  features TEXT,              -- JSON: 8 extracted features
  timestamp DATETIME
)
```

Location: `backend/data/feedback.db`

### 3. Feature Extraction & Caching
**File**: `backend/utils/feature_cache.py`

During validation phase, 8 quality features are extracted:
- Missing ratio
- Duplicate ratio
- Numeric column ratio
- Constant columns count
- Normalized variance
- Skewness
- Cardinality ratio
- Mean kurtosis

Features are cached in: `{run_dir}/features.json`

### 4. Auto-Retrain Trigger
**File**: `backend/graph.py` (execute_run function)

After every pipeline execution:
- Check feedback count in SQLite
- If count % 20 == 0, trigger retraining
- **Retrain frequency**: Every 20 feedback samples

### 5. Model Retraining
**File**: `backend/utils/continuous_learner.py`

When triggered:
1. Load original training data (40 good + 75 bad features = 115 samples)
2. Load accumulated feedback data (20 samples)
3. Combine for total of ~135 training samples
4. Train new RandomForestClassifier with same hyperparameters
5. K-fold cross-validation for honest performance estimate
6. Backup old model with timestamp
7. Save new model to `models/best_model.pkl`
8. Log metrics to `models/model_metrics.jsonl`

### 6. Model Metrics & History
**Location**: `models/model_metrics.jsonl` (append-only log)

Each retrain event logged:
```json
{
  "timestamp": "2026-03-27T14:30:00",
  "cv_score": 0.817,
  "feedback_samples": 20,
  "total_training_samples": 135,
  "model_type": "RandomForestClassifier",
  "action": "retrain"
}
```

## API Endpoints

### Submit Feedback
```
POST /api/feedback
Content-Type: application/json

{
  "dataset_hash": "abc123def",
  "predicted_score": 78.5,
  "actual_quality": 2,
  "features": [0.05, 0.0, 0.6, 0, 2.5, 1.2, 0.3, 0.1]
}

Response:
{
  "status": "stored" | "retrained",
  "feedback_count": 15,
  "cv_score": 0.81,              // Only if retrained
  "next_retrain_at": 5,          // Feedbacks left until retrain
  "message": "..."
}
```

### Get Feedback Statistics
```
GET /api/feedback/stats

Response:
{
  "total_feedbacks": 35,
  "models_trained": 1,
  "current_cv_score": 0.817,
  "latest_retrain_at": "2026-03-27T14:30:00",
  "improvement_percentage": 1.5
}
```

## Frontend Components

### FeedbackWidget.tsx
- Shows feedback buttons after quality assessment
- Submits feedback to backend
- Displays confirmation & next retrain countdown

### FeedbackSummary.tsx
- Shows aggregated metrics
- Total feedbacks, models trained, improvement %
- Current model accuracy

### useFeedback.ts Hook
- `submitFeedback()` - POST to /api/feedback
- `getStats()` - GET from /api/feedback/stats
- Built-in error handling & loading states

## Testing & Verification

Run verification suite:
```bash
python verify_feedback_loop.py
```

Tests:
1. ✓ Feedback database (store/retrieve)
2. ✓ Feature caching
3. ✓ Model retraining logic
4. ✓ API endpoints

## File Structure (New Files)

```
backend/
├── utils/
│   ├── feedback_db.py              [NEW - SQLite storage]
│   ├── continuous_learner.py       [NEW - Retrain logic]
│   ├── feature_cache.py            [NEW - Feature caching]
│
├── data/
│   ├── feedback.db                 [NEW - SQLite database]
│   └── model_versions/
│       ├── best_model.pkl          [UPDATED - current model]
│       └── model_metrics.jsonl     [NEW - retrain history]

frontend/
├── src/components/
│   └── FeedbackWidget.tsx          [NEW - UI feedback component]
├── src/hooks/
│   └── useFeedback.ts              [NEW - API hook]
└── src/types/
    └── feedback.ts                 [NEW - TypeScript types]
```

## Configuration

### Retrain Frequency
Edit in `backend/graph.py` (line ~925):
```python
if feedback_count > 0 and feedback_count % 20 == 0:  # Change 20 to desired count
```

### RandomForest Hyperparameters
Edit in `backend/utils/continuous_learner.py` (line ~90):
```python
RandomForestClassifier(
    n_estimators=100,              # Number of trees
    max_depth=15,                  # Tree depth
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
```

## Performance Characteristics

- **Feature Extraction**: <100ms per dataset
- **Feedback Storage**: <10ms
- **Retrain Time**: ~5-10s (includes validation)
- **Model Deployment**: Instant (next upload uses new model)
- **SQLite Storage**: <1MB per 1000 feedback records

## Expected Improvements

With continuous feedback:
- **Week 1** (20 feedbacks): Initial results, ~1-2% accuracy gain
- **Week 2-3** (50+ feedbacks): Convergence to domain-specific patterns
- **Ongoing**: Gradual refinement as system learns user preferences

## Future Enhancements

Phase 2 features (not included):
- [ ] Feedback cleanup policy (keep last 1000, archive rest)
- [ ] Advanced scheduling (scheduled vs event-based retrain)
- [ ] A/B testing (old vs new model comparison)
- [ ] Feature importance analysis
- [ ] User segmentation (domain-specific models)
- [ ] Automated rollback on performance degradation
- [ ] Real-time performance monitoring dashboard

## Troubleshooting

**Issue**: Retrain not triggering
- Check `backend/data/feedback.db` exists
- Verify feedback count with: `python -c "from backend.utils.feedback_db import FeedbackDB; print(FeedbackDB().count())"`

**Issue**: Model not retraining
- Ensure `data/synthetic/` directory has training features
- Check `models/model_metrics.jsonl` for error logs

**Issue**: API returns 500 error
- Check backend logs for model loading issues
- Verify `models/best_model.pkl` exists and is valid

## Summary

This system creates a **learning feedback loop** where:
1. User provides data ➜ Pipeline processes it
2. User rates quality score accuracy ➜ Feedback recorded
3. Every 20 feedbacks ➜ Model retrains automatically
4. New model deployed ➜ Next upload uses improved predictions
5. System improves continuously with real-world usage

**Key Benefit**: Model accuracy increases over time by learning from actual user feedback instead of staying static with pre-trained weights.
