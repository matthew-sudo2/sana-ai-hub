# Backup Management System

## Overview

The Sana AI Hub implements **automatic backup rotation** to keep the `models/` directory clean while maintaining disaster recovery capability.

## Architecture

```
models/
├── best_model.pkl                    # Current model (ACTIVE)
├── best_model_candidate.pkl          # During validation (TEMP)
├── best_model_bak_20260327_063042.pkl   # Recent (KEPT - recent 1)
├── best_model_bak_20260327_062407.pkl   # Recent (KEPT - recent 2)
├── best_model_bak_20260327_061621.pkl   # Recent (KEPT - recent 3)
│
└── archived/
    ├── model_rejected_validation_*.pkl   # Rejected candidates (AUDIT TRAIL)
    │
    └── backups/
        ├── best_model_bak_20260327_061432.pkl     # OLD (archived)
        ├── best_model_bak_20260327_055321.pkl     # OLD (archived)
        └── best_model_bak_20260327_051108.pkl     # OLD (archived)
```

## Auto-Rotation Policy

### Default Behavior (Automatic)
- **Keep 3 most recent backups** in `models/` directory
- When a 4th backup is created, oldest is automatically moved to `models/archived/backups/`
- Happens silently during `retrain()` process

### Example Timeline
```
[Step 1] First retrain  → Create best_model_bak_20260327_061432.pkl
                          Backups in models/: 1
                          
[Step 2] Second retrain → Create best_model_bak_20260327_061621.pkl
                          Backups in models/: 2
                          
[Step 3] Third retrain  → Create best_model_bak_20260327_062407.pkl
                          Backups in models/: 3 (at threshold)
                          
[Step 4] Fourth retrain → Create best_model_bak_20260327_063042.pkl
                          Backups in models/: 4
                          ⚙️ AUTO-ROTATE TRIGGERED
                          → Archive best_model_bak_20260327_061432.pkl to archived/backups/
                          Backups in models/: 3 (back to threshold)
```

## Manual Cleanup

### Run Cleanup Script
```bash
# Archive old backups (keep 3 recent)
python cleanup_backups.py

# Keep 5 recent instead of 3
python cleanup_backups.py --keep 5

# Preview what would be archived (dry-run)
python cleanup_backups.py --dry-run
```

### Example Output
```
================================================================================
                    MODEL BACKUP CLEANUP UTILITY
================================================================================

Found 7 backup(s)
Policy: Keep 3 most recent, archive older

ACTION: Archive 4 old backup(s)
        Keep 3 recent backup(s)

RECENT BACKUPS (keeping in models/):
  ✓ best_model_bak_20260327_063042.pkl
  ✓ best_model_bak_20260327_062407.pkl
  ✓ best_model_bak_20260327_061621.pkl

OLD BACKUPS (archiving to models/archived/backups/):
  ✗ best_model_bak_20260327_061432.pkl (from 2026-03-27 06:14:32)
  ✗ best_model_bak_20260327_055321.pkl (from 2026-03-27 05:53:21)
  ✗ best_model_bak_20260327_051108.pkl (from 2026-03-27 05:11:08)
  ✗ best_model_bak_20260327_041645.pkl (from 2026-03-27 04:16:45)

================================================================================
✓ CLEANUP COMPLETE: Archived 4/4 old backup(s)
================================================================================
```

## Directory Structure

### `models/best_model_bak_*.pkl`
- **Purpose**: Disaster recovery
- **Kept**: 3 most recent (auto-rotated)
- **Lifetime**: Days to weeks (recent retrains only)
- **Used for**: Quick rollback if latest promotion is bad

### `models/archived/backups/best_model_bak_*.pkl`
- **Purpose**: Historical record  
- **Kept**: All (indefinite)
- **Lifetime**: Entire project history
- **Used for**: Long-term audit trail, very rare rollback

### `models/archived/model_rejected_*.pkl`
- **Purpose**: Rejected candidate models (validation failed)
- **Kept**: All (indefinite)
- **Lifetime**: Per retraining session
- **Used for**: Understanding why candidates failed validation

## File Size Considerations

Each model backup is typically **~1-5 MB** depending on:
- RandomForest complexity (n_estimators, max_depth)
- Serialization overhead (pickle)

Keeping 3 recent backups: **3-15 MB**
Archived backups (e.g., 50 old): **50-250 MB**

## Recovery Scenarios

### Scenario 1: Rollback to Previous Model
```bash
# If best_model.pkl is bad, restore from recent backup
cp models/best_model_bak_20260327_063042.pkl models/best_model.pkl

# Update metrics log to note manual rollback
echo '{"action": "manual_rollback", "timestamp": "..."}' >> models/model_metrics.jsonl
```

### Scenario 2: Investigate Old Backup
```bash
# If you need to check performance of a model from 2 weeks ago
python validate_model.py models/archived/backups/best_model_bak_20260310_145230.pkl
```

### Scenario 3: Clean Up Historical Data
```bash
# Delete backups older than specific date (archive management)
# Typically done once per month/quarter
rm models/archived/backups/best_model_bak_20260301_*.pkl
```

## Configuration

### Adjust Keep Threshold
To keep **5 recent backups** instead of 3:

**Option A: In continuous_learner.py**
```python
# Around line where retrain calls _backup_old_model
self._backup_old_model(keep_recent=5)  # Change 3 to 5
```

**Option B: When calling cleanup script**
```bash
python cleanup_backups.py --keep 5
```

### Adjust Auto-Rotation Point
To trigger rotation at 5 backups instead of 4:
```python
# In continuous_learner.py, _backup_old_model method
self._rotate_old_backups(keep_recent=4)  # Allow up to 4 before archiving
```

## Monitoring

### Check Backup Count
```bash
ls -lh models/best_model_bak_*.pkl | wc -l
```

### List All Backups (Recent + Archived)
```bash
echo "=== RECENT BACKUPS ===" && \
ls -lh models/best_model_bak_*.pkl 2>/dev/null && \
echo && echo "=== ARCHIVED BACKUPS ===" && \
ls -lh models/archived/backups/best_model_bak_*.pkl 2>/dev/null
```

### Calculate Total Disk Usage
```bash
du -sh models/
du -sh models/archived/
```

## Best Practices

✅ **DO**:
- Run cleanup script monthly to keep old archives organized
- Keep archived backups for entire project lifetime
- Check backup disk usage before auto-rotation fills up
- Label important backups for easy identification

❌ **DON'T**:
- Delete recent backups (keep at least 3)
- Delete archived backups unless disk space critical
- Manually rename backup files (breaks auto-rotation)
- Edit backup files directly (use shadow validation instead)

## Troubleshooting

**Q: Backup rotation failed during retrain?**
- Check disk space: `df -h models/`
- Manually run: `python cleanup_backups.py`
- Check permissions on `models/archived/backups/`

**Q: How do I restore a specific backup?**
```bash
# Find the timestamp you want
ls -lh models/archived/backups/best_model_bak_*.pkl | grep "specific date"

# Copy to active
cp models/archived/backups/best_model_bak_20260320_143022.pkl models/best_model.pkl

# Rebuild model cache if needed
python -c "from backend.agents.analyst import MLQualityScorer; MLQualityScorer().reload_model()"
```

**Q: Can I manually archive all backups at once?**
```bash
python cleanup_backups.py --keep 0  # Archives all
python cleanup_backups.py --keep 1  # Keeps only 1 recent
```
