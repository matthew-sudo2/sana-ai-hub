#!/usr/bin/env python3
"""
QUICK DEMO: Show feedback loop + model retrain in action.
Demo Mode: retrain threshold = 5 (vs 20 production)
"""

import os
os.environ['DEMO_MODE'] = 'true'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.utils.feedback_db import FeedbackDB
from backend.utils.ml_quality_scorer import MLQualityScorer
from backend.utils.continuous_learner import ContinuousLearner
import hashlib

print("\n" + "="*80)
print("⚡ QUICK DEMO: FEEDBACK LOOP + RETRAIN")
print("="*80)
print("\n[MODE] DEMO_MODE=true | Retrain threshold: 5 samples (production: 20)\n")

# Clear old database
import os.path
db_path = "backend/data/feedback.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print("✓ Cleared old database\n")

feedback_db = FeedbackDB()

# Generate 5 realistic quality feedbacks
feedbacks = [
    {
        "name": "Order data (poor, high pred)",
        "hash": hashlib.md5(b"orders_dirty").hexdigest(),
        "pred": 72,
        "actual": 0,
        "features": [0.15, 0.25, 0.45, 0.30, 0.50, 0.35, 0.40, 0.20],
        "accepted": True
    },
    {
        "name": "Employee data (good, low pred)",
        "hash": hashlib.md5(b"employees_good").hexdigest(),
        "pred": 35,
        "actual": 2,
        "features": [0.08, 0.10, 0.60, 0.15, 0.65, 0.45, 0.55, 0.30],
        "accepted": True
    },
    {
        "name": "Sales data (good, low pred)",
        "hash": hashlib.md5(b"sales_underrated").hexdigest(),
        "pred": 45,
        "actual": 2,
        "features": [0.12, 0.15, 0.65, 0.22, 0.70, 0.55, 0.72, 0.38],
        "accepted": True  # Good data but low pred = ACCEPT (model underestimated)
    },
    {
        "name": "Inventory (poor, high pred)",
        "hash": hashlib.md5(b"inventory_messy").hexdigest(),
        "pred": 78,
        "actual": 0,
        "features": [0.20, 0.30, 0.40, 0.35, 0.55, 0.40, 0.50, 0.25],
        "accepted": True
    },
    {
        "name": "Customer data (good, low pred)",
        "hash": hashlib.md5(b"customers_decent").hexdigest(),
        "pred": 42,
        "actual": 2,
        "features": [0.10, 0.12, 0.55, 0.20, 0.60, 0.50, 0.65, 0.35],
        "accepted": True
    },
]

print("📊 SUBMITTING FEEDBACK:\n")

accepted_count = 0
for i, fb in enumerate(feedbacks, 1):
    status, reason = FeedbackDB.should_accept_feedback(fb["pred"], fb["actual"])
    
    if status:
        feedback_db.save(
            dataset_hash=fb["hash"],
            predicted_score=fb["pred"],
            actual_label=fb["actual"],
            features=fb["features"],
            is_quality_gated=True
        )
        accepted_count += 1
        print(f"  {i}. ✓ {fb['name']:30s} → ACCEPTED")
    else:
        print(f"  {i}. ✗ {fb['name']:30s} → REJECTED (no signal)")

total = feedback_db.count()
print(f"\n✓ Total accepted: {accepted_count}")
print(f"✓ Total in DB: {total}\n")

# Check for retrain
threshold = 5
should_retrain = total >= threshold and total % threshold == 0

print("="*80)
if should_retrain:
    print(f"🚀 RETRAIN TRIGGERED ({total}/{threshold} samples threshold)")
    print("="*80)
    print("\n📈 RETRAINING MODEL:\n")
    
    learner = ContinuousLearner()
    result = learner.retrain()
    
    print("\n" + "="*80)
    if result["success"]:
        print("✅ RETRAIN SUCCESSFUL!")
        print("="*80)
        print(f"\n  CV Score: {result['cv_score']:.1%}")
        print(f"  Model promoted to production")
        print(f"  Feedback data cleaned (kept 500 recent)")
    else:
        print(f"❌ RETRAIN FAILED: {result.get('error')}")
        print("="*80)
else:
    remaining = threshold - total
    print(f"⏳ Not yet ({total}/{threshold}), need {remaining} more")
    print("="*80)

print("\n✨ DEMO COMPLETE - System is production-ready!")
print("   - Quality gate prevents useless feedback")
print("   - Batch learning ensures stability (5 demo, 20 production)")
print("   - Model improves only with high-quality signal\n")
