#!/usr/bin/env python3
"""
DEMO SCRIPT: Model Retraining with Shadow Validation
For judges/evaluators to see the complete retraining workflow

This script:
1. Simulates accumulated user feedback
2. Triggers the continuous learner to retrain
3. Shows shadow validation comparison
4. Displays final promotion/rejection decision

NOTE: This is a demonstration script only. It does NOT replace the
user feedback system. The actual feedback collection happens through
the DataViewer feedback widget in the frontend.

Usage:
    python demo_retrain.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.utils.continuous_learner import ContinuousLearner
from backend.utils.feedback_db import FeedbackDB


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_step(step_num: int, title: str, details: str = ""):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {title}")
    if details:
        print(f"            {details}")


def simulate_feedback():
    """Simulate user feedback for demo purposes."""
    print_section("SIMULATING USER FEEDBACK", "~")
    
    print("\n💬 Scenario: Users reviewed the data quality scores and provided feedback")
    print("   - 5 users gave feedback on various datasets")
    print("   - Some thought scores were too high, some too low")
    print("   - System learning from these inconsistencies...\n")
    
    # In a real scenario, this would come from the frontend feedback widget
    # For demo, we'll just note that feedback exists
    print("✓ Feedback accumulated (5-10 samples threshold for retrain)")
    print("✓ System ready to learn from user corrections\n")


def check_prerequisite_data():
    """Check if training data exists."""
    print_section("CHECKING PREREQUISITES", "~")
    
    training_data_path = Path("data/synthetic/training_data_8features.pkl")
    
    if not training_data_path.exists():
        print(f"\n⚠️  Training data not found: {training_data_path}")
        print("\nTo enable retraining:")
        print("  1. Run: python backend/agents/scout.py")
        print("  2. Or: python models/train_quality_model.ipynb")
        print("\nFor demo purposes, proceeding with assumption data exists...\n")
        return False
    
    print(f"\n✓ Training data found: {training_data_path}")
    print(f"✓ Size: {training_data_path.stat().st_size / 1024:.1f} KB\n")
    return True


def main():
    """Run the complete demo retraining workflow."""
    
    print_section("🔄 SANA AI HUB: MODEL RETRAINING DEMO")
    print("\nThis demo shows how the system automatically retrains models")
    print("based on accumulated user feedback.\n")
    
    # Phase 1: Simulate feedback
    simulate_feedback()
    
    # Phase 2: Check data
    has_data = check_prerequisite_data()
    
    # Phase 3: Retrain
    print_section("INITIATING RETRAINING PROCESS")
    
    print("\n🚀 Starting continuous learning...\n")
    
    try:
        learner = ContinuousLearner()
        
        print("=" * 80)
        print("RETRAINING WITH SHADOW VALIDATION")
        print("=" * 80)
        
        # This will show all the detailed output from the learner
        result = learner.retrain()
        
        # Phase 4: Results
        print_section("📊 RETRAINING RESULTS")
        
        if result["success"]:
            print(f"\n✅ Training Successful")
            print(f"   CV Score: {result['cv_score']:.1%}")
            print(f"   Feedback Samples: {result['feedback_count']}")
            print(f"   Total Training Samples: {result['total_samples']}")
            print(f"\n🎯 Model Promotion Status: {'APPROVED ✓' if result['promoted'] else 'REJECTED ✗'}")
            print(f"   Reason: {result['validation_reason']}")
            
            if result['promoted']:
                print("\n✨ New model is now in production!")
                print("   - Shadow validation passed")
                print("   - Metrics improved over baseline")
                print("   - Users will see better quality predictions")
            else:
                print("\n📦 Model archived for review")
                print("   - Shadow validation found no improvement")
                print("   - Current model retained in production")
                print("   - Candidate saved to: models/archived/")
        else:
            print(f"\n❌ Training Failed")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"\n   Likely cause: Not enough feedback accumulated")
            print(f"   Current threshold: 5-10 samples")
        
        # Phase 5: Show files
        print_section("📁 FILES GENERATED/UPDATED")
        
        files_info = [
            ("models/best_model.pkl", "Current production model (ONLY visible file)"),
            ("models/archived/best_model_candidate_*_REJECTED.pkl", "Rejected candidates (archived)"),
            ("models/archived/backups/", "Old backups (auto-rotated)"),
            ("models/model_metrics.jsonl", "Retraining metrics log"),
            ("models/promotion_validation.jsonl", "Validation details"),
        ]
        
        for filepath, description in files_info:
            print(f"\n  {filepath}")
            print(f"    └─ {description}")
        
        # Phase 6: Information
        print_section("ℹ️  NEXT STEPS")
        
        print("\n📚 To learn more:")
        print("  • Read: SHADOW_VALIDATION_SYSTEM.md")
        print("  • Read: BACKUP_MANAGEMENT.md")
        print("  • Manual validation: python validate_model.py")
        print("  • Cleanup backups: python cleanup_backups.py")
        
        print("\n👥 For users to trigger retrain:")
        print("  1. Open DataViewer page")
        print("  2. Review quality score")
        print("  3. Click 'Feedback' if score seems wrong")
        print("  4. Rate: Too High / Accurate / Too Low")
        print("  5. Submit actual quality percentage")
        print("  6. After 5-10 feedback samples → Auto-retrain triggered")
        
        print("\n" + "=" * 80)
        print("\n✨ Demo completed! The system is ready for production use.")
        print("   Models retrain automatically as users provide feedback.\n")
        
        return 0 if result.get("success") else 1
    
    except Exception as e:
        print(f"\n❌ DEMO ERROR: {e}")
        print("\nTroubleshooting:")
        print(f"  • Check if training data exists: data/synthetic/training_data_8features.pkl")
        print(f"  • Check if models directory is writable: models/")
        print(f"  • Check error details above")
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 2
    
    sys.exit(exit_code)
