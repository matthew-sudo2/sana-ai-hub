#!/usr/bin/env python3
"""
Model Promotion CLI: Clear comparison of candidate vs current model
Run from command line to easily see pass/fail results

Usage:
    python validate_model.py                          # Validate best_model_candidate.pkl
    python validate_model.py path/to/model.pkl        # Validate specific model
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.utils.validate_model_promotion import ModelPromotionValidator


def print_header(text: str):
    """Print formatted header."""
    width = 80
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width)


def main():
    # Get candidate model path from arguments or use default
    if len(sys.argv) > 1:
        candidate_path = sys.argv[1]
    else:
        candidate_path = "models/best_model_candidate.pkl"
    
    candidate_path = Path(candidate_path)
    
    # Validate path exists
    if not candidate_path.exists():
        print_header("❌ MODEL NOT FOUND")
        print(f"\nPath: {candidate_path.absolute()}")
        print(f"\nMake sure the model file exists before validating.")
        sys.exit(1)
    
    print_header("🔍 MODEL PROMOTION VALIDATION")
    print(f"\nCandidate Model: {candidate_path.name}")
    print(f"Full Path: {candidate_path.absolute()}\n")
    
    # Run validation
    validator = ModelPromotionValidator()
    results = validator.validate(candidate_path)
    
    # Print final decision
    print_header("FINAL DECISION")
    
    if results["promoted"]:
        print("\n✅ MODEL PROMOTION APPROVED\n")
        print(f"Reason: {results['reason']}\n")
        decision = "APPROVED"
        exit_code = 0
    else:
        print("\n❌ MODEL PROMOTION REJECTED\n")
        print(f"Reason: {results['reason']}\n")
        print("The candidate model has been archived for reference.")
        print("The current production model remains in use.\n")
        decision = "REJECTED"
        exit_code = 1
    
    # Print improvement details
    if results.get("improvement"):
        print("-" * 80)
        print("Metric Improvements:\n")
        improvements = results["improvement"]
        
        for metric, delta in improvements.items():
            indicator = "↑" if delta > 0 else "↓" if delta < 0 else "="
            color = "✓" if delta > 0 else "✗" if delta < 0 else "○"
            print(f"  {color} {metric:25s}: {delta:+.4f}")
        
        print("\n" + "-" * 80)
    
    print(f"\nDecision: {decision}")
    print(f"Timestamp: {results['timestamp']}\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_header("⚠️  ERROR")
        print(f"\n{str(e)}\n")
        sys.exit(2)
