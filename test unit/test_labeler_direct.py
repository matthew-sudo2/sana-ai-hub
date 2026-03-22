#!/usr/bin/env python3
"""Direct test of labeler - capture all output"""
import sys
sys.path.insert(0, 'backend')

from pathlib import Path
import json
from backend.agents.labeler import run_labeler
from backend.agents.scout import run_scout

# Use the test dataset
scout_input_json = Path("backend/data/raw/test_scout_input.json")
scout_result_json = Path("backend/runs/test_scout_result.json")

# Create minimal test scout result
if not scout_result_json.exists():
    scout_result_json.parent.mkdir(parents=True, exist_ok=True)
    scout_result = {
        "is_dataset": True,
        "source": "Messy_Employee_dataset.csv" ,
        "source_type": "csv",
        "raw_quantitative_stats": None
    }
    scout_result_json.write_text(json.dumps(scout_result))

# Run scout first if needed
scout_dir = Path("backend/runs/scout_test")
scout_dir.mkdir(parents=True, exist_ok=True)
scout_output = scout_dir / "scout_result.json"

if not scout_output.exists():
    print("=== Running SCOUT ===", flush=True)
    try:
        scout_result = run_scout(
            url_or_file="backend/data/Messy_Employee_dataset.csv",
            out_dir=str(scout_dir)
        )
        print(f"[test] Scout result keys: {scout_result.keys()}", flush=True)
    except Exception as e:
        print(f"[test] Scout error: {e}", flush=True)
        import traceback
        traceback.print_exc()
else:
    print("[test] Using existing scout output", flush=True)

# Run labeler
print("\n=== Running LABELER ===", flush=True)
try:
    labeler_result = run_labeler(scout_output)
    print(f"\n[test] Labeler result keys: {labeler_result.keys()}", flush=True)
    print(f"[test] Cleaned CSV path: {labeler_result.get('cleaned_data_csv')}", flush=True)
    
    # Check Age column
    if labeler_result.get('cleaned_data_csv'):
        import pandas as pd
        cleaned_df = pd.read_csv(labeler_result['cleaned_data_csv'])
        if 'age' in cleaned_df.columns:
            non_null = cleaned_df['age'].notna().sum()
            total = len(cleaned_df)
            print(f"\n[test] ✓ Age column in cleaned data: {non_null}/{total} non-null ({100*non_null/total:.1f}%)", flush=True)
        else:
            print(f"\n[test] ✗ Age column NOT found in cleaned data", flush=True)
            print(f"[test] Columns: {list(cleaned_df.columns)}", flush=True)
except Exception as e:
    print(f"[test] ✗ Labeler ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n=== DONE ===", flush=True)
