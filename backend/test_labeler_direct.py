#!/usr/bin/env python3
"""Direct test of labeler - capture all output"""
import sys
from pathlib import Path
import json

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from agents.labeler import run_labeler
from agents.scout import run_scout

# Use the test dataset
scout_result_json = Path(__file__).parent / "runs" / "test_scout_result.json"

# Create minimal test scout result
scout_result_json.parent.mkdir(parents=True, exist_ok=True)
scout_result = {
    "is_dataset": True,
    "source": "Messy_Employee_dataset.csv",
    "source_type": "csv",
    "raw_quantitative_stats": None
}
scout_result_json.write_text(json.dumps(scout_result))

# Run scout first
scout_dir = scout_result_json.parent
scout_output = scout_dir / "scout_result.json"

print("=== Running SCOUT ===", flush=True)
try:
    scout_result = run_scout(
        url_or_file="data/Messy_Employee_dataset.csv",
        out_dir=str(scout_dir)
    )
    print(f"[test] Scout completed", flush=True)
    Scout_output_file = scout_dir / "scout_result.json"
    print(f"[test] Scout result file: {Scout_output_file}", flush=True)
except Exception as e:
    print(f"[test] Scout error: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Run labeler
print("\n=== Running LABELER ===", flush=True)
try:
    scout_result_file = scout_dir / "scout_result.json"
    print(f"[test] Using scout output: {scout_result_file}", flush=True)
    print(f"[test] File exists: {scout_result_file.exists()}", flush=True)
    
    labeler_result = run_labeler(scout_result_file)
    print(f"\n[test] LABELER COMPLETED", flush=True)
    print(f"[test] Result keys: {labeler_result.keys()}", flush=True)
    
    # Check Age column
    import pandas as pd
    cleaned_csv = Path(labeler_result.get('cleaned_data_csv'))
    if cleaned_csv.exists():
        cleaned_df = pd.read_csv(cleaned_csv)
        if 'age' in cleaned_df.columns:
            non_null = cleaned_df['age'].notna().sum()
            total = len(cleaned_df)
            print(f"\n[test] ✓ Age column: {non_null}/{total} non-null ({100*non_null/total:.1f}%)", flush=True)
            if non_null == total:
                print("[test] ✓✓✓ SUCCESS - Age column FULLY FILLED!", flush=True)
            else:
                print(f"[test] ⚠️  STILL MISSING {total - non_null} values", flush=True)
        else:
            print(f"[test] ✗ Age column NOT found", flush=True)
            print(f"[test] Columns: {list(cleaned_df.columns)}", flush=True)
except Exception as e:
    print(f"[test] ✗ Labeler ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n=== DONE ===", flush=True)
