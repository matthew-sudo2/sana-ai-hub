#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.labeler import run_labeler
import pandas as pd

scout_result_file = Path(__file__).parent / "runs" / "20260322T125309Z_C__Users_User_Documents_GitHub_Information-Management-Finals-E-Commerce-Dashboar" / "scout_result.json"
print(f"Using scout result: {scout_result_file}", flush=True)
print(f"File exists: {scout_result_file.exists()}", flush=True)

if not scout_result_file.exists():
    print("ERROR: Scout result file not found!", flush=True)
    sys.exit(1)

print("\n=== Running LABELER ===", flush=True)
try:
    labeler_result = run_labeler(scout_result_file)
    print(f"\n[labeler] COMPLETED", flush=True)
    
    # Check Age column
    cleaned_csv_path = Path(labeler_result.get('cleaned_data_csv'))
    if cleaned_csv_path.exists():
        df = pd.read_csv(cleaned_csv_path)
        if 'age' in df.columns:
            non_null = df['age'].notna().sum()
            total = len(df)
            pct = 100 * non_null / total
            print(f"\n[result] Age column: {non_null}/{total} non-null ({pct:.1f}%)", flush=True)
            
            if non_null == total:
                print("[result] SUCCESS - Age FULLY FILLED!", flush=True)
            else:
                missing = total - non_null
                print(f"[result] INFO - Still missing {missing} values in age", flush=True)
        else:
            print(f"[result] ERROR - Age column not found", flush=True)
            print(f"[result] Columns: {list(df.columns)}", flush=True)
    else:
        print(f"[result] ERROR - Cleaned CSV not found: {cleaned_csv_path}", flush=True)
except Exception as e:
    print(f"\n[error] {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
