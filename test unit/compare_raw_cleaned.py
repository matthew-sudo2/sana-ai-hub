#!/usr/bin/env python3
"""Compare raw vs cleaned Age column"""

import pandas as pd
import glob
from pathlib import Path

runs = sorted(glob.glob(r"backend\runs\20260322T*"), key=lambda x: x, reverse=True)
latest_run = runs[0]

raw_path = Path(latest_run) / "raw_data.csv"
cleaned_path = Path(latest_run) / "cleaned_data.csv"

print(f"Run: {Path(latest_run).name}")
print(f"\nComparing raw vs cleaned Age column:")

raw = pd.read_csv(raw_path)
cleaned = pd.read_csv(cleaned_path)

print(f"Raw Age non-null: {raw['age'].notna().sum()}")
print(f"Cleaned Age non-null: {cleaned['age'].notna().sum()}")
print(f"\nRaw Age first 20: {raw['age'].head(20).tolist()}")
print(f"Cleaned Age first 20: {cleaned['age'].head(20).tolist()}")

# Check if they're the same
if (raw['age'] == cleaned['age']).all() or (raw['age'].isna() == cleaned['age'].isna()).all():
    print(f"\n✗ Raw and cleaned are IDENTICAL - no filling happening!")
else:
    print(f"\n✓ Raw and cleaned are DIFFERENT - filling is working!")
    
# Check if the fill is partial
filled_count = cleaned['age'].notna().sum() - raw['age'].notna().sum()
print(f"\nValues filled: {filled_count} (expected: 211)")
