#!/usr/bin/env python3
"""Check column names in raw_data.csv"""

import pandas as pd
import glob
from pathlib import Path

runs = sorted(glob.glob(r"backend\runs\20260322T*"), key=lambda x: x, reverse=True)
latest_run = runs[0]

raw_path = Path(latest_run) / "raw_data.csv"
cleaned_path = Path(latest_run) / "cleaned_data.csv"

print(f"Run: {Path(latest_run).name}")

raw = pd.read_csv(raw_path)
cleaned = pd.read_csv(cleaned_path)

print(f"\nRaw columns: {list(raw.columns)}")
print(f"Cleaned columns: {list(cleaned.columns)}")

# Find the age-like column
age_col_raw = None
for col in raw.columns:
    if 'age' in col.lower():
        age_col_raw = col
        break

age_col_cleaned = None
for col in cleaned.columns:
    if 'age' in col.lower():
        age_col_cleaned = col
        break

print(f"\nAge column in raw: {age_col_raw}")
print(f"Age column in cleaned: {age_col_cleaned}")

if age_col_raw:
    print(f"\nRaw {age_col_raw} non-null: {raw[age_col_raw].notna().sum()}")
    print(f"Raw {age_col_raw} first 20: {raw[age_col_raw].head(20).tolist()}")

if age_col_cleaned:
    print(f"\nCleaned {age_col_cleaned} non-null: {cleaned[age_col_cleaned].notna().sum()}")
    print(f"Cleaned {age_col_cleaned} first 20: {cleaned[age_col_cleaned].head(20).tolist()}")
