#!/usr/bin/env python3
"""Check Age column in latest cleaned_data.csv"""

import pandas as pd
from pathlib import Path
import glob

# Find the latest run
runs = sorted(glob.glob(r"backend\runs\20260322T*"), key=lambda x: x, reverse=True)
if not runs:
    print("No runs found")
    exit(1)

latest_run = runs[0]
csv_path = Path(latest_run) / "cleaned_data.csv"

print(f"Latest run: {Path(latest_run).name}")
print(f"CSV path: {csv_path}")
print(f"CSV exists: {csv_path.exists()}")

if not csv_path.exists():
    print("ERROR: CSV not found!")
    exit(1)

# Read CSV and check Age column
df = pd.read_csv(csv_path)

print(f"\n✓ Data shape: {df.shape}")
print(f"\nAge column analysis:")
print(f"  Total rows: {len(df):,}")
print(f"  Non-null count: {df['age'].notna().sum():,}")
print(f"  Null count: {df['age'].isna().sum():,}")
print(f"  Completeness: {df['age'].notna().sum() / len(df) * 100:.1f}%")
print(f"  Dtype: {df['age'].dtype}")
print(f"  Min: {df['age'].min()}")
print(f"  Max: {df['age'].max()}")
print(f"\nFirst 20 Age values:")
print(df['age'].head(20).tolist())

# Check if we still have missing values
if df['age'].isna().sum() > 0:
    print(f"\n⚠️ WARNING: Still have {df['age'].isna().sum()} missing Age values!")
    missing_indices = df[df['age'].isna()].index.tolist()[:10]
    print(f"First 10 missing indices: {missing_indices}")
else:
    print(f"\n✓✓✓ SUCCESS! Age column is 100% complete! ✓✓✓")

print(f"\nPhase 2 summary:")
phase2_path = Path(latest_run) / "phase2_summary.json"
if phase2_path.exists():
    import json
    with open(phase2_path) as f:
        phase2 = json.load(f)
    print(f"  Rows: {phase2.get('rows')}")
    print(f"  Columns: {len(phase2.get('columns', []))}")
    print(f"  Age dtype reported: {phase2.get('dtypes', {}).get('age')}")
