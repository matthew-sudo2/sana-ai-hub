#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, '.')

# Just check if the latest run saved Age properly
import pandas as pd

runs = sorted(Path('runs').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
for run_dir in runs[:3]:
    csv_path = run_dir / 'cleaned_data.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'age' in df.columns:
            non_null = df['age'].notna().sum()
            total = len(df)
            print(f"{run_dir.name}: age = {non_null}/{total}")
            if non_null == total:
                print("  SUCCESS")
                break
