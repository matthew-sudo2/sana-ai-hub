import pandas as pd
import os
from pathlib import Path

# Find the most recent run directory
runs_dir = Path("backend/runs")
latest_run = max(runs_dir.iterdir(), key=os.path.getctime)
print(f"Latest run: {latest_run.name}")

raw_data_path = latest_run / "raw_data.csv"
cleaned_data_path = latest_run / "cleaned_data.csv"

print("\n=== RAW_DATA.CSV ===")
df_raw = pd.read_csv(raw_data_path)
print(f"Age dtype: {df_raw['Age'].dtype}")
print(f"Age non-null count: {df_raw['Age'].notna().sum()} out of {len(df_raw)}")
print(f"Age null count: {df_raw['Age'].isna().sum()}")
print(f"Age unique values (first 30): {df_raw['Age'].unique()[:30]}")
print(f"\nAge sample (first 20 rows):")
print(df_raw['Age'].head(20).tolist())

print("\n=== CLEANED_DATA.CSV ===")
df_cleaned = pd.read_csv(cleaned_data_path)
print(f"Columns in cleaned_data.csv: {list(df_cleaned.columns)}")
if 'Age' in df_cleaned.columns:
    print(f"Age dtype: {df_cleaned['Age'].dtype}")
    print(f"Age non-null count: {df_cleaned['Age'].notna().sum()} out of {len(df_cleaned)}")
    print(f"Age null count: {df_cleaned['Age'].isna().sum()}")
    print(f"Age unique values (first 30): {df_cleaned['Age'].unique()[:30]}")
    print(f"\nAge sample (first 20 rows):")
    print(df_cleaned['Age'].head(20).tolist())
else:
    print("ERROR: Age column NOT found in cleaned_data.csv!")
    # Look for similar columns
    similar = [c for c in df_cleaned.columns if 'age' in c.lower()]
    print(f"Columns containing 'age': {similar}")

# Check if they're the same
print("\n=== COMPARISON ===")
print(f"Raw and cleaned are identical: {df_raw.equals(df_cleaned)}")
print(f"Raw size: {len(df_raw)}, Cleaned size: {len(df_cleaned)}")

# Check ALL columns dtype in raw
print("\n=== ALL COLUMNS DTYPES (RAW) ===")
for col in df_raw.columns:
    print(f"{col}: {df_raw[col].dtype}")
