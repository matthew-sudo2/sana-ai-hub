"""
Test to reproduce and debug the Age fill issue step-by-step
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the actual raw_data from the latest run
runs_dir = Path('backend/runs')
latest_run = max(runs_dir.iterdir(), key=__import__('os').path.getctime)
raw_data_path = latest_run / 'raw_data.csv'

print("="*80)
print("REPRODUCING THE LABELER MEDIAN FILL LOGIC")
print("="*80)

# Load data exactly as labeler does
df = pd.read_csv(raw_data_path)

print(f"\n1️⃣  INITIAL STATE (from raw_data.csv):")
print(f"   Age dtype: {df['Age'].dtype}")
print(f"   Age non-null: {df['Age'].notna().sum()}")
print(f"   Age missing: {df['Age'].isna().sum()}")
print(f"   Age sample: {df['Age'].head(5).tolist()}")

# Standardize column names (like labeler does)
df.columns = [__import__('re').sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_') for c in df.columns]

print(f"\n2️⃣  AFTER STANDARDIZING COLUMN NAMES:")
print(f"   Columns: {list(df.columns)}")
if 'age' in df.columns:
    print(f"   age (lowercase) dtype: {df['age'].dtype}")
    print(f"   age non-null: {df['age'].notna().sum()}")
    print(f"   age missing: {df['age'].isna().sum()}")
    print(f"   age sample: {df['age'].head(5).tolist()}")

# Now apply the median fill EXACTLY as labeler.py does
print(f"\n3️⃣  APPLYING MEDIAN FILL:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"   Numeric columns: {list(numeric_cols)}")

missing_before = df.isna().sum().sum()
print(f"   Total missing before: {missing_before}")

for col in numeric_cols:
    col_missing = df[col].isna().sum()
    if col_missing > 0:
        col_median = df[col].median()
        print(f"\n   Processing '{col}':")
        print(f"      Missing: {col_missing}, median: {col_median}")
        print(f"      Before: {df[col].isna().sum()} missing")
        
        # This is the exact line from labeler.py
        df[col] = df[col].fillna(col_median)
        
        after_fill = df[col].isna().sum()
        print(f"      After: {after_fill} missing ✓")

missing_after = df.isna().sum().sum()
print(f"\n   Total missing after: {missing_after}")
print(f"   Filled: {missing_before - missing_after}")

print(f"\n4️⃣  FINAL STATE (in-memory df):")
if 'age' in df.columns:
    print(f"   age non-null: {df['age'].notna().sum()}")
    print(f"   age missing: {df['age'].isna().sum()}")
    print(f"   age sample: {df['age'].head(5).tolist()}")

# Now save to CSV and reload to verify
test_csv = Path('test_cleaned_age.csv')
df.to_csv(test_csv, index=False)
print(f"\n5️⃣  SAVED TO CSV AND RELOADED:")

df_back = pd.read_csv(test_csv)
if 'age' in df_back.columns:
    print(f"   age non-null: {df_back['age'].notna().sum()}")
    print(f"   age missing: {df_back['age'].isna().sum()}")
    print(f"   age sample: {df_back['age'].head(5).tolist()}")
    if df_back['age'].isna().sum() == 0:
        print("\n✅ SUCCESS: Age column properly filled and saved!")
    else:
        print("\n❌ PROBLEM: Age column lost values during save!")
else:
    print("❌ Age column not found")

test_csv.unlink()  # Clean up
