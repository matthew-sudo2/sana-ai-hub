import pandas as pd
import numpy as np
import random

# Set seed for reproducibility but still create obvious corruption
random.seed(42)
np.random.seed(42)

# Load the dataset
file_path = r"traindata\raw\disuguaglianza-economica-globale-e-povert-1980-2024.csv"
df = pd.read_csv(file_path)

print(f"Original shape: {df.shape}")
print(f"Original missing ratio: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")

# Strategy 1: Remove random values (create massive missing data - target 35-40% missing)
missing_fraction = 0.38
mask = np.random.random(df.shape) < missing_fraction
df_corrupted = df.copy()
df_corrupted = df_corrupted.mask(mask)

# Strategy 2: Remove entire rows randomly (10% of rows)
rows_to_drop = np.random.choice(df_corrupted.index, size=int(len(df_corrupted) * 0.10), replace=False)
df_corrupted = df_corrupted.drop(rows_to_drop)

# Strategy 3: Introduce encoding errors by corrupting some values
# Convert numeric columns to object/string type to allow mixed data
numeric_cols = ['population', 'gdp', 'gdp_per_capita', 'poverty_rate', 'gini_index']
for col in numeric_cols:
    df_corrupted[col] = df_corrupted[col].astype('object')
    # Corrupt 5% of values with invalid strings
    corrupt_indices = np.random.choice(df_corrupted.index, size=int(len(df_corrupted) * 0.05), replace=False)
    for idx in corrupt_indices:
        if pd.notna(df_corrupted.loc[idx, col]):
            # Replace with invalid values
            bad_values = ['N/A', 'NULL', '###', '???', 'ERROR', 'MISSING']
            df_corrupted.loc[idx, col] = random.choice(bad_values)

# Strategy 4: Duplicate some rows (data quality issue)
dup_indices = np.random.choice(df_corrupted.index, size=int(len(df_corrupted) * 0.08), replace=False)
duplicates = df_corrupted.loc[dup_indices].copy()
df_corrupted = pd.concat([df_corrupted, duplicates], ignore_index=True)

# Strategy 5: Scramble some column values (inconsistent data)
if 'iso_code' in df_corrupted.columns:
    scramble_indices = np.random.choice(df_corrupted.index, size=int(len(df_corrupted) * 0.06), replace=False)
    for idx in scramble_indices:
        df_corrupted.loc[idx, 'iso_code'] = 'XXX'

# Strategy 6: Create inconsistent year format for some rows
df_corrupted['year'] = df_corrupted['year'].astype('object')
year_indices = np.random.choice(df_corrupted.index, size=int(len(df_corrupted) * 0.04), replace=False)
for idx in year_indices:
    # Add garbage to year field
    if pd.notna(df_corrupted.loc[idx, 'year']):
        df_corrupted.loc[idx, 'year'] = str(df_corrupted.loc[idx, 'year']) + 'X'

print(f"\nCorrupted shape: {df_corrupted.shape}")
print(f"Corrupted missing ratio: {df_corrupted.isnull().sum().sum() / (df_corrupted.shape[0] * df_corrupted.shape[1]):.2%}")

# Save to traindata/raw
output_path = r"traindata\raw\economic_inequality_messy_corrupted.csv"
df_corrupted.to_csv(output_path, index=False)

print(f"\n✅ Corrupted dataset saved to: {output_path}")
print(f"   Rows: {df_corrupted.shape[0]}")
print(f"   Columns: {df_corrupted.shape[1]}")
print(f"   Missing data: {df_corrupted.isnull().sum().sum()} cells ({df_corrupted.isnull().sum().sum() / (df_corrupted.shape[0] * df_corrupted.shape[1]):.2%})")
print(f"   Duplicates: {len(duplicates)}")
