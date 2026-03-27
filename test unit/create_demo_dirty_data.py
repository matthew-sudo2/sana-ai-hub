#!/usr/bin/env python3
"""
Create a demo dataset with intentional missing values and nulls.
This script adds various types of data quality issues to Amazon_stock_data.csv
to demonstrate the system's cleaning capabilities.
"""

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Read the original data
df = pd.read_csv('Amazon_stock_data.csv')
print(f"Original dataset: {len(df)} rows, {df.columns.tolist()} columns")

# Create a copy to modify and convert all to object dtype for flexibility
df_dirty = df.copy()
for col in df_dirty.columns:
    df_dirty[col] = df_dirty[col].astype('object')

# Columns to introduce missing values in
columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Date']

# Strategy: Add missing values to 600+ rows (about 8% of the dataset)
num_rows_to_corrupt = 650
corrupted_indices = np.random.choice(df_dirty.index, size=num_rows_to_corrupt, replace=False)

print(f"\nIntroducing missing values to {num_rows_to_corrupt} rows...")

# Types of corruption to apply
corruption_types = {
    'empty_string': 0,
    'nan': 0,
    'whitespace': 0,
    'partial_missing': 0,
    'entire_row': 0
}

for idx in corrupted_indices:
    corruption_type = random.choice(list(corruption_types.keys()))
    
    if corruption_type == 'empty_string':
        # Replace with empty string (5% of columns)
        cols_to_corrupt = random.sample(columns, k=random.randint(1, 2))
        for col in cols_to_corrupt:
            df_dirty.at[idx, col] = ''
        corruption_types['empty_string'] += 1
        
    elif corruption_type == 'nan':
        # Replace with NaN (10% of columns)
        cols_to_corrupt = random.sample(columns, k=random.randint(1, 3))
        for col in cols_to_corrupt:
            df_dirty.at[idx, col] = np.nan
        corruption_types['nan'] += 1
        
    elif corruption_type == 'whitespace':
        # Replace with whitespace only (3-5 spaces)
        cols_to_corrupt = random.sample(columns, k=random.randint(1, 2))
        for col in cols_to_corrupt:
            df_dirty.at[idx, col] = '   '
        corruption_types['whitespace'] += 1
        
    elif corruption_type == 'partial_missing':
        # Leave some columns, remove others (60% missing)
        cols_to_corrupt = random.sample(columns, k=random.randint(3, 5))
        for col in cols_to_corrupt:
            df_dirty.at[idx, col] = np.nan
        corruption_types['partial_missing'] += 1
        
    elif corruption_type == 'entire_row':
        # Make most of the row empty
        for col in columns:
            if random.random() > 0.3:  # 70% of columns in this row
                df_dirty.at[idx, col] = np.nan
        corruption_types['entire_row'] += 1

# Print corruption summary
print("\nCorruption applied:")
for corruption_type, count in corruption_types.items():
    print(f"  {corruption_type}: {count} rows")

# Check for duplicates and add some
print("\nAdding intentional duplicates (10 rows)...")
duplicate_indices = np.random.choice(df_dirty.index, size=10, replace=False)
duplicates = df_dirty.iloc[duplicate_indices].copy()
df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

# Add some rows with inconsistent date formats
print("Adding inconsistent date formats (5 rows)...")
date_issues_indices = np.random.choice(df_dirty.index[:100], size=5, replace=False)
for idx in date_issues_indices:
    # Change date format inconsistently
    date_str = str(df_dirty.at[idx, 'Date'])
    if '1997' in date_str or '1998' in date_str:
        # Convert to different format
        parts = date_str.split('-')
        if len(parts) == 3:
            df_dirty.at[idx, 'Date'] = f"{parts[1]}/{parts[2]}/{parts[0]}"

# Save the corrupted dataset
output_file = 'Amazon_stock_data.csv'
df_dirty.to_csv(output_file, index=False)

print(f"\n✅ Demo dataset saved to {output_file}")
print(f"   Total rows (with duplicates): {len(df_dirty)}")
print(f"   Corrupted rows: {num_rows_to_corrupt} ({100*num_rows_to_corrupt/len(df_dirty):.1f}%)")
print(f"\nData quality before cleaning:")
print(f"   Total missing values: {df_dirty.isnull().sum().sum()}")
print(f"   Per column:")
for col in df_dirty.columns:
    missing = df_dirty[col].isnull().sum()
    if missing > 0:
        print(f"     {col}: {missing} missing ({100*missing/len(df_dirty):.1f}%)")
