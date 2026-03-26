"""
Generate corrupted datasets smartly - with proper corruption that doesn't break encoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path



def generate_bad_datasets():
    """Generate 4 corrupted versions - simpler approach"""
    
    good_csv = Path('data/labeled/good/games.csv')
    bad_dir = Path('data/labeled/bad')
    bad_dir.mkdir(parents=True, exist_ok=True)
    
    if not good_csv.exists():
        print(f"Error: {good_csv} not found")
        return
    
    print(f"Loading {good_csv.name}...")
    df_original = pd.read_csv(good_csv)
    print(f"  Loaded {len(df_original)} rows, {len(df_original.columns)} columns")
    
    # Corruption 1: Heavy missing values (30% of rows removed)
    print("\n[1] Creating corruption_heavy_missing.csv (30% missing rows)...")
    df1 = df_original.copy()
    # Simply remove 30% of rows instead of trying to set NaN
    drop_idx = np.random.choice(df1.index, size=int(0.30 * len(df1)), replace=False)
    df1 = df1.drop(drop_idx).reset_index(drop=True)
    df1.to_csv(bad_dir / 'corruption_heavy_missing.csv', index=False)
    print(f"    ✓ Saved: {len(df1)} rows (removed 30% of original)")
    
    # Corruption 2: Duplicate rows (20%)
    print("\n[2] Creating corruption_many_duplicates.csv (20% duplicates)...")
    df2 = df_original.copy()
    n_dups = int(len(df2) * 0.20)
    if n_dups > 0:
        dup_indices = np.random.choice(df2.index, n_dups, replace=True)
        dups = df2.loc[dup_indices].reset_index(drop=True)
        df2 = pd.concat([df2, dups], ignore_index=True)
    df2.to_csv(bad_dir / 'corruption_many_duplicates.csv', index=False)
    print(f"    ✓ Saved: {len(df2)} rows (original {len(df_original)} + {n_dups} duplicates)")
    
    # Corruption 3: Mixed issues (remove 15% + add 10% duplicates)
    print("\n[3] Creating corruption_mixed_issues.csv (missing + duplicates)...")
    df3 = df_original.copy()
    # Remove 15% of rows
    drop_idx = np.random.choice(df3.index, size=int(0.15 * len(df3)), replace=False)
    df3 = df3.drop(drop_idx).reset_index(drop=True)
    # Add 10% duplicates
    n_dups = int(len(df3) * 0.10)
    if n_dups > 0:
        dup_indices = np.random.choice(df3.index, n_dups, replace=True)
        dups = df3.loc[dup_indices].reset_index(drop=True)
        df3 = pd.concat([df3, dups], ignore_index=True)
    df3.to_csv(bad_dir / 'corruption_mixed_issues.csv', index=False)
    print(f"    ✓ Saved: {len(df3)} rows with mixed corruption")
    
    # Corruption 4: Inconsistent column names + remove some rows
    print("\n[4] Creating corruption_inconsistent_columns.csv (inconsistent columns + sparse data)...")
    df4 = df_original.copy()
    # Randomly change column name case/style
    new_cols = []
    for col in df4.columns:
        r = np.random.random()
        if r > 0.66:
            new_cols.append(col.lower())
        elif r > 0.33:
            new_cols.append(col.upper())
        else:
            new_cols.append(col.replace('_', ' '))
    df4.columns = new_cols
    # Remove 20% of rows
    drop_idx = np.random.choice(df4.index, size=int(0.20 * len(df4)), replace=False)
    df4 = df4.drop(drop_idx).reset_index(drop=True)
    df4.to_csv(bad_dir / 'corruption_inconsistent_columns.csv', index=False)
    print(f"    ✓ Saved: {len(df4)} rows with inconsistent columns and 20% missing")
    
    print("\n" + "="*60)
    print("✓ Generated 4 corrupted datasets in data/labeled/bad/")
    print("="*60)
    print("\nSummary:")
    print("  1. corruption_heavy_missing.csv         - 30% rows removed")
    print("  2. corruption_many_duplicates.csv       - 20% duplicate rows added")
    print("  3. corruption_mixed_issues.csv          - 15% rows removed + 10% duplicates")
    print("  4. corruption_inconsistent_columns.csv  - Column names changed + 20% rows removed")

if __name__ == '__main__':
    generate_bad_datasets()
