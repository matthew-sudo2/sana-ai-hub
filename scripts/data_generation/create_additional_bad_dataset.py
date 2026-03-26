"""
Create one more corrupted BAD dataset to balance class distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_additional_bad_dataset():
    """Create a 5th corrupted bad dataset"""
    
    good_csv = Path('data/labeled/good/games.csv')
    bad_dir = Path('data/labeled/bad')
    
    if not good_csv.exists():
        print(f"Error: {good_csv} not found")
        return
    
    print("Loading games.csv...")
    df_original = pd.read_csv(good_csv)
    
    # Corruption 5: Rows with all values at extremes (artificial outliers)
    print("\n[5] Creating corruption_extreme_outliers.csv (synthetic outliers)...")
    df5 = df_original.copy()
    
    # Add some rows with extreme values to make data suspicious
    n_extreme = int(len(df5) * 0.12)
    extreme_indices = np.random.choice(df5.index, n_extreme, replace=False)
    
    # Make certain numeric columns have extreme values
    numeric_cols = df5.select_dtypes(include=[np.number]).columns.tolist()
    for idx in extreme_indices:
        for col in numeric_cols[:3]:  # Modify first 3 numeric columns
            df5.loc[idx, col] = np.random.choice([999999, -999999])
    
    df5.to_csv(bad_dir / 'corruption_extreme_outliers.csv', index=False)
    print(f"    ✓ Saved: {len(df5)} rows with 12% extreme outliers")
    
    print("\n" + "="*60)
    print("✓ Created 5th corrupted BAD dataset")
    print("="*60)

if __name__ == '__main__':
    create_additional_bad_dataset()
