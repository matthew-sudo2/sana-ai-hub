"""
Generate corrupted dataset variations for training the quality model.
Systematically introduces realistic data quality issues.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def corrupt_dataset(df: pd.DataFrame, 
                   corruption_level: float = 0.3,
                   seed: int = None) -> pd.DataFrame:
    """
    Introduce realistic data quality issues to a dataset.
    
    Args:
        df: Input DataFrame to corrupt
        corruption_level: Intensity of corruption (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted DataFrame
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    corrupted_df = df.copy()
    n_rows, n_cols = corrupted_df.shape
    
    # 1. Add missing values (only to numeric columns)
    numeric_cols_list = list(corrupted_df.select_dtypes(include=[np.number]).columns)
    if numeric_cols_list:
        n_missing = int(n_rows * len(numeric_cols_list) * corruption_level * 0.3)
        for _ in range(n_missing):
            row = np.random.randint(0, n_rows)
            col_name = np.random.choice(numeric_cols_list)
            corrupted_df.loc[row, col_name] = np.nan
    
    # 2. Add duplicate rows
    n_duplicates = int(n_rows * corruption_level * 0.2)
    if n_duplicates > 0:
        dup_rows = corrupted_df.sample(n=min(n_duplicates, n_rows), replace=True)
        corrupted_df = pd.concat([corrupted_df, dup_rows], ignore_index=True)
    
    # 3. Add noise to numeric columns
    numeric_cols = corrupted_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(corrupted_df[col].dropna()) > 0:
            noise_indices = np.random.choice(
                corrupted_df.index, 
                size=int(len(corrupted_df) * corruption_level * 0.2),
                replace=False
            )
            col_std = corrupted_df[col].std()
            if col_std > 0:
                noise = np.random.normal(0, col_std, len(noise_indices))
                corrupted_df.loc[noise_indices, col] += noise
    
    # 4. Introduce incorrect data types (convert some numeric to string)
    if len(numeric_cols) > 0 and corruption_level > 0.1:
        col_to_corrupt = np.random.choice(list(numeric_cols), size=1)[0]
        n_to_convert = int(len(corrupted_df) * corruption_level * 0.1)
        if n_to_convert > 0:
            indices = np.random.choice(corrupted_df.index, size=min(n_to_convert, len(corrupted_df)), replace=False)
            # Convert column to object type first to allow string assignment
            corrupted_df[col_to_corrupt] = corrupted_df[col_to_corrupt].astype(object)
            corrupted_df.loc[indices, col_to_corrupt] = "INVALID_TEXT"
    
    return corrupted_df


def generate_quality_datasets(df: pd.DataFrame, n_good: int = 8, n_bad: int = 15) -> tuple:
    """
    Generate multiple good and bad variations of a dataset.
    
    Args:
        df: Original dataset
        n_good: Number of good variations to generate
        n_bad: Number of bad variations to generate
        
    Returns:
        Tuple of (good_dfs, bad_dfs)
    """
    
    good_dfs = []
    bad_dfs = []
    
    # Generate good variations (minor cleaning)
    for i in range(n_good):
        good_df = df.copy()
        # Minimal corruption or random minor issues
        if i % 2 == 0:
            # Light missing value introduction (less than 2%)
            n_missing = int(good_df.shape[0] * good_df.shape[1] * 0.01)
            for _ in range(n_missing):
                row = np.random.randint(0, good_df.shape[0])
                col = np.random.randint(0, good_df.shape[1])
                good_df.iloc[row, col] = np.nan
        good_dfs.append(good_df)
    
    # Generate bad variations (severe corruption)
    for i in range(n_bad):
        corruption_intensity = 0.15 + (i / n_bad) * 0.35  # Range 0.15 to 0.5
        bad_df = corrupt_dataset(df, corruption_level=corruption_intensity, seed=i)
        bad_dfs.append(bad_df)
    
    return good_dfs, bad_dfs
