"""
Extract statistical meta-features from datasets for quality assessment.
These 6 features capture different aspects of dataset quality.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew


def extract_quality_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 6 quality features from a dataset.
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        numpy array of shape (1, 6) with features:
        [missing_ratio, duplicate_ratio, numeric_ratio, constant_columns, avg_variance, avg_skewness]
    """
    
    # 1. Missing ratio: proportion of missing values
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0.0
    
    # 2. Duplicate ratio: proportion of duplicate rows
    total_rows = df.shape[0]
    duplicate_rows = df.duplicated().sum()
    duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0.0
    
    # 3. Numeric ratio: proportion of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / df.shape[1] if df.shape[1] > 0 else 0.0
    
    # 4. Constant columns: number of columns with only one unique value
    constant_columns = sum(1 for col in df.columns if df[col].nunique() <= 1)
    
    # 5. Average variance: mean variance of numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_cols) > 0:
        avg_variance = numeric_df.var().mean()
        avg_variance = 0.0 if pd.isna(avg_variance) else avg_variance
    else:
        avg_variance = 0.0
    
    # 6. Average skewness: mean absolute skewness of numeric columns
    if len(numeric_cols) > 0:
        skewness_values = []
        for col in numeric_cols:
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                s = skew(col_data)
                skewness_values.append(abs(s))
        avg_skewness = np.mean(skewness_values) if skewness_values else 0.0
    else:
        avg_skewness = 0.0
    
    # Return as (1, 6) array
    features = np.array([
        missing_ratio,
        duplicate_ratio,
        numeric_ratio,
        constant_columns,
        avg_variance,
        avg_skewness
    ]).reshape(1, -1)
    
    return features
