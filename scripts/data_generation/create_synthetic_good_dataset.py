"""
Create a synthetic clean dataset with low variance (like chess ratings).
This teaches the model that GOOD data can have low variance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_synthetic_good_dataset():
    """Generate 5000 clean customer transaction records"""
    
    np.random.seed(42)
    n_rows = 5000
    
    print("Creating synthetic good dataset...")
    
    data = {
        'customer_id': np.arange(1, n_rows + 1),
        'transaction_date': pd.date_range('2025-01-01', periods=n_rows, freq='h'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_rows),
        'amount': np.random.normal(50, 15, n_rows).clip(5, 200).round(2),
        'quantity': np.random.randint(1, 10, n_rows),
        'customer_rating': np.random.randint(1, 6, n_rows),  # 1-5: LOW variance like chess
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'is_member': np.random.choice([True, False], n_rows),
        'loyalty_points': np.random.randint(0, 1000, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()} (ZERO - GOOD)")
    print(f"  Duplicates: {df.duplicated().sum()} (ZERO - GOOD)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Variance (mean): {df[numeric_cols].var().mean():.2f} (LOW - like chess)")
    print(f"  Skewness (mean): {df[numeric_cols].skew().mean():.3f}")
    
    # Save
    out_path = Path('data/labeled/good/synthetic_clean_transactions.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"\n✓ Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    create_synthetic_good_dataset()
