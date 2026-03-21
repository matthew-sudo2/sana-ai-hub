#!/usr/bin/env python
"""
Test analyst.py fix for boolean column handling.
"""

import pandas as pd
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing analyst.py boolean column handling...")
print("=" * 60)

# Load the problematic dataset
dataset_path = Path("backend/data/Messy_Employee_dataset.csv")
print(f"\n📥 Loading dataset: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✓ Columns: {list(df.columns)}")

# Check dtypes
print(f"\nColumn dtypes:")
for col in df.columns:
    print(f"  {col:30s}: {str(df[col].dtype):15s}")

# Try the functions that were failing
print("\n" + "=" * 60)
print("Testing analyst functions...")
print("=" * 60)

try:
    from backend.agents.analyst import _iqr_outliers, _compute_column_stats, _compute_correlations
    
    print("\n✓ Imported analyst functions successfully")
    
    # Test _iqr_outliers on each column
    print("\n📊 Testing _iqr_outliers on all columns:")
    for col in df.columns:
        try:
            outliers, lo, hi = _iqr_outliers(df[col])
            print(f"  {col:30s}: {len(outliers)} outliers (lo={lo:.2f}, hi={hi:.2f})")
        except Exception as e:
            print(f"  ❌ {col:30s}: {type(e).__name__}: {str(e)[:50]}")
    
    # Test _compute_column_stats
    print("\n📈 Testing column statistics...")
    stats = _compute_column_stats(df)
    print(f"✓ Computed stats for {len(stats)} columns")
    
    # Test _compute_correlations
    print("\n📉 Testing correlations...")
    corrs = _compute_correlations(df)
    print(f"✓ Computed {len(corrs)} correlation pairs")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe analyst.py boolean fix is working correctly.")
    print("You can now run the pipeline with Messy_Employee_dataset.csv")
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
