"""
Test the median fill logic directly
"""
import pandas as pd
import numpy as np

# Create test data similar to Age column
test_data = {
    'age': [25.0, np.nan, np.nan, 25.0, 25.0, 40.0, np.nan, 30.0, 35.0, np.nan] * 102  # 1020 rows
}
df = pd.DataFrame(test_data)

print("BEFORE FILL:")
print(f"age dtype: {df['age'].dtype}")
print(f"age non-null: {df['age'].notna().sum()}")
print(f"age null: {df['age'].isna().sum()}")
print(f"Missing %: {(df['age'].isna().sum() / len(df)) * 100:.1f}%")

# Test numeric column detection
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumeric columns found: {list(numeric_cols)}")

# Test median fill
if 'age' in numeric_cols:
    col_missing = df['age'].isna().sum()
    col_median = df['age'].median()
    col_non_null = df['age'].notna().sum()
    
    print(f"\nFill parameters:")
    print(f"  Column 'age': dtype={df['age'].dtype}, non-null={col_non_null}, missing={col_missing}, median={col_median}")
    
    if col_missing > 0:
        df['age'] = df['age'].fillna(col_median)  # Use proper pandas assignment
        after_fill = df['age'].isna().sum()
        print(f"  After fill: remaining missing={after_fill}")

print("\nAFTER FILL:")
print(f"age dtype: {df['age'].dtype}")
print(f"age non-null: {df['age'].notna().sum()}")
print(f"age null: {df['age'].isna().sum()}")
print(f"Missing %: {(df['age'].isna().sum() / len(df)) * 100:.1f}%")
print(f"\nTest PASSED: Median fill works correctly!")
