#!/usr/bin/env python3
"""
Create corrupted variants from new GOOD datasets to balance BAD dataset count
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def corrupt_missing_values(df, missing_ratio=0.2):
    """Randomly drop values"""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        null_count = int(len(df_copy) * missing_ratio)
        null_indices = np.random.choice(df_copy.index, size=null_count, replace=False)
        df_copy.loc[null_indices, col] = np.nan
    return df_copy

def corrupt_duplicates(df, dup_ratio=0.15):
    """Add duplicate rows"""
    df_copy = df.copy()
    dup_count = int(len(df_copy) * dup_ratio)
    dup_indices = np.random.choice(df_copy.index, size=dup_count, replace=False)
    dup_rows = df_copy.loc[dup_indices].copy()
    df_copy = pd.concat([df_copy, dup_rows], ignore_index=True)
    return df_copy

def corrupt_outliers(df, outlier_ratio=0.1):
    """Introduce extreme outliers"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out_count = int(len(df_copy) * outlier_ratio)
        out_indices = np.random.choice(df_copy.index, size=out_count, replace=False)
        col_max = df_copy[col].max()
        col_min = df_copy[col].min()
        df_copy.loc[out_indices, col] = np.random.uniform(col_max * 2, col_max * 5, out_count)
    return df_copy

def corrupt_type_mixing(df, ratio=0.1):
    """Mix data types in columns"""
    df_copy = df.copy()
    numeric_cols = list(df_copy.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return df_copy
    
    col = np.random.choice(numeric_cols)
    corrupt_count = int(len(df_copy) * ratio)
    corrupt_indices = np.random.choice(df_copy.index, size=corrupt_count, replace=False)
    df_copy.loc[corrupt_indices, col] = "CORRUPTED_TEXT"
    return df_copy

def corrupt_combined(df):
    """Multiple corruption types"""
    df_copy = df.copy()
    df_copy = corrupt_missing_values(df_copy, 0.15)
    df_copy = corrupt_duplicates(df_copy, 0.1)
    return df_copy

def main():
    good_dir = 'data/labeled/good'
    bad_dir = 'data/labeled/bad'
    
    # New datasets to corrupt
    new_files = [
        'AI_Impact_on_Jobs_2030.csv',
        'Amazon_stock_data.csv',
        'Exam_Score_Prediction.csv',
        'student_exam_scores.csv'
    ]
    
    print('\n' + '='*80)
    print('CREATING CORRUPTED VARIANTS OF NEW GOOD DATASETS')
    print('='*80)
    
    corruption_types = [
        ('heavy_missing', corrupt_missing_values),
        ('many_duplicates', corrupt_duplicates),
        ('extreme_outliers', corrupt_outliers),
        ('type_mixing', corrupt_type_mixing),
        ('mixed_issues', corrupt_combined),
    ]
    
    created_count = 0
    
    for file in new_files:
        filepath = os.path.join(good_dir, file)
        if not os.path.exists(filepath):
            print(f"\n❌ {file}: NOT FOUND")
            continue
        
        try:
            df = pd.read_csv(filepath)
            print(f"\n✓ Loaded {file} ({len(df)} rows, {len(df.columns)} cols)")
            
            # Create 2 corrupted variants per file
            for i, (corr_type, corr_func) in enumerate(corruption_types[:2]):
                corrupted_df = corr_func(df)
                
                base_name = Path(file).stem
                output_name = f"corruption_{corr_type}_{base_name}.csv"
                output_path = os.path.join(bad_dir, output_name)
                
                corrupted_df.to_csv(output_path, index=False)
                print(f"  ├─ {output_name} ({len(corrupted_df)} rows)")
                created_count += 1
                
        except Exception as e:
            print(f"  └─ Error: {e}")
    
    print(f"\n" + "="*80)
    print(f"✓ CREATED {created_count} CORRUPTED VARIANTS")
    print("="*80)
    
    # List final BAD directory
    print(f"\nBAD datasets in {bad_dir}:")
    bad_files = sorted([f for f in os.listdir(bad_dir) if f.endswith('.csv')])
    for i, f in enumerate(bad_files, 1):
        print(f"  {i}. {f}")

if __name__ == '__main__':
    main()
