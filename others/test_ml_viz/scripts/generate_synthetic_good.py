#!/usr/bin/env python3
"""
Generate high-quality synthetic GOOD datasets by:
1. Bootstrap sampling from real GOOD datasets
2. Adding controlled variations to create realistic variants
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_good_bootstrap(df, num_samples=None, mutation_ratio=0.05):
    """Bootstrap synthetic good data by sampling with replacement + small mutations"""
    if num_samples is None:
        num_samples = len(df)
    
    df_synth = df.sample(n=num_samples, replace=True).reset_index(drop=True)
    
    # Small mutations (add noise) while keeping data quality high
    numeric_cols = df_synth.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Small random noise (±5% of std)
        noise = np.random.normal(0, df_synth[col].std() * 0.05, len(df_synth))
        df_synth[col] = df_synth[col] + noise
    
    return df_synth

def generate_synthetic_good_interpolate(df, num_samples=None):
    """Generate synthetic good data by interpolating between real samples"""
    if num_samples is None:
        num_samples = len(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    str_cols = df.select_dtypes(include=['object']).columns
    
    df_synth = pd.DataFrame()
    
    for _ in range(num_samples):
        # Pick 2 random rows
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        row1, row2 = df.iloc[idx1], df.iloc[idx2]
        
        new_row = {}
        
        # Interpolate numeric columns
        for col in numeric_cols:
            alpha = np.random.uniform(0, 1)
            new_row[col] = row1[col] * alpha + row2[col] * (1 - alpha)
        
        # Copy categorical columns
        for col in str_cols:
            new_row[col] = np.random.choice([row1[col], row2[col]])
        
        df_synth = pd.concat([df_synth, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_synth

def main():
    good_dir = 'data/labeled/good'
    
    print('\n' + '='*80)
    print('GENERATING SYNTHETIC GOOD DATASETS')
    print('='*80)
    
    # Load real GOOD datasets (exclude existing synthetic ones)
    good_files = [f for f in os.listdir(good_dir) if f.endswith('.csv') and not f.startswith('synthetic')]
    
    print(f"\nLoading {len(good_files)} real GOOD datasets...")
    
    creation_methods = [
        ('bootstrap', generate_synthetic_good_bootstrap),
        ('interpolate', generate_synthetic_good_interpolate),
    ]
    
    created_count = 0
    
    # Create 2 synthetic variants from each real GOOD dataset
    for file in good_files:
        filepath = os.path.join(good_dir, file)
        if not os.path.exists(filepath):
            continue
        
        try:
            df = pd.read_csv(filepath)
            print(f"\n✓ {file} ({len(df)} rows, {len(df.columns)} cols)")
            
            # Create 2 synthetic variants per file
            for method_name, method_func in creation_methods:
                try:
                    df_synth = method_func(df)
                    
                    base_name = Path(file).stem
                    output_name = f"synthetic_good_{method_name}_{base_name}.csv"
                    output_path = os.path.join(good_dir, output_name)
                    
                    df_synth.to_csv(output_path, index=False)
                    print(f"  ├─ {output_name} ({len(df_synth)} rows)")
                    created_count += 1
                except Exception as e:
                    print(f"  ├─ {method_name}: Error - {str(e)[:50]}")
        except Exception as e:
            print(f"❌ {file}: {e}")
    
    print(f"\n" + "="*80)
    print(f"✓ CREATED {created_count} SYNTHETIC GOOD DATASETS")
    print("="*80)
    
    # List final GOOD directory
    print(f"\nGOOD datasets in {good_dir}:")
    good_files_final = sorted([f for f in os.listdir(good_dir) if f.endswith('.csv')])
    real_count = sum(1 for f in good_files_final if not f.startswith('synthetic'))
    synth_count = sum(1 for f in good_files_final if f.startswith('synthetic'))
    
    for i, f in enumerate(good_files_final, 1):
        tag = "[REAL]" if not f.startswith('synthetic') else "[SYNTH]"
        print(f"  {i:2d}. {tag} {f}")
    
    print(f"\nSummary:")
    print(f"  • Real GOOD:      {real_count}")
    print(f"  • Synthetic GOOD: {synth_count}")
    print(f"  • Total GOOD:     {real_count + synth_count}")
    
    bad_dir = 'data/labeled/bad'
    bad_files = [f for f in os.listdir(bad_dir) if f.endswith('.csv')]
    print(f"  • Total BAD:      {len(bad_files)}")
    print(f"\n✅ Ready for balanced retraining: {real_count + synth_count} GOOD vs {len(bad_files)} BAD")

if __name__ == '__main__':
    main()
