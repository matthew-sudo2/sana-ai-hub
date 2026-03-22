"""
Complete diagnostic of the Age column through the pipeline
"""
import pandas as pd
import json
from pathlib import Path
import sys

# Get latest run
runs_dir = Path("backend/runs")
latest_run = max(runs_dir.iterdir(), key=__import__('os').path.getctime)
print(f"\n{'='*80}")
print(f"LATEST RUN: {latest_run.name}")
print(f"{'='*80}\n")

# Check Scout Result
print("1. SCOUT RESULT")
print("-" * 80)
scout_result_path = latest_run / "scout_result.json"
with open(scout_result_path) as f:
    scout_data = json.load(f)
    print(f"Columns in scout: {scout_data.get('columns', [])[:5]}...")
    print(f"Raw data path: {scout_data.get('raw_data_path')}")

# Check Raw Data
print("\n2. RAW_DATA.CSV (Scout output, Labeler input)")
print("-" * 80)
raw_data_path = latest_run / "raw_data.csv"
df_raw = pd.read_csv(raw_data_path)
print(f"Shape: {df_raw.shape}")
print(f"Age dtype: {df_raw['Age'].dtype}")
print(f"Age non-null: {df_raw['Age'].notna().sum()} / {len(df_raw)}")
print(f"Age null: {df_raw['Age'].isna().sum()}")
print(f"Age sample (first 10): {df_raw['Age'].head(10).tolist()}")
print(f"All columns: {list(df_raw.columns)}")

# Check Phase 2 Summary (Labeler output)
print("\n3. PHASE 2 SUMMARY (Labeler metadata)")
print("-" * 80)
phase2_path = latest_run / "phase2_summary.json"
with open(phase2_path) as f:
    phase2_data = json.load(f)
    print(json.dumps(phase2_data, indent=2))

# Check Cleaned Data
print("\n4. CLEANED_DATA.CSV (Labeler output)")
print("-" * 80)
cleaned_data_path = latest_run / "cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_data_path)
print(f"Shape: {df_cleaned.shape}")
print(f"Columns: {list(df_cleaned.columns)}")
if 'age' in df_cleaned.columns:
    print(f"age dtype: {df_cleaned['age'].dtype}")
    print(f"age non-null: {df_cleaned['age'].notna().sum()} / {len(df_cleaned)}")
    print(f"age null: {df_cleaned['age'].isna().sum()}")
    print(f"age sample (first 10): {df_cleaned['age'].head(10).tolist()}")
if 'Age' in df_cleaned.columns:
    print(f"Age dtype: {df_cleaned['Age'].dtype}")
    print(f"Age non-null: {df_cleaned['Age'].notna().sum()} / {len(df_cleaned)}")
    print(f"Age null: {df_cleaned['Age'].isna().sum()}")
    print(f"Age sample (first 10): {df_cleaned['Age'].head(10).tolist()}")

# Comparison
print("\n5. BEFORE vs AFTER COMPARISON")
print("-" * 80)
print(f"Row count changed: {len(df_raw)} -> {len(df_cleaned)}")
print(f"Column count changed: {len(df_raw.columns)} -> {len(df_cleaned.columns)}")
print(f"Are dataframes equal: {df_raw.equals(df_cleaned)}")

# Check Analysis Result
print("\n6. ANALYSIS RESULT (Data Quality)")
print("-" * 80)
analysis_path = latest_run / "analysis_result.json"
with open(analysis_path) as f:
    analysis_data = json.load(f)
    col_stats = analysis_data.get('column_stats', [])
    for stat in col_stats:
        if stat['column'] == 'age':
            print(f"Age stats from analytics:")
            print(f"  dtype: {stat['dtype']}")
            print(f"  missing_pct: {stat['missing_pct']}%")
            print(f"  missing_count: {stat['missing_count']}")
            print(f"  non-null would be: {len(df_cleaned) - stat['missing_count']}")
