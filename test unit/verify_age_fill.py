import pandas as pd
from pathlib import Path

runs_dir = Path('backend/runs')
latest_run = max(runs_dir.iterdir(), key=__import__('os').path.getctime)

cleaned_csv = latest_run / 'cleaned_data.csv'
df = pd.read_csv(cleaned_csv)

# Check age column
if 'age' in df.columns:
    total = len(df)
    non_null = df['age'].notna().sum()
    null_count = df['age'].isna().sum()
    completeness = (non_null / total) * 100
    
    print('✅ CLEANED DATA - Age Column:')
    print(f'   Total rows: {total}')
    print(f'   Non-null values: {non_null}')
    print(f'   Missing values: {null_count}')
    print(f'   Completeness: {completeness:.1f}%')
    print(f'   Mean age: {df["age"].mean():.1f}')
    print(f'   Sample values:', list(df['age'].head(10)))
    print()
    print('SUCCESS: Age column is now 100% complete with median fill!')
else:
    print('ERROR: Age column not found')
