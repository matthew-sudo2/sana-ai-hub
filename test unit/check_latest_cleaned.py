import pandas as pd
from pathlib import Path

runs_dir = Path('backend/runs')
latest_run = max(runs_dir.iterdir(), key=__import__('os').path.getctime)

print(f'Latest run: {latest_run.name}')
print()

# Check cleaned data
cleaned_csv = latest_run / 'cleaned_data.csv'
df = pd.read_csv(cleaned_csv)

if 'age' in df.columns:
    print('Age column in cleaned_data.csv:')
    print(f'  Non-null: {df["age"].notna().sum()}')
    print(f'  Missing: {df["age"].isna().sum()}')
    print(f'  Sample: {df["age"].head(10).tolist()}')
    completeness = (df["age"].notna().sum() / len(df)) * 100
    print(f'  Completeness: {completeness:.1f}%')
else:
    print(f'Age column NOT found')
    print(f'Columns available: {list(df.columns)}')
