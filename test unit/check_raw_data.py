import pandas as pd
from pathlib import Path

runs_dir = Path('backend/runs')
latest_run = max(runs_dir.iterdir(), key=__import__('os').path.getctime)

# Read raw_data and check structure
raw_csv = latest_run / 'raw_data.csv'
df_raw = pd.read_csv(raw_csv)

print('RAW_DATA.CSV on disk (what scout saved):')
print(f'  Employee_ID first 3: {df_raw["Employee_ID"].head(3).tolist()}')
print(f'  Employee_ID dtype: {df_raw["Employee_ID"].dtype}')
print(f'  Age first 3: {df_raw["Age"].head(3).tolist()}')
print(f'  Age missing: {df_raw["Age"].isna().sum()}')
