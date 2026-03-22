#!/usr/bin/env python3
"""Test labeler exit code with fixed ensure_ascii"""
import subprocess
import sys
from pathlib import Path

# Run labeler directly as subprocess
scout_file = Path("backend/runs/20260322T125309Z_C__Users_User_Documents_GitHub_Information-Management-Finals-E-Commerce-Dashboar/scout_result.json")

if not scout_file.exists():
    print(f"Scout file not found: {scout_file}")
    sys.exit(1)

print(f"Running labeler with scout: {scout_file.name}")
cmd = [sys.executable, "backend/agents/labeler.py", str(scout_file)]

result = subprocess.run(cmd, capture_output=True, text=True)
print(f"\nExit code: {result.returncode}")
print(f"\nStdout (last 500 chars):\n{result.stdout[-500:]}\n")
if result.stderr:
    print(f"Stderr (last 500 chars):\n{result.stderr[-500:]}\n")

# Check if CSV was filled
if result.returncode == 0:
    import pandas as pd
    csv_path = Path("backend/runs/20260322T125309Z_C__Users_User_Documents_GitHub_Information-Management-Finals-E-Commerce-Dashboar/cleaned_data.csv")
    df = pd.read_csv(csv_path)
    non_null = df['age'].notna().sum()
    print(f"\nCleaned CSV Age: {non_null}/1020")
    if non_null == 1020:
        print("SUCCESS!")
