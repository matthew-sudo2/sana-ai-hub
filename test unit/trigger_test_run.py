#!/usr/bin/env python3
"""Trigger a test pipeline run to verify Bug #4 fix."""

import requests
import json
import time
from pathlib import Path

csv_path = Path("backend/data/Messy_Employee_dataset.csv").resolve()

print(f"Uploading: {csv_path}")
print(f"File exists: {csv_path.exists()}")

# Start the pipeline run
with open(csv_path, 'rb') as f:
    files = {'file': f}
    resp = requests.post('http://localhost:8000/run', files=files, data={'source': str(csv_path)})

result = resp.json()
print(f"\n[OK] Status: {resp.status_code}")
print(f"[OK] Run ID: {result.get('run_id')}")
print(f"\nFull response:")
print(json.dumps(result, indent=2))

# Poll for completion
run_id = result.get('run_id')
if run_id:
    print(f"\nWaiting for pipeline to complete...")
    for attempt in range(60):  # Max 3 minutes
        time.sleep(2)
        status_resp = requests.get(f'http://localhost:8000/runs/{run_id}')
        status_result = status_resp.json()
        phase = status_result.get('phase')
        state = status_result.get('state')
        print(f"  [{attempt+1}] Phase: {phase}, State: {state}")
        
        if state == 'COMPLETED':
            print(f"\n[SUCCESS] PIPELINE COMPLETED")
            break
        elif state == 'FAILED':
            print(f"\n[ERROR] PIPELINE FAILED")
            print(f"Error: {status_result.get('error')}")
            break
        elif attempt >= 59:
            print(f"\nTimeout waiting for completion")
            break

print("\n[OK] Done. Check: backend/runs/ for output")
