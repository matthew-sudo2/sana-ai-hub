"""Check actual markdown content"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import pandas as pd
from backend.agents.validator import run_validation

# Create test directory
test_run_dir = Path(__file__).parent / "test_check_urls"
test_run_dir.mkdir(exist_ok=True)

# Create test data
df = pd.DataFrame({
    'A': range(10),
    'B': range(10, 20),
    'C': ['x'] * 10,
})

df.to_csv(test_run_dir / "raw_data.csv", index=False)
df.to_csv(test_run_dir / "cleaned_data.csv", index=False)

# Run validation
result = run_validation(test_run_dir)

# Check markdown
report_path = test_run_dir / "validation_report.md"
content = report_path.read_text()

print("🔍 Checking markdown image URLs:\n")

# Find all image references
import re
img_refs = re.findall(r'!\[.*?\]\((.*?)\)', content)

print(f"Found {len(img_refs)} image references:\n")
for url in img_refs:
    print(f"  {url}")

# Check for API URLs
api_refs = [url for url in img_refs if url.startswith("/api/")]
print(f"\n✓ {len(api_refs)} using absolute API paths")

relative_refs = [url for url in img_refs if not url.startswith("/") and not url.startswith("http")]
if relative_refs:
    print(f"⚠️  {len(relative_refs)} still using relative paths: {relative_refs}")

print("\n📄 Sample markdown section with images:")
ml_start = content.find("## Visualizations")
if ml_start > 0:
    ml_end = min(ml_start + 800, len(content))
    print(content[ml_start:ml_end])

# Cleanup
import shutil
shutil.rmtree(test_run_dir)
