#!/usr/bin/env python3
"""
Demonstration of the NEW validation report format vs OLD format.
Shows how the redesigned report is much more concise and readable.
"""

# Simulated validation result
sample_report_old = """# Validation Report

## Summary
- **Status:** APPROVED
- **Overall Confidence:** 92.5%
- **Source:** `C:\\Users\\User\\Documents\\...\\data.csv`
- **Type:** csv
- **Rows / Columns:** 1000 / 5
- **Charts Validated:** 2
- **Generated:** 2026-03-21T16:31:16.056495+00:00

## Dimension Scores
- completeness: 100.0% (weight 25%, 8/8)
- sanity: 78.4% (weight 35%, 5/8)
- consistency: 100.0% (weight 25%, 22/22)
- visualization: 100.0% (weight 15%, 7/7)

## Warnings
- ['State']
- 1000 out-of-range values

## Check Details
- **PASS** | `completeness` | `error` | Scout output JSON present and non-empty
- **PASS** | `completeness` | `error` | Labeler cleaned CSV present and non-empty
- **PASS** | `completeness` | `error` | Analysis result JSON present and non-empty
- **PASS** | `completeness` | `warning` | Analysis markdown report present and non-empty
- **PASS** | `completeness` | `warning` | Enriched CSV present and non-empty
- **PASS** | `completeness` | `error` | Visualization summary JSON present and non-empty
- **PASS** | `completeness` | `warning` | Labeler summary present and non-empty
- **PASS** | `completeness` | `warning` | Analysis summary present and non-empty
- **PASS** | `sanity` | `error` | Row count matches analysis
- **PASS** | `sanity` | `warning` | Column count matches analysis
- **PASS** | `sanity` | `warning` | No fully-empty rows
- **FAIL** | `sanity` | `warning` | No fully-empty columns (['State'])
- **FAIL** | `sanity` | `info` | No duplicate rows (2 duplicate rows)
- **FAIL** | `sanity` | `warning` | Percent-like `Administration` in [0,100] (1000 out-of-range values)
- **PASS** | `sanity` | `error` | At least 3 rows
- **PASS** | `sanity` | `error` | At least 2 columns
... (and many more)
"""

sample_report_new = """# Validation Report

**Status:** APPROVED | **Confidence:** 92.5%

## Dataset Information
- Rows: 1,000 | Columns: 5
- Source: C:/Users/User/Documents/.../data.csv
- Charts Validated: 2

## Quality Scores
**✓ Completeness:** 100.0% (8/8 passed)
**⚠ Sanity:** 78.4% (5/8 passed)
**✓ Consistency:** 100.0% (22/22 passed)
**✓ Visualization:** 100.0% (7/7 passed)

## Issues Found
### Sanity (3 issues)
- 🟡 No fully-empty columns — State
- 🔴 No duplicate rows — 2 duplicate rows
- 🟡 Percent-like `Administration` in [0,100] — 1000 out-of-range values

---
**Validation Complete:** 37/40 checks passed
**Generated:** 2026-03-21T16:31:16.056495+00:00
"""

print("=" * 80)
print("VALIDATION REPORT FORMAT COMPARISON")
print("=" * 80)

print("\n❌ OLD FORMAT (Verbose, Cluttered, All 40+ checks listed):")
print("-" * 80)
print(sample_report_old)

print("\n\n✅ NEW FORMAT (Clean, Compact, Only issues highlighted):")
print("-" * 80)
print(sample_report_new)

print("\n\n📊 COMPARISON:")
print("-" * 80)
old_lines = sample_report_old.split("\n")
new_lines = sample_report_new.split("\n")
print(f"Old format lines: {len(old_lines)}")
print(f"New format lines: {len(new_lines)}")
print(f"Reduction: {len(old_lines) - len(new_lines)} fewer lines ({100*(len(old_lines)-len(new_lines))/len(old_lines):.0f}% shorter)")

print("\n📌 KEY IMPROVEMENTS:")
print("-" * 80)
print("✓ Shows ONLY failed checks (not all 40+ passing checks)")
print("✓ Grouped by category with counts")
print("✓ Status icons (✓ ✗ ⚠️ 🔴 🟡) for quick visual scanning")
print("✓ Clean, readable format without pipes and backticks")
print("✓ Concise dataset summary with essential metrics")
print("✓ Much shorter and easier to scan")
print("✓ No vertical crowding - horizontal layout for headers")
