SCOUT.PY -

raw_data_path added to ScoutResult — the schema now carries the absolute path to raw_data.csv so downstream agents and graph.py always know exactly where the full dataset lives without guessing.
_build_result() now saves raw_data.csv — this is the core fix. When run_dir is provided, the complete DataFrame is written to disk with df.to_csv(raw_csv, index=False) before anything else. sample_data in the JSON stays at 5 rows for UI display only and is never read by any downstream agent.
run_dir parameter added to scout() and run_scout() — graph.py passes the run directory in so the scout knows where to save. Without this parameter the old code had no way to persist anything.
URLs are rejected immediately — one clean error message tells the user exactly what to do: download the file and upload it. No BeautifulSoup, no requests, no scraping code at all.
run_scout() now exists as a real importable function — graph.py's from .agents.scout import run_scout now works correctly and passes run_dir through.
_dispatch() centralizes loader routing — one place handles the auto-try fallback (CSV → Excel → JSON) for both bytes and file path inputs, eliminating the duplicated fallback logic from before.
No new dependencies — removed beautifulsoup4, lxml, and requests since URL scraping is gone. Only pandas, pydantic, and standard library remain.

LABELER.PY -

to_dataframe() is completely replaced by _load_full_dataframe() — the old method was a method on ScoutResult that tried to re-read the original upload path (which may no longer exist) and fell back to sample_data. The new standalone function has a clear 3-priority chain with explicit logging at each step so you can see exactly which path was taken.
Priority 1 — raw_data.csv in run_dir — this is what the fixed scout now always writes. The labeler looks for it first, logs how many rows it found, and uses it. This is the fix for the 5-row problem.
Priority 2 — raw_data_path from scout JSON — if for some reason raw_data.csv isn't in run_dir (e.g. the scout wrote it to a different location), the absolute path stored in the scout JSON is used and the file is copied into run_dir for consistency with downstream agents.
Priority 3 — sample_data fallback with a loud warning — rather than silently using 5 rows, it now prints a clear warning message explaining exactly what went wrong and what needs to be fixed. This makes debugging obvious instead of invisible.
num_rows in the prompt is now len(df) (actual) — the old code passed scout.num_rows which is correct for the full dataset, but now the labeler verifies it by passing the actual row count of the loaded DataFrame, so the LLM gets truthful context.
raw_data_path added to ScoutResult schema — the labeler's local copy of the schema now includes this new field so it can read raw_data_path from the scout JSON without validation errors.

ANALYST.PY -

Only two targeted fixes, nothing else touched:
_drop_internal_cols() called at the top of run_analysis() — strips _is_outlier (and any future pipeline-internal columns in _INTERNAL_COLS) before any stats, correlations, or anomaly detection runs. Without this, on a re-run where cleaned_data.csv somehow already contains _is_outlier, it would be treated as a numeric boolean column and corrupt the correlation matrix, outlier counts, and insights.
_iqr_outliers() now guards on len(s) < 4 — if a numeric column has fewer than 4 non-NaN values (fully-NaN column that survived cleaning, or a near-empty column), quantile() would raise or return meaningless results. Now it returns ([], 0.0, 0.0) cleanly and the column shows outlier_count=0 in the stats.
Also added print(f"[analyst] Loaded {len(df):,} rows ...") at the top so you can see in the logs that the full dataset (not 5 rows) is flowing through.