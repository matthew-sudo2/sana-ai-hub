"""
Labeler Agent — Phase 2
Receives scout_result.json (from scout_agent.py), cleans the FULL dataset,
and produces cleaned_data.csv + visual_report.png + phase2_summary.json.

Key fix: reads raw_data.csv (full dataset written by scout) instead of
         sample_data (5-row preview) or the original upload path.

Data loading priority:
  1. raw_data.csv in the run directory  ← always the full dataset
  2. raw_data_path field in scout JSON  ← absolute path fallback
  3. sample_data from scout JSON        ← last resort (preview only, warns)
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import numpy as np
import pandas as pd

# ← FIX: Set non-interactive matplotlib backend FIRST to avoid tkinter threading errors
# This must happen before any plotting occurs
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, thread-safe)

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Input schema — mirrors ScoutResult from scout_agent.py
# ---------------------------------------------------------------------------

class ScoutResult(BaseModel):
    is_dataset: bool
    source: str = Field(min_length=1)
    source_type: str                        # "csv" | "xlsx" | "json" | "rejected"
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    num_rows: int = 0
    num_columns: int = 0
    sample_data: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    rejection_reason: str | None = None
    raw_data_path: str | None = None        # absolute path to raw_data.csv on disk
    timestamp: str = ""

    # back-compat fields from old Phase1Metadata shape
    source_url: str | None = None
    primary_topic: str | None = None
    publication_date: str | None = None
    raw_quantitative_stats: Any = None

    def effective_source(self) -> str:
        return self.source or self.source_url or "unknown"


# ---------------------------------------------------------------------------
# Deterministic Cleaning Profile (replaces LLM approach)
# ---------------------------------------------------------------------------

class CleaningProfile(BaseModel):
    """Pre-configured cleaning rules — no LLM dependency needed"""
    drop_empty_rows: bool = True
    drop_empty_cols: bool = True
    drop_duplicates: bool = True
    standardize_column_names: bool = True
    auto_detect_dtypes: bool = True
    fill_missing_strategy: Literal["drop", "ffill", "mean", "median"] = "median"
    handle_outliers: bool = False
    outlier_method: Literal["iqr", "zscore"] = "iqr"


DEFAULT_CLEANING_PROFILE = CleaningProfile()


def _apply_deterministic_cleaning(
    df: pd.DataFrame, profile: CleaningProfile = DEFAULT_CLEANING_PROFILE
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fast, reproducible data cleaning without LLM. Runs in <1 second.
    Returns (cleaned_df, metrics_dict) to track what was removed/cleaned.
    """
    print(f"[labeler] _apply_deterministic_cleaning START: {len(df)} rows x {len(df.columns)} columns", flush=True)
    print(f"[labeler]   Column names: {list(df.columns)}", flush=True)
    
    metrics = {
        "rows_input": len(df),
        "columns_input": len(df.columns),
        "duplicates_removed": 0,
        "empty_rows_removed": 0,
        "empty_columns_removed": 0,
        "columns_standardized": 0,
        "missing_values_filled": 0,
        "outliers_removed": 0,
    }

    # Standardize column names to snake_case
    if profile.standardize_column_names:
        old_cols = list(df.columns)
        df.columns = [
            re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')
            for c in df.columns
        ]
        # Count how many changed
        metrics["columns_standardized"] = sum(1 for o, n in zip(old_cols, df.columns) if o != n)

    # Drop fully empty rows
    if profile.drop_empty_rows:
        rows_before = len(df)
        df = df.dropna(how='all')
        removed = rows_before - len(df)
        metrics["empty_rows_removed"] = removed
        print(f"[labeler]   After drop_empty_rows: {len(df)} rows (removed {removed}), {len(df.columns)} columns", flush=True)

    # Drop fully empty columns
    if profile.drop_empty_cols:
        cols_before = len(df.columns)
        cols_before_names = list(df.columns)
        df = df.dropna(axis=1, how='all')
        cols_removed = cols_before - len(df.columns)
        cols_after_names = list(df.columns)
        metrics["empty_columns_removed"] = cols_removed
        print(f"[labeler]   After drop_empty_cols: {len(df)} rows, {len(df.columns)} columns (removed {cols_removed})", flush=True)
        if cols_removed > 0:
            removed_cols = set(cols_before_names) - set(cols_after_names)
            print(f"[labeler]     Removed columns: {removed_cols}", flush=True)


    # Remove duplicate rows
    if profile.drop_duplicates:
        rows_before = len(df)
        df = df.drop_duplicates()
        metrics["duplicates_removed"] = rows_before - len(df)

    # Auto-detect and convert numeric-looking columns (handles "age" stored as string)
    numeric_name_hints = {'age', 'salary', 'price', 'cost', 'amount', 'revenue', 'score', 'rating', 'years', 'experience', 'count', 'total', 'quantity', 'id', 'year', 'month', 'day'}
    for col in df.columns:
        if df[col].dtype == 'object':  # String/mixed type column
            # Try to convert to numeric, coerce blanks to NaN
            converted = pd.to_numeric(df[col], errors='coerce')
            col_name_lower = col.lower()
            conversion_ratio = converted.notna().sum() / len(df)
            
            # Convert if: (1) 20%+ values converted OR (2) column name hints it's numeric
            is_numeric_by_name = any(hint in col_name_lower for hint in numeric_name_hints)
            if conversion_ratio > 0.2 or (is_numeric_by_name and converted.notna().sum() > 0):
                print(f"[labeler] Converting '{col}' to numeric ({conversion_ratio*100:.1f}% non-null, numeric_name={is_numeric_by_name})", flush=True)
                df[col] = converted

    # Handle missing values
    if profile.fill_missing_strategy == "ffill":
        missing_before = df.isna().sum().sum()
        df = df.fillna(method="ffill")
        missing_after = df.isna().sum().sum()
        metrics["missing_values_filled"] = missing_before - missing_after
    elif profile.fill_missing_strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_before = df.isna().sum().sum()
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        missing_after = df.isna().sum().sum()
        metrics["missing_values_filled"] = missing_before - missing_after
    elif profile.fill_missing_strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"[labeler] MEDIAN_FILL_DEBUG: Numeric columns found: {list(numeric_cols)}", flush=True)
        
        missing_before = df.isna().sum().sum()
        print(f"[labeler] MEDIAN_FILL_DEBUG: Total missing values before: {missing_before}", flush=True)
        
        for col in numeric_cols:
            col_missing = df[col].isna().sum()
            col_median = df[col].median()
            col_non_null = df[col].notna().sum()
            print(f"[labeler]   Column '{col}': dtype={df[col].dtype}, non-null={col_non_null}, missing={col_missing}, median={col_median}", flush=True)
            
            if col_missing > 0:
                print(f"[labeler]     BEFORE fill: {df[col].isna().sum()} missing in df[{col}]", flush=True)
                df[col] = df[col].fillna(col_median)
                after_fill = df[col].isna().sum()
                print(f"[labeler]     AFTER fill: remaining missing={after_fill} in df[{col}]", flush=True)
                print(f"[labeler]     Verify: current df[{col}] has {df[col].notna().sum()} non-null values", flush=True)
        
        missing_after = df.isna().sum().sum()
        metrics["missing_values_filled"] = missing_before - missing_after
        print(f"[labeler] MEDIAN_FILL_DEBUG: Total missing values after: {missing_after}, filled: {metrics['missing_values_filled']}", flush=True)

    # Optional outlier removal
    if profile.handle_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if profile.outlier_method == "iqr":
            rows_before = len(df)
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            metrics["outliers_removed"] = rows_before - len(df)

    # Final metrics
    metrics["rows_output"] = len(df)
    metrics["columns_output"] = len(df.columns)
    
    print(f"[labeler] _apply_deterministic_cleaning END: {len(df)} rows x {len(df.columns)} columns", flush=True)
    print(f"[labeler]   Final columns: {list(df.columns)}", flush=True)
    
    # DEBUG: Check final state before returning
    if 'age' in df.columns:
        age_missing_final = df['age'].isna().sum()
        age_non_null_final = df['age'].notna().sum()
        print(f"[labeler] _apply_deterministic_cleaning FINAL CHECK: age has {age_non_null_final} non-null, {age_missing_final} missing", flush=True)
        print(f"[labeler] DEBUG: age sample before return: {df['age'].head(5).tolist()}", flush=True)
        print(f"[labeler] DEBUG: age dtype at return: {df['age'].dtype}", flush=True)
        # VERIFY: No NaN should exist after median fill
        if age_missing_final > 0:
            print(f"[labeler] ⚠️ WARNING: {age_missing_final} NaN values still exist in age after median fill!", flush=True)
            print(f"[labeler] Age NaN indices: {df[df['age'].isna()].index.tolist()[:10]}", flush=True)
    else:
        print(f"[labeler] WARNING: 'age' column not found! Columns are: {list(df.columns)}", flush=True)

    return df, metrics


def _make_simple_visual_report(
    df: pd.DataFrame, out_png_path: str, profile: CleaningProfile = DEFAULT_CLEANING_PROFILE
) -> None:
    """Fast builtin visualization without LLM. Runs in <2 seconds."""
    sns.set_theme(style="whitegrid")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    n = min(len(numeric_cols), 4)

    if n == 0:
        # No numeric columns—show a placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No numeric columns to plot', ha='center', va='center', fontsize=14)
        ax.axis('off')
    elif n == 1:
        # Single numeric column—histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[numeric_cols[0]].dropna(), bins=20, ax=ax, kde=True)
        ax.set_title(f'Distribution of {numeric_cols[0]}')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel('Frequency')
    else:
        # Multiple numeric columns—grid of histograms
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        for i, col in enumerate(numeric_cols[:n]):
            sns.histplot(df[col].dropna(), bins=20, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Caching Layer (optional speedup for large datasets)
# ---------------------------------------------------------------------------

def _compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute MD5 hash of DataFrame for change detection."""
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return hashlib.md5(csv_bytes).hexdigest()


def _get_cache_marker_path(run_dir: Path) -> Path:
    """Location where we store the data fingerprint."""
    return run_dir / ".labeler_cache"


class CacheMarker(BaseModel):
    """Tracks if cleaning has already been applied to this dataset."""
    data_hash: str
    profile_hash: str
    timestamp: str


def _should_skip_cleaning(
    df: pd.DataFrame, profile: CleaningProfile, run_dir: Path
) -> bool:
    """Check if this exact dataset with this profile was already cleaned."""
    cache_file = _get_cache_marker_path(run_dir)
    if not cache_file.exists():
        return False
    
    try:
        cached = CacheMarker.model_validate_json(cache_file.read_text())
        current_data_hash = _compute_dataframe_hash(df)
        current_profile_hash = hashlib.md5(
            json.dumps(profile.model_dump(), sort_keys=True).encode()
        ).hexdigest()
        
        return (cached.data_hash == current_data_hash and 
                cached.profile_hash == current_profile_hash)
    except Exception:
        return False


def _save_cache_marker(
    df: pd.DataFrame, profile: CleaningProfile, run_dir: Path
) -> None:
    """Save the data+profile fingerprints to mark this dataset as cached."""
    data_hash = _compute_dataframe_hash(df)
    profile_hash = hashlib.md5(
        json.dumps(profile.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    
    marker = CacheMarker(
        data_hash=data_hash,
        profile_hash=profile_hash,
        timestamp=dt.datetime.now(timezone.utc).isoformat(),
    )
    _get_cache_marker_path(run_dir).write_text(marker.model_dump_json())


# ---------------------------------------------------------------------------
# Full dataset loader
# ---------------------------------------------------------------------------

def _apply_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Apply scout-inferred dtype hints to DataFrame columns.
    
    IMPORTANT: Only apply conversions if they would actually preserve data.
    Skip conversions that would result in all-NaN columns (data destruction).
    """
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        
        if dtype in ("integer", "float", "numeric"):
            # Test conversion on a sample - only apply if we won't lose all data
            test_converted = pd.to_numeric(df[col], errors="coerce")
            non_null_ratio = test_converted.notna().sum() / len(test_converted) if len(test_converted) > 0 else 0
            
            # Only convert if we keep at least some data (avoid NaN-destroying conversions)
            if non_null_ratio > 0.1 or (df[col].notna().sum() == 0):
                # Either: conversion keeps 10%+ of values, OR column is already all-NaN
                df[col] = test_converted
            else:
                # Skip conversion - would destroy valuable data
                print(f"[labeler] Skipped numeric conversion for '{col}' ({non_null_ratio*100:.1f}% would convert - data preserved as string)", flush=True)
        
        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        elif dtype == "boolean":
            df[col] = (
                df[col].astype(str).str.lower()
                .map({"true": True, "false": False})
                .astype("boolean")
            )
    return df


def _load_full_dataframe(scout: ScoutResult, run_dir: Path) -> pd.DataFrame:
    """
    Load the COMPLETE dataset in priority order:

    1. raw_data.csv in run_dir           — written by scout, always preferred
    2. raw_data_path from scout JSON     — absolute path fallback
    3. sample_data from scout JSON       — last resort, logs a warning

    Never reads from scout.source (original upload path) because that file
    may be a temp path that no longer exists after the upload completes.
    """

    # ── Priority 1: raw_data.csv in the shared run directory ─────────────
    raw_csv = run_dir / "raw_data.csv"
    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        df = pd.read_csv(raw_csv)
        if not df.empty:
            print(f"[labeler] Reading full dataset from raw_data.csv ({len(df):,} rows)", flush=True)
            return _apply_dtypes(df, scout.dtypes)

    # ── Priority 2: raw_data_path stored in scout JSON ────────────────────
    if scout.raw_data_path:
        p = Path(scout.raw_data_path)
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p)
            if not df.empty:
                print(f"[labeler] Reading full dataset from raw_data_path ({len(df):,} rows)", flush=True)
                # Copy into run_dir so downstream agents can find it
                df.to_csv(raw_csv, index=False)
                return _apply_dtypes(df, scout.dtypes)

    # ── Priority 3: sample_data — last resort ────────────────────────────
    if scout.sample_data:
        print(
            f"[labeler] WARNING: raw_data.csv not found. "
            f"Falling back to sample_data ({len(scout.sample_data)} rows only). "
            "The full dataset was not persisted by the Scout Agent. "
            "Ensure run_dir is passed to scout() so raw_data.csv is written.",
            flush=True,
        )
        df = pd.DataFrame(scout.sample_data)
        return _apply_dtypes(df, scout.dtypes)

    raise RuntimeError(
        "No dataset available. Expected raw_data.csv in the run directory. "
        "Make sure the Scout Agent receives run_dir so it can persist the full dataset."
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/Path types to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__module__') and 'numpy' in obj.__module__:
        return str(obj)
    else:
        return obj


def _write_json(path: Path, obj: Any) -> None:
    safe_obj = _make_json_safe(obj)
    path.write_text(json.dumps(safe_obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core runner — importable by graph.py / UI
# ---------------------------------------------------------------------------

def run_labeler(
    scout_json_path: Path,
    model: str = None,
    ollama_host: str = None,
    out_dir: str = "runs",
    cleaning_profile: CleaningProfile = DEFAULT_CLEANING_PROFILE,
) -> dict[str, Any]:
    """
    Main labeler pipeline. Reads the full dataset from raw_data.csv,
    cleans it deterministically, generates a visual report, writes outputs.

    Returns phase2_summary dict.

    Note: model and ollama_host parameters are kept for API compatibility
    but are no longer used (replaced with deterministic cleaning).
    """
    raw_obj = _read_json(scout_json_path)

    # ── Parse scout output ────────────────────────────────────────────────
    try:
        scout = ScoutResult.model_validate(raw_obj)
    except ValidationError:
        # Back-compat: wrap old Phase1Metadata format
        scout = ScoutResult(
            is_dataset=True,
            source=raw_obj.get("source_url", "unknown"),
            source_type="csv",
            raw_quantitative_stats=raw_obj.get("raw_quantitative_stats"),
            primary_topic=raw_obj.get("primary_topic"),
            publication_date=raw_obj.get("publication_date"),
        )

    if not scout.is_dataset:
        raise ValueError(
            f"Scout rejected the input as not a dataset: {scout.rejection_reason}"
        )

    run_dir = scout_json_path.parent
    _ensure_dir(run_dir)

    # ── Load FULL dataset ─────────────────────────────────────────────────
    df = _load_full_dataframe(scout, run_dir)

    if df.empty:
        raise RuntimeError("Loaded DataFrame is empty — nothing to clean.")
    
    # DEBUG: Show dtypes after loading
    print(f"[labeler] DEBUG: After loading, columns and dtypes:", flush=True)
    for col in df.columns:
        print(f"[labeler]   {col}: {df[col].dtype}, non-null count={df[col].notna().sum()}, null count={df[col].isna().sum()}", flush=True)

    # ── Write preview and input copy ──────────────────────────────────────
    df_preview_csv = df.head(8).to_csv(index=False)
    _write_text(run_dir / "df_preview.csv", df_preview_csv)
    _write_json(run_dir / "scout_input.json", raw_obj)

    # ── Check cache: skip if identical dataset + profile ──────────────────
    print(f"[labeler] Checking cache for duplicate cleaning...", flush=True)
    # TEMPORARY DEBUG: Disable cache to force cleaning to run
    if False:  # Cache temporarily disabled
        print(f"[labeler] Dataset+profile already cleaned. Skipping.", flush=True)
        # Load cached result
        cleaned_csv_path = run_dir / "cleaned_data.csv"
        if cleaned_csv_path.exists():
            cleaned = pd.read_csv(cleaned_csv_path)
            # Compute metrics for the cached result
            empty_rows = len(df) - len(df.dropna(how='all'))
            empty_cols = len(df.columns) - len(df.dropna(axis=1, how='all'))
            duplicates = len(df) - len(df.drop_duplicates())
            summary = {
                "source": scout.effective_source(),
                "source_type": scout.source_type,
                "confidence_score": scout.confidence_score,
                "cleaned_data_csv": str(cleaned_csv_path),
                "visual_report_png": str(run_dir / "visual_report.png"),
                "rows": int(cleaned.shape[0]),
                "columns": list(map(str, cleaned.columns)),
                "dtypes": {col: str(cleaned[col].dtype) for col in cleaned.columns},
                "cleaning_metrics": {
                    "rows_input": len(df),
                    "columns_input": len(df.columns),
                    "rows_output": len(cleaned),
                    "columns_output": len(cleaned.columns),
                    "empty_rows_removed": empty_rows,
                    "empty_columns_removed": empty_cols,
                    "duplicates_removed": duplicates,
                    "columns_standardized": 0,
                    "missing_values_filled": 0,
                    "outliers_removed": 0,
                },
            }
            return summary

    # ── Clean the FULL dataset (deterministic, no LLM) ────────────────────
    print(f"[labeler] Cleaning: {len(df):,} rows x {len(df.columns)} columns", flush=True)
    
    # DEBUG: State of df BEFORE cleaning
    if 'age' in df.columns:
        print(f"[labeler] DEBUG START: df['age'] has {df['age'].notna().sum()} non-null, {df['age'].isna().sum()} missing BEFORE cleaning", flush=True)
    
    cleaned, cleaning_metrics = _apply_deterministic_cleaning(df, cleaning_profile)
    
    if not isinstance(cleaned, pd.DataFrame):
        raise RuntimeError("clean_dataframe() did not return a pandas DataFrame.")
    
    # DEBUG: State of cleaned AFTER function returns
    if 'age' in cleaned.columns:
        print(f"[labeler] DEBUG END: cleaned['age'] has {cleaned['age'].notna().sum()} non-null, {cleaned['age'].isna().sum()} missing AFTER _apply_deterministic_cleaning", flush=True)

    print(f"[labeler] Cleaned: {len(cleaned):,} rows × {len(cleaned.columns)} columns", flush=True)
    
    # Print metrics (convert numpy types to Python types for JSON serialization)
    try:
        # Convert numpy types to native Python types
        metrics_json_safe = {}
        for k, v in cleaning_metrics.items():
            if hasattr(v, 'item'):  # numpy types have .item()
                metrics_json_safe[k] = v.item()
            else:
                metrics_json_safe[k] = v
        print(f"[labeler] Metrics: {json.dumps(metrics_json_safe, indent=2)}", flush=True)
    except Exception as e:
        print(f"[labeler] Warning: Could not serialize metrics for logging: {e}", flush=True)
        print(f"[labeler] Metrics (raw): {cleaning_metrics}", flush=True)

    
    # DEBUG: Check age values right before saving
    if 'age' in cleaned.columns:
        print(f"[labeler] DEBUG BEFORE CSV SAVE: age has {cleaned['age'].notna().sum()} non-null, {cleaned['age'].isna().sum()} missing", flush=True)
        print(f"[labeler] DEBUG: age sample before save: {cleaned['age'].head(5).tolist()}", flush=True)
    
    # CRITICAL FIX: Ensure we're using the SAME dataframe object throughout
    # Create explicit copy to prevent any reference issues
    final_cleaned = cleaned.copy(deep=True)
    
    print(f"[labeler] About to save cleaned_data.csv with columns: {list(final_cleaned.columns)}", flush=True)
    print(f"[labeler] Column dtypes before save: {final_cleaned.dtypes.to_dict()}", flush=True)
    if 'age' in final_cleaned.columns:
        print(f"[labeler] FINAL VERIFY: age in final_cleaned: {final_cleaned['age'].notna().sum()} non-null", flush=True)
    
    cleaned_csv_path = run_dir / "cleaned_data.csv"
    log_file = run_dir / "labeler_save_debug.txt"
    
    # Write comprehensive debug info FIRST
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== CSV SAVE DEBUG LOG ===\n")
            f.write(f"Timestamp: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
            f.write(f"DataFrame shape: {final_cleaned.shape}\n")
            f.write(f"Columns: {list(final_cleaned.columns)}\n")
            f.write(f"Dtypes:\n{final_cleaned.dtypes}\n")
            if 'age' in final_cleaned.columns:
                f.write(f"\nAge column analysis BEFORE CSV SAVE:\n")
                f.write(f"  Non-null count: {final_cleaned['age'].notna().sum()}\n")
                f.write(f"  Null count: {final_cleaned['age'].isna().sum()}\n")
                f.write(f"  Dtype: {final_cleaned['age'].dtype}\n")
                f.write(f"  Min: {final_cleaned['age'].min()}\n")
                f.write(f"  Max: {final_cleaned['age'].max()}\n")
                f.write(f"  First 10 values: {final_cleaned['age'].head(10).tolist()}\n")
                f.write(f"  Last 10 values: {final_cleaned['age'].tail(10).tolist()}\n")
            f.write(f"\n--- ATTEMPTING CSV WRITE ---\n")
    except Exception as e:
        print(f"[labeler] [X] ERROR writing debug log: {type(e).__name__}: {str(e)}", flush=True)
        raise
    
    # SIMPLE, ROBUST CSV WRITE with explicit error handling
    csv_write_success = False
    try:
        # Use text mode write instead of pandas to_csv for more control
        csv_content = final_cleaned.to_csv(index=False)
        
        # Write to file with explicit flush
        with open(cleaned_csv_path, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
            f.flush()  # Explicit flush
        
        csv_write_success = True
        print(f"[labeler] [OK] CSV WRITTEN to {cleaned_csv_path} ({len(csv_content)} bytes)", flush=True)
        
    except Exception as e:
        print(f"[labeler] [X] ERROR IN CSV WRITE: {type(e).__name__}: {str(e)}", flush=True)
        with open(log_file, 'a') as f:
            f.write(f"\nCSV WRITE ERROR: {type(e).__name__}\n{str(e)}\n")
        raise
    
    # VERIFY: Read back and compare
    if csv_write_success:
        try:
            verify_df = pd.read_csv(cleaned_csv_path)
            print(f"[labeler] [OK] CSV VERIFIED: {len(verify_df)} rows x {len(verify_df.columns)} columns", flush=True)
            
            with open(log_file, 'a') as f:
                f.write(f"\nCSV WRITE: SUCCESS ({len(csv_content)} bytes written)\n")
                f.write(f"\nCSV VERIFY RELOAD:\n")
                f.write(f"  Shape: {verify_df.shape}\n")
                f.write(f"  Columns: {list(verify_df.columns)}\n")
                if 'age' in verify_df.columns:
                    f.write(f"\nAge column analysis AFTER CSV SAVE:\n")
                    f.write(f"  Non-null count: {verify_df['age'].notna().sum()}\n")
                    f.write(f"  Null count: {verify_df['age'].isna().sum()}\n")
                    f.write(f"  Dtype after reload: {verify_df['age'].dtype}\n")
                    f.write(f"  Min: {verify_df['age'].min()}\n")
                    f.write(f"  Max: {verify_df['age'].max()}\n")
                    f.write(f"  First 10 values: {verify_df['age'].head(10).tolist()}\n")
                    f.write(f"  Last 10 values: {verify_df['age'].tail(10).tolist()}\n")
            
            if 'age' in verify_df.columns:
                print(f"[labeler] DEBUG AFTER CSV RELOAD: age has {verify_df['age'].notna().sum()} non-null, {verify_df['age'].isna().sum()} missing", flush=True)
                print(f"[labeler] DEBUG: age sample after reload: {verify_df['age'].head(5).tolist()}", flush=True)
        except Exception as e:
            print(f"[labeler] [X] ERROR VERIFYING CSV: {type(e).__name__}: {str(e)}", flush=True)
            with open(log_file, 'a') as f:
                f.write(f"\nCSV VERIFY ERROR: {type(e).__name__}: {str(e)}\n")


    
    # ── Mark cache as valid ──────────────────────────────────────────────
    _save_cache_marker(df, cleaning_profile, run_dir)

    # ── Visual report (fast builtin visualization) ────────────────────────
    print(f"[labeler] Generating visual report...", flush=True)
    report_png_path = run_dir / "visual_report.png"
    _make_simple_visual_report(cleaned, str(report_png_path), cleaning_profile)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "source": scout.effective_source(),
        "source_type": scout.source_type,
        "confidence_score": scout.confidence_score,
        "cleaned_data_csv": str(cleaned_csv_path),
        "visual_report_png": str(report_png_path),
        "rows": int(cleaned.shape[0]),
        "columns": list(map(str, cleaned.columns)),
        "dtypes": {col: str(cleaned[col].dtype) for col in cleaned.columns},
        "cleaning_metrics": cleaning_metrics,  # ← NEW: Track what was cleaned
    }
    _write_json(run_dir / "phase2_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Labeler Agent (Phase 2): clean the full dataset from scout output."
    )
    parser.add_argument(
        "scout_json",
        help="Path to scout_result.json written by the Scout Agent.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE2_MODEL", None),
        help="(Deprecated - kept for backward compatibility)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", None),
        help="(Deprecated - kept for backward compatibility)",
    )
    parser.add_argument(
        "--out-dir", 
        default="runs",
        help="Output directory (default: runs)"
    )
    args = parser.parse_args()

    path = Path(args.scout_json)
    if not path.exists():
        raise FileNotFoundError(str(path))

    summary = run_labeler(
        scout_json_path=path,
        model=args.model,
        ollama_host=args.ollama_host,
        out_dir=args.out_dir,
        cleaning_profile=DEFAULT_CLEANING_PROFILE,
    )
    
    # Use module-level _make_json_safe to ensure numpy types are properly handled
    safe_summary = _make_json_safe(summary)
    print(json.dumps(safe_summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValidationError, ValueError) as e:
        raise SystemExit(f"[labeler] Input error: {e}")