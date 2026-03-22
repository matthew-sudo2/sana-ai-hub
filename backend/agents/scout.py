"""
Scout Agent — Phase 1
Identifies and acquires usable datasets from uploaded files only.

Responsibilities:
  1. File Handling      – CSV / XLSX / JSON uploads; no URL scraping
  2. Data Validation    – minimum row/column checks, dtype inference
  3. Schema Discovery   – column names, types, sample preview
  4. Full Dataset Save  – writes raw_data.csv (FULL dataset) to run_dir
                          so every downstream agent reads the complete data
  5. Output             – scout_result.json (metadata + preview)
                          raw_data.csv      (full dataset — the key fix)

URL inputs are rejected immediately with a clear message directing
the user to download the file and upload it directly.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ScoutResult(BaseModel):
    is_dataset: bool
    source: str
    source_type: str                        # "csv" | "xlsx" | "json" | "rejected" | "unknown"
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    num_rows: int = 0
    num_columns: int = 0
    sample_data: list[dict[str, Any]] = Field(default_factory=list)  # first 5 rows, display only
    confidence_score: float = 0.0
    rejection_reason: Optional[str] = None
    raw_data_path: Optional[str] = None     # absolute path to raw_data.csv written to run_dir
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ROWS = 3
MIN_COLS = 2
SAMPLE_PREVIEW_ROWS = 5   # rows stored in JSON for UI display only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_name(series: pd.Series) -> str:
    kind = series.dtype.kind
    mapping = {
        "i": "integer",
        "u": "integer",
        "f": "float",
        "M": "datetime",
        "b": "boolean",
    }
    return mapping.get(kind, "categorical")


def _infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to upgrade object columns to numeric or datetime.
    
    IMPORTANT: Only convert if MOST values are actually numeric/datetime.
    This prevents destroying string columns like Employee_ID, Email, etc.
    """
    df = df.infer_objects()
    for col in df.select_dtypes(include="object"):
        # Try datetime first
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            continue
        except Exception:
            pass
        
        # Try numeric ONLY if 70%+ of values can be converted
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_ratio = converted.notna().sum() / len(converted)
            
            # Only apply if majority of values converted successfully (70%+ threshold)
            # This preserves string columns that accidentally look numeric
            if non_null_ratio > 0.70:
                df[col] = converted
                print(f"[scout] Converted '{col}' to numeric ({non_null_ratio*100:.1f}% values converted)", flush=True)
            else:
                print(f"[scout] Keeping '{col}' as string ({non_null_ratio*100:.1f}% values convertible, below 70% threshold)", flush=True)
        except Exception:
            pass
    return df


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Quality Metrics Computation — Stricter Confidence Scoring
# ---------------------------------------------------------------------------

def _compute_completeness(df: pd.DataFrame) -> float:
    """
    Completeness: Proportion of non-null cells in the dataset.
    Range: [0, 1], where 1.0 = all cells filled, 0 = all null.
    Capped at 0.99 for conservatism.
    """
    if df.empty:
        return 0.0
    total_cells = df.shape[0] * df.shape[1]
    non_null = df.notna().sum().sum()
    score = non_null / total_cells if total_cells > 0 else 0.0
    return min(score, 0.99)


def _compute_consistency(df: pd.DataFrame) -> float:
    """
    Consistency: Proportion of rows that match expected dtype patterns.
    Checks if values in numeric/datetime columns can be parsed correctly.
    Range: [0, 1], capped at 0.99.
    """
    if df.empty:
        return 0.0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not numeric_cols and not datetime_cols:
        return 0.95  # All categorical is consistent but not fully validated
    
    total_checks = 0
    passed_checks = 0
    
    # Check numeric columns for parseable values
    for col in numeric_cols:
        total_checks += len(df)
        for val in df[col]:
            if pd.isna(val):
                passed_checks += 1  # NaN is acceptable
            elif isinstance(val, (int, float, np.number)):
                passed_checks += 1
            else:
                try:
                    float(val)
                    passed_checks += 1
                except (ValueError, TypeError):
                    pass
    
    # Check datetime columns
    for col in datetime_cols:
        total_checks += len(df)
        passed_checks += sum(1 for val in df[col] if pd.isna(val) or isinstance(val, pd.Timestamp))
    
    score = passed_checks / total_checks if total_checks > 0 else 1.0
    return min(score, 0.99)


def _compute_accuracy_score(df: pd.DataFrame) -> float:
    """
    Accuracy: Basic type validation — checks dtype inference credibility.
    Validates that inferred types match actual data patterns.
    Range: [0, 1], capped at 0.99.
    """
    if df.empty:
        return 0.0
    
    accuracy_checks = 0
    total_columns = len(df.columns)
    
    for col in df.columns:
        col_dtype = df[col].dtype
        non_null_vals = df[col].dropna()
        
        if non_null_vals.empty:
            accuracy_checks += 1  # Empty column is valid
            continue
        
        # Type validation rules
        if col_dtype.kind == 'i' or col_dtype.kind == 'u':  # integer
            try:
                pd.to_numeric(df[col], errors='coerce')
                accuracy_checks += 1
            except:
                pass
        elif col_dtype.kind == 'f':  # float
            try:
                pd.to_numeric(df[col], errors='coerce')
                accuracy_checks += 1
            except:
                pass
        elif col_dtype.kind == 'M':  # datetime
            try:
                pd.to_datetime(df[col], errors='coerce')
                accuracy_checks += 1
            except:
                pass
        else:  # categorical / object — just check it's not all null
            accuracy_checks += 1
    
    score = accuracy_checks / total_columns if total_columns > 0 else 1.0
    return min(score, 0.99)


def _compute_duplicate_score(df: pd.DataFrame) -> float:
    """
    Duplicate Score: (1 - duplicate_ratio)
    Measures uniqueness of rows. Score = 1.0 if no duplicates, decreases with ratio.
    Range: [0, 1], capped at 0.99.
    """
    if df.empty:
        return 0.0
    
    total_rows = len(df)
    duplicate_count = df.duplicated().sum()
    duplicate_ratio = duplicate_count / total_rows if total_rows > 0 else 0.0
    score = 1.0 - duplicate_ratio
    return min(score, 0.99)


def _compute_outlier_score(df: pd.DataFrame) -> float:
    """
    Outlier Score: (1 - outlier_pct) using IQR method on numeric columns.
    Detects outliers; score penalizes high outlier percentages.
    Range: [0, 1], capped at 0.99.
    """
    if df.empty:
        return 0.0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If no numeric columns, consider as no outlier risk
    if not numeric_cols:
        return 0.95
    
    total_data_points = 0
    outlier_count = 0
    
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        
        total_data_points += len(series)
        
        # IQR-based outlier detection
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR
            outliers_in_col = ((series < lower_fence) | (series > upper_fence)).sum()
            outlier_count += outliers_in_col
    
    if total_data_points == 0:
        return 0.95
    
    outlier_pct = outlier_count / total_data_points
    score = 1.0 - outlier_pct
    return min(score, 0.99)


def _compute_confidence_weighted(
    completeness: float,
    consistency: float,
    accuracy: float,
    duplicates: float,
    outliers: float,
) -> float:
    """
    Weighted confidence formula per user spec:
    Confidence = 0.25*C + 0.25*K + 0.20*A + 0.15*D + 0.15*O
    
    All inputs should be in [0, 1] range already (pre-capped).
    Final result capped at 0.99.
    """
    weights = {
        "completeness": 0.25,
        "consistency": 0.25,
        "accuracy": 0.20,
        "duplicates": 0.15,
        "outliers": 0.15,
    }
    
    confidence = (
        weights["completeness"] * completeness +
        weights["consistency"] * consistency +
        weights["accuracy"] * accuracy +
        weights["duplicates"] * duplicates +
        weights["outliers"] * outliers
    )
    
    # Cap at 0.99 — acknowledges data inherent uncertainty
    return min(confidence, 0.99)


# ---------------------------------------------------------------------------
# Core builder — validates DataFrame, saves raw_data.csv, returns ScoutResult
# ---------------------------------------------------------------------------

def _build_result(
    df: pd.DataFrame,
    source: str,
    source_type: str,
    run_dir: Path | None = None,
) -> ScoutResult:
    """
    Validate the DataFrame, infer dtypes, persist the FULL dataset to disk,
    and return a ScoutResult with computed weighted confidence score.

    raw_data.csv  → complete dataset, read by Labeler Agent
    sample_data   → first 5 rows only, stored in JSON for UI display
    
    Confidence computation:
    - Computes 5 quality metrics (Completeness, Consistency, Accuracy, Duplicates, Outliers)
    - Combines using weighted formula: 0.25*C + 0.25*K + 0.20*A + 0.15*D + 0.15*O
    - All metrics capped individually at 0.99; final confidence also capped at 0.99
    """
    # ── Size guard ───────────────────────────────────────────────────────
    if df.empty or len(df) < MIN_ROWS or len(df.columns) < MIN_COLS:
        return ScoutResult(
            is_dataset=False,
            source=source,
            source_type=source_type,
            rejection_reason=(
                f"Dataset too small: {len(df)} row(s), {len(df.columns)} column(s). "
                f"Minimum required: {MIN_ROWS} rows × {MIN_COLS} columns."
            ),
        )

    # ── Dtype inference ──────────────────────────────────────────────────
    df = _infer_dtypes(df)

    columns  = list(df.columns.astype(str))
    dtypes   = {str(col): _dtype_name(df[col]) for col in df.columns}

    # ── Compute quality metrics and confidence ────────────────────────────
    completeness = _compute_completeness(df)
    consistency = _compute_consistency(df)
    accuracy = _compute_accuracy_score(df)
    duplicates = _compute_duplicate_score(df)
    outliers = _compute_outlier_score(df)
    
    # Compute weighted confidence
    confidence = _compute_confidence_weighted(completeness, consistency, accuracy, duplicates, outliers)
    
    # Debug logging (optional, can be removed later)
    print(
        f"[scout] Confidence metrics for {source}:\n"
        f"  Completeness: {completeness:.2f}\n"
        f"  Consistency: {consistency:.2f}\n"
        f"  Accuracy: {accuracy:.2f}\n"
        f"  Duplicates: {duplicates:.2f}\n"
        f"  Outliers: {outliers:.2f}\n"
        f"  → Weighted Confidence: {confidence:.2f}",
        flush=True
    )

    # ── Persist FULL dataset ─────────────────────────────────────────────
    raw_data_path: str | None = None
    if run_dir is not None:
        _ensure_dir(run_dir)
        raw_csv = run_dir / "raw_data.csv"
        df.to_csv(raw_csv, index=False)
        raw_data_path = str(raw_csv.resolve())

    # ── Preview (UI display only) ────────────────────────────────────────
    sample = (
        df.head(SAMPLE_PREVIEW_ROWS)
        .astype(str)
        .to_dict(orient="records")
    )

    return ScoutResult(
        is_dataset=True,
        source=source,
        source_type=source_type,
        columns=columns,
        dtypes=dtypes,
        num_rows=len(df),
        num_columns=len(columns),
        sample_data=sample,
        confidence_score=round(confidence, 2),
        raw_data_path=raw_data_path,
    )


# ---------------------------------------------------------------------------
# File loaders — each reads the COMPLETE file into a DataFrame
# ---------------------------------------------------------------------------

def _load_csv(
    source: "str | Path | bytes",
    label: str,
    run_dir: Path | None = None,
) -> ScoutResult:
    try:
        df = pd.read_csv(io.BytesIO(bytes(source))) if isinstance(source, (bytes, bytearray)) else pd.read_csv(source)
        return _build_result(df, label, "csv", run_dir=run_dir)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=label, source_type="csv",
                           rejection_reason=f"CSV parse error: {e}")


def _load_excel(
    source: "str | Path | bytes",
    label: str,
    run_dir: Path | None = None,
) -> ScoutResult:
    try:
        df = pd.read_excel(io.BytesIO(bytes(source))) if isinstance(source, (bytes, bytearray)) else pd.read_excel(source)
        return _build_result(df, label, "xlsx", run_dir=run_dir)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=label, source_type="xlsx",
                           rejection_reason=f"Excel parse error: {e}")


def _load_json(
    source: "str | Path | bytes",
    label: str,
    run_dir: Path | None = None,
) -> ScoutResult:
    try:
        raw = json.loads(bytes(source).decode()) if isinstance(source, (bytes, bytearray)) else json.loads(Path(source).read_text(encoding="utf-8"))

        if isinstance(raw, list):
            df = pd.json_normalize(raw)
        elif isinstance(raw, dict):
            list_key = next((k for k, v in raw.items() if isinstance(v, list)), None)
            df = pd.json_normalize(raw[list_key]) if list_key else pd.json_normalize([raw])
        else:
            return ScoutResult(is_dataset=False, source=label, source_type="json",
                               rejection_reason="JSON root must be a list or dict.")

        return _build_result(df, label, "json", run_dir=run_dir)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=label, source_type="json",
                           rejection_reason=f"JSON parse error: {e}")


# ---------------------------------------------------------------------------
# Type resolution helper
# ---------------------------------------------------------------------------

def _resolve_type(source_type: str, ext: str) -> str:
    """Map source_type + file extension to a concrete loader key."""
    if source_type != "auto":
        return source_type
    if ext == ".csv":
        return "csv"
    if ext in {".xlsx", ".xls"}:
        return "xlsx"
    if ext == ".json":
        return "json"
    return "auto"   # unknown → try all loaders


def _dispatch(
    resolved_type: str,
    source: "str | Path | bytes",
    label: str,
    run_dir: Path | None,
) -> ScoutResult:
    """Route to the correct loader; fall back to auto-try on unknown type."""
    if resolved_type == "csv":
        return _load_csv(source, label, run_dir)
    if resolved_type == "xlsx":
        return _load_excel(source, label, run_dir)
    if resolved_type == "json":
        return _load_json(source, label, run_dir)

    # Auto-try: CSV → Excel → JSON
    for loader in (_load_csv, _load_excel, _load_json):
        result = loader(source, label, run_dir)
        if result.is_dataset:
            return result

    return ScoutResult(
        is_dataset=False, source=label, source_type="unknown",
        rejection_reason=(
            "Could not parse file as CSV, Excel, or JSON. "
            "Please ensure the file is a valid dataset."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scout(
    source: "str | Path | bytes",
    *,
    source_type: str = "auto",
    filename: str = "",
    run_dir: "str | Path | None" = None,
) -> dict[str, Any]:
    """
    Main Scout Agent entry point.

    Parameters
    ----------
    source      : File path (str / Path) or raw bytes from an upload.
                  URLs are rejected — users must upload the file directly.
    source_type : "auto" | "csv" | "xlsx" | "json"
    filename    : Original filename hint — used for type detection when
                  source is raw bytes (e.g. Streamlit UploadedFile.read()).
    run_dir     : Pipeline run directory. When provided, the FULL dataset
                  is saved as raw_data.csv here for downstream agents.
                  If None, no file is written (preview-only mode).

    Returns
    -------
    dict matching ScoutResult schema.
    raw_data_path is populated only when run_dir is given and file is valid.
    """
    run_dir_path = Path(run_dir) if run_dir else None

    # ── Reject URLs ──────────────────────────────────────────────────────
    if isinstance(source, (bytes, bytearray)):
        source_str = ""
    else:
        source_str = str(source).strip()

    if source_str.startswith("http://") or source_str.startswith("https://"):
        return ScoutResult(
            is_dataset=False,
            source=source_str,
            source_type="rejected",
            rejection_reason=(
                "URL inputs are not supported. "
                "Please download the dataset and upload it as a "
                "CSV, Excel (.xlsx), or JSON file."
            ),
        ).model_dump()

    # ── Streamlit UploadedFile duck-type ─────────────────────────────────
    if hasattr(source, "read") and hasattr(source, "name"):
        raw_bytes = source.read()
        fname = getattr(source, "name", filename)
        return scout(raw_bytes, source_type=source_type,
                     filename=fname, run_dir=run_dir)

    # ── Bytes (raw upload) ────────────────────────────────────────────────
    if isinstance(source, (bytes, bytearray)):
        ext   = Path(filename).suffix.lower() if filename else ""
        label = filename or "uploaded_file"
        rtype = _resolve_type(source_type, ext)
        return _dispatch(rtype, source, label, run_dir_path).model_dump()

    # ── File path ─────────────────────────────────────────────────────────
    path = Path(source_str)
    if not path.exists():
        return ScoutResult(
            is_dataset=False, source=source_str, source_type="unknown",
            rejection_reason=f"File not found: {source_str}",
        ).model_dump()

    ext   = path.suffix.lower()
    label = path.name
    rtype = _resolve_type(source_type, ext)
    return _dispatch(rtype, path, label, run_dir_path).model_dump()


def run_scout(
    source: "str | Path | bytes",
    *,
    source_type: str = "auto",
    filename: str = "",
    run_dir: "str | Path | None" = None,
) -> dict[str, Any]:
    """
    Canonical entry point used by graph.py orchestrator.
    Wraps scout() — exists so graph.py can do: from agents.scout import run_scout
    """
    return scout(source, source_type=source_type,
                 filename=filename, run_dir=run_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    from dotenv import load_dotenv

    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Scout Agent: validate and load an uploaded dataset file."
    )
    parser.add_argument("source", help="Path to a CSV, Excel, or JSON file.")
    parser.add_argument(
        "--type", dest="source_type", default="auto",
        choices=["auto", "csv", "xlsx", "json"],
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Pipeline run directory. raw_data.csv will be saved here.",
    )
    args = parser.parse_args()

    result = scout(
        args.source,
        source_type=args.source_type,
        run_dir=args.run_dir,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["is_dataset"] else 1


if __name__ == "__main__":
    raise SystemExit(main())