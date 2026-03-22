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
# Core builder — validates DataFrame, saves raw_data.csv, returns ScoutResult
# ---------------------------------------------------------------------------

def _build_result(
    df: pd.DataFrame,
    source: str,
    source_type: str,
    confidence: float,
    run_dir: Path | None = None,
) -> ScoutResult:
    """
    Validate the DataFrame, infer dtypes, persist the FULL dataset to disk,
    and return a ScoutResult.

    raw_data.csv  → complete dataset, read by Labeler Agent
    sample_data   → first 5 rows only, stored in JSON for UI display
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
        return _build_result(df, label, "csv", confidence=0.95, run_dir=run_dir)
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
        return _build_result(df, label, "xlsx", confidence=0.95, run_dir=run_dir)
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

        return _build_result(df, label, "json", confidence=0.90, run_dir=run_dir)
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