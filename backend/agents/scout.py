"""
Scout Agent — Phase 1
Identifies and acquires usable datasets from URLs or uploaded files.

Responsibilities:
  1. URL Evaluation     – trusted-source check + dataset-content detection
  2. Content Fetching   – HTML / file download, strips UI noise
  3. Data Structure Discovery – tables, rows, columns, dtype inference
  4. Schema Validation  – consistency, size heuristics, optional LLM confirmation
  5. File Handling      – CSV / XLSX / JSON uploads bypass web scraping
  6. Output             – structured JSON ready for downstream agents
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ScoutResult(BaseModel):
    is_dataset: bool
    source: str
    source_type: str                        # "url" | "csv" | "xlsx" | "json" | "text"
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    num_rows: int = 0
    num_columns: int = 0
    sample_data: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0           # 0.0 – 1.0
    rejection_reason: Optional[str] = None
    tables_found: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS: set[str] = {
    # Government / statistical
    "data.gov", "data.gov.ph", "census.gov", "bls.gov", "data.worldbank.org",
    "stats.oecd.org", "data.un.org", "eurostat.ec.europa.eu",
    # Research / academic
    "kaggle.com", "huggingface.co", "zenodo.org", "figshare.com",
    "datadryad.org", "osf.io", "ncbi.nlm.nih.gov", "ourworldindata.org",
    # Open data portals
    "opendata.ph", "data.cityofnewyork.us", "data.london.gov.uk",
    "data.medicare.gov", "healthdata.gov",
    # Raw / repo hosts
    "raw.githubusercontent.com", "github.com", "gitlab.com",
    # Common dataset hubs
    "archive.ics.uci.edu", "datasetsearch.research.google.com",
}

DATASET_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".tsv", ".parquet"}

MIN_ROWS = 3
MIN_COLS = 2
FETCH_TIMEOUT = 20  # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ScoutAgent/1.0; +https://github.com/scout-agent)"
    )
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    """Return the registered domain of a URL (strips www.)."""
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _is_trusted(url: str) -> bool:
    host = _domain(url)
    if any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS):
        return True
    # Not in allow-list — still proceed but lower confidence
    return False


def _ext(url: str) -> str:
    path = urlparse(url).path
    return Path(path).suffix.lower()


def _dtype_name(series: pd.Series) -> str:
    kind = series.dtype.kind
    mapping = {"i": "integer", "u": "integer", "f": "float",
                "M": "datetime", "b": "boolean"}
    return mapping.get(kind, "categorical")


def _summarise_df(df: pd.DataFrame, source: str, source_type: str,
                  confidence: float, tables_found: int = 1) -> ScoutResult:
    """Convert a DataFrame into a ScoutResult."""
    if df.empty or len(df) < MIN_ROWS or len(df.columns) < MIN_COLS:
        return ScoutResult(
            is_dataset=False,
            source=source,
            source_type=source_type,
            rejection_reason=(
                f"Table too small: {len(df)} row(s), {len(df.columns)} column(s). "
                f"Minimum required: {MIN_ROWS} rows × {MIN_COLS} columns."
            ),
            tables_found=tables_found,
        )

    # Infer better dtypes
    df = df.infer_objects()
    for col in df.select_dtypes(include="object"):
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        except Exception:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    columns = list(df.columns.astype(str))
    dtypes = {col: _dtype_name(df[col]) for col in df.columns}
    sample = df.head(5).astype(str).to_dict(orient="records")

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
        tables_found=tables_found,
    )

# ---------------------------------------------------------------------------
# LLM confirmation (Ollama, optional)
# ---------------------------------------------------------------------------

def _check_ollama(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _llm_confirm_dataset(snippet: str, ollama_host: str, model: str) -> tuple[bool, float, str]:
    """
    Ask a local LLM whether the snippet looks like dataset content.
    Returns (is_dataset, confidence 0-1, reason).
    Falls back to (True, 0.5, "llm_unavailable") gracefully.
    """
    if not _check_ollama(ollama_host):
        return True, 0.5, "llm_unavailable"

    prompt = (
        "You are a data quality classifier.\n"
        "Determine if the following content is a structured dataset "
        "(tables, repeated rows, CSV-like structures, JSON arrays of records).\n"
        "Reply ONLY with a JSON object containing:\n"
        '  "is_dataset": true or false\n'
        '  "confidence": float 0.0-1.0\n'
        '  "reason": one short sentence\n\n'
        f"Content snippet:\n{snippet[:1500]}\n"
    )

    try:
        import ollama  # type: ignore
        client = ollama.Client(host=ollama_host)
        resp = client.generate(model=model, prompt=prompt, stream=False,
                               options={"temperature": 0.0}, keep_alive=0)
        text = str(resp.get("response", "")).strip()
    except Exception:
        try:
            completed = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=25, check=True,
            )
            text = completed.stdout.decode(errors="replace").strip()
        except Exception:
            return True, 0.5, "llm_unavailable"

    # Parse JSON from response
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start:end + 1])
        return bool(obj.get("is_dataset", True)), float(obj.get("confidence", 0.5)), str(obj.get("reason", ""))
    except Exception:
        return True, 0.5, "llm_parse_error"

# ---------------------------------------------------------------------------
# File handlers
# ---------------------------------------------------------------------------

def _load_csv(path_or_bytes: "str | Path | bytes", source: str) -> ScoutResult:
    try:
        if isinstance(path_or_bytes, bytes):
            df = pd.read_csv(io.BytesIO(path_or_bytes))
        else:
            df = pd.read_csv(path_or_bytes)
        return _summarise_df(df, source, "csv", confidence=0.95)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=source, source_type="csv",
                           rejection_reason=f"CSV parse error: {e}")


def _load_excel(path_or_bytes: "str | Path | bytes", source: str) -> ScoutResult:
    try:
        if isinstance(path_or_bytes, bytes):
            df = pd.read_excel(io.BytesIO(path_or_bytes))
        else:
            df = pd.read_excel(path_or_bytes)
        return _summarise_df(df, source, "xlsx", confidence=0.95)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=source, source_type="xlsx",
                           rejection_reason=f"Excel parse error: {e}")


def _load_json(path_or_bytes: "str | Path | bytes", source: str) -> ScoutResult:
    try:
        if isinstance(path_or_bytes, bytes):
            raw = json.loads(path_or_bytes.decode())
        else:
            raw = json.loads(Path(path_or_bytes).read_text(encoding="utf-8"))

        # Accept list-of-dicts or dict-with-records key
        if isinstance(raw, list):
            df = pd.json_normalize(raw)
        elif isinstance(raw, dict):
            # Try to find a key that holds a list
            list_key = next((k for k, v in raw.items() if isinstance(v, list)), None)
            if list_key:
                df = pd.json_normalize(raw[list_key])
            else:
                df = pd.json_normalize([raw])
        else:
            return ScoutResult(is_dataset=False, source=source, source_type="json",
                               rejection_reason="JSON root is neither a list nor a dict.")
        return _summarise_df(df, source, "json", confidence=0.90)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=source, source_type="json",
                           rejection_reason=f"JSON parse error: {e}")

# ---------------------------------------------------------------------------
# URL handler
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> tuple[bytes, str]:
    """Fetch URL content; returns (raw_bytes, content_type)."""
    r = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT, stream=True)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()
    return r.content, content_type


def _score_html_table(df: pd.DataFrame) -> float:
    """Heuristic confidence for an HTML-extracted table."""
    score = 0.5
    if len(df) >= 10:
        score += 0.15
    if len(df) >= 50:
        score += 0.10
    if len(df.columns) >= 4:
        score += 0.10
    numeric_cols = df.select_dtypes(include="number").shape[1]
    if numeric_cols >= 2:
        score += 0.10
    # Penalise tables that look like navigation / layout
    col_names = " ".join(df.columns.astype(str)).lower()
    if any(w in col_names for w in ("button", "action", "link", "nav", "menu")):
        score -= 0.20
    return min(max(score, 0.0), 1.0)


def _html_to_best_df(html: str) -> tuple[pd.DataFrame | None, int]:
    """
    Parse HTML, strip noise, find all tables, return the best one.
    Returns (best_df | None, total_tables_found).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                               "aside", "form", "button", "iframe", "noscript",
                               "svg", "figure", "figcaption"]):
        tag.decompose()

    # Attempt 1: pandas read_html (fastest)
    try:
        tables = pd.read_html(io.StringIO(str(soup)), flavor="bs4")
    except Exception:
        tables = []

    if not tables:
        return None, 0

    # Filter out tiny tables
    candidates = [t for t in tables if len(t) >= MIN_ROWS and len(t.columns) >= MIN_COLS]
    if not candidates:
        return None, len(tables)

    # Pick table with the best heuristic score
    best = max(candidates, key=_score_html_table)
    return best, len(tables)


def _scout_url(url: str, use_llm: bool = False,
               ollama_host: str = "http://127.0.0.1:11434",
               ollama_model: str = "llama3.2:3b") -> ScoutResult:
    """Full pipeline for URL-sourced data."""

    trusted = _is_trusted(url)
    extension = _ext(url)

    # ---- Direct file download ----
    if extension in DATASET_EXTENSIONS:
        try:
            raw, _ = _fetch_url(url)
        except Exception as e:
            return ScoutResult(is_dataset=False, source=url, source_type="url",
                               rejection_reason=f"Download failed: {e}")

        if extension == ".csv":
            result = _load_csv(raw, url)
        elif extension in {".xlsx", ".xls"}:
            result = _load_excel(raw, url)
        elif extension == ".json":
            result = _load_json(raw, url)
        else:
            return ScoutResult(is_dataset=False, source=url, source_type="url",
                               rejection_reason=f"Unsupported extension: {extension}")

        if not trusted:
            result.confidence_score = round(result.confidence_score * 0.85, 2)
        return result

    # ---- HTML scraping ----
    try:
        raw, content_type = _fetch_url(url)
    except Exception as e:
        return ScoutResult(is_dataset=False, source=url, source_type="url",
                           rejection_reason=f"Fetch failed: {e}")

    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return ScoutResult(is_dataset=False, source=url, source_type="url",
                           rejection_reason=(
                               f"Non-HTML content type '{content_type}' "
                               "with no recognised dataset extension."
                           ))

    html = raw.decode(errors="replace")
    best_df, tables_found = _html_to_best_df(html)

    if best_df is None:
        # Optional LLM fallback
        if use_llm:
            ok, conf, reason = _llm_confirm_dataset(html[:3000], ollama_host, ollama_model)
            if not ok:
                return ScoutResult(is_dataset=False, source=url, source_type="url",
                                   rejection_reason=f"LLM rejected: {reason}",
                                   tables_found=tables_found)
        return ScoutResult(is_dataset=False, source=url, source_type="url",
                           rejection_reason="No usable HTML tables found.",
                           tables_found=tables_found)

    confidence = _score_html_table(best_df)
    if not trusted:
        confidence = round(confidence * 0.85, 2)

    # Optional LLM confirmation for borderline cases
    if use_llm and confidence < 0.70:
        snippet = best_df.head(10).to_csv(index=False)
        ok, llm_conf, reason = _llm_confirm_dataset(snippet, ollama_host, ollama_model)
        if not ok:
            return ScoutResult(is_dataset=False, source=url, source_type="url",
                               rejection_reason=f"LLM rejected: {reason}",
                               tables_found=tables_found)
        confidence = (confidence + llm_conf) / 2

    return _summarise_df(best_df, url, "url", confidence, tables_found)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scout(
    source: "str | Path | bytes",
    *,
    source_type: str = "auto",          # "auto" | "url" | "csv" | "xlsx" | "json" | "text"
    filename: str = "",                  # hint for bytes input
    use_llm: bool = False,
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2:3b",
) -> dict[str, Any]:
    """
    Main entry-point for the Scout Agent.

    Parameters
    ----------
    source      : URL string, local file path, or raw bytes (for uploads).
    source_type : Explicit type override; defaults to auto-detect.
    filename    : Original filename — used to detect type when source is bytes.
    use_llm     : Enable optional Ollama confirmation for edge cases.
    ollama_host : Ollama server base URL.
    ollama_model: Ollama model name.

    Returns
    -------
    dict matching the ScoutResult schema.
    """

    # ---- Bytes (Streamlit UploadedFile / raw upload) ----
    if isinstance(source, (bytes, bytearray)):
        ext = Path(filename).suffix.lower() if filename else ""
        src_label = filename or "uploaded_file"

        if ext == ".csv" or source_type == "csv":
            result = _load_csv(bytes(source), src_label)
        elif ext in {".xlsx", ".xls"} or source_type == "xlsx":
            result = _load_excel(bytes(source), src_label)
        elif ext == ".json" or source_type == "json":
            result = _load_json(bytes(source), src_label)
        else:
            # Try CSV → Excel → JSON in sequence
            for loader in (_load_csv, _load_excel, _load_json):
                result = loader(bytes(source), src_label)
                if result.is_dataset:
                    break
            else:
                result = ScoutResult(
                    is_dataset=False, source=src_label, source_type="unknown",
                    rejection_reason="Could not parse uploaded bytes as CSV, Excel, or JSON."
                )
        return result.model_dump()

    # ---- Streamlit UploadedFile duck-type ----
    if hasattr(source, "read") and hasattr(source, "name"):
        raw = source.read()
        return scout(raw, source_type=source_type, filename=source.name,
                     use_llm=use_llm, ollama_host=ollama_host, ollama_model=ollama_model)

    source = str(source).strip()

    # ---- Auto-detect type ----
    if source_type == "auto":
        if source.startswith("http://") or source.startswith("https://"):
            source_type = "url"
        elif source.lower().endswith(".csv"):
            source_type = "csv"
        elif source.lower().endswith((".xlsx", ".xls")):
            source_type = "xlsx"
        elif source.lower().endswith(".json"):
            source_type = "json"
        else:
            source_type = "text"

    # ---- Dispatch ----
    if source_type == "url":
        result = _scout_url(source, use_llm=use_llm,
                            ollama_host=ollama_host, ollama_model=ollama_model)

    elif source_type == "csv":
        result = _load_csv(source, source)

    elif source_type == "xlsx":
        result = _load_excel(source, source)

    elif source_type == "json":
        result = _load_json(source, source)

    else:  # raw text — attempt CSV parse
        try:
            df = pd.read_csv(io.StringIO(source))
            result = _summarise_df(df, "raw_text_input", "text", confidence=0.60)
        except Exception:
            result = ScoutResult(
                is_dataset=False, source="raw_text_input", source_type="text",
                rejection_reason="Raw text could not be parsed as tabular data."
            )

    return result.model_dump()


def run_scout(
    source: "str | Path | bytes",
    *,
    source_type: str = "auto",
    filename: str = "",
    use_llm: bool = False,
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2:3b",
) -> dict[str, Any]:
    """Canonical Scout runner used by orchestration."""
    return scout(
        source,
        source_type=source_type,
        filename=filename,
        use_llm=use_llm,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )

# ---------------------------------------------------------------------------
# Streamlit integration helper
# ---------------------------------------------------------------------------

def streamlit_scout_widget() -> None:
    """
    Drop-in Streamlit UI component for the Scout Agent.
    Call this inside a Streamlit app to get a complete upload / URL widget.
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("streamlit is required for streamlit_scout_widget().")

    st.subheader("🔍 Scout Agent — Dataset Acquisition")

    tab_url, tab_file = st.tabs(["🌐 URL", "📁 Upload File"])

    result: dict[str, Any] | None = None

    with tab_url:
        url_input = st.text_input("Dataset URL", placeholder="https://data.gov/dataset.csv")
        use_llm = st.checkbox("Use LLM confirmation (requires local Ollama)", value=False)
        if st.button("Scout URL", key="scout_url_btn"):
            if url_input.strip():
                with st.spinner("Scouting…"):
                    result = scout(url_input.strip(), use_llm=use_llm)
            else:
                st.warning("Please enter a URL.")

    with tab_file:
        uploaded = st.file_uploader(
            "Upload CSV, Excel, or JSON",
            type=["csv", "xlsx", "xls", "json"],
        )
        if uploaded is not None:
            with st.spinner("Loading file…"):
                result = scout(uploaded)

    if result:
        _render_result(result)


def _render_result(result: dict[str, Any]) -> None:
    try:
        import streamlit as st
    except ImportError:
        print(json.dumps(result, indent=2))
        return

    if result["is_dataset"]:
        st.success(
            f"✅ Valid dataset — {result['num_rows']:,} rows × "
            f"{result['num_columns']} columns "
            f"(confidence: {result['confidence_score']:.0%})"
        )
        st.json({
            "source": result["source"],
            "source_type": result["source_type"],
            "columns": result["columns"],
            "dtypes": result["dtypes"],
            "num_rows": result["num_rows"],
            "num_columns": result["num_columns"],
            "tables_found": result["tables_found"],
        })
        if result["sample_data"]:
            st.subheader("Sample Data (first 5 rows)")
            import pandas as pd
            st.dataframe(pd.DataFrame(result["sample_data"]))
    else:
        st.error(f"❌ Not a valid dataset — {result.get('rejection_reason', 'Unknown reason')}")
        st.json(result)

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    from dotenv import load_dotenv

    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Scout Agent: identify and acquire usable datasets."
    )
    parser.add_argument("source", help="URL, file path (.csv/.xlsx/.json), or raw text")
    parser.add_argument("--type", dest="source_type", default="auto",
                        choices=["auto", "url", "csv", "xlsx", "json", "text"])
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable optional Ollama LLM confirmation")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--out-dir", default=None,
                        help="If set, save result JSON to this directory")
    args = parser.parse_args()

    result = scout(
        args.source,
        source_type=args.source_type,
        use_llm=args.use_llm,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)

    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out_path / f"scout_{ts}.json").write_text(output, encoding="utf-8")

    return 0 if result["is_dataset"] else 1


if __name__ == "__main__":
    raise SystemExit(main())