"""
LangGraph orchestrator for Sana pipeline.
Flow: scout -> labeler -> analyst -> artist -> validator.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
import sys
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict, total=False):
    """Pipeline state for LangGraph - compatible TypedDict."""
    run_id: str
    source: str
    phase: str
    status: str
    error: str | None
    source_type: str
    run_dir: str | None
    scout_output_dir: str | None
    labeler_output_dir: str | None
    analyst_output_dir: str | None
    artist_output_dir: str | None
    validator_output_dir: str | None


def _slug(text: str, max_len: int = 80) -> str:
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch in {"_", "-"}) else "_")
    slug = "".join(keep).strip("_")
    return (slug[:max_len] or "run").strip("_") or "run"


def _state_error(state: PipelineState, phase: str, message: str) -> PipelineState:
    state["phase"] = phase
    state["status"] = "error"
    state["error"] = message
    return state


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def _json_from_output(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Unable to parse JSON object from subprocess output")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(state: PipelineState) -> Path:
    if state["run_dir"]:
        run_dir = Path(state["run_dir"])
    else:
        ts = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(__file__).parent / "runs" / f"{ts}_{_slug(state['source'])}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state["run_dir"] = str(run_dir)
    return run_dir


def _compat_scout(source: str, run_dir: Path | None = None) -> dict[str, Any]:
    """Fallback scout for local files and text when agent unavailable."""
    path = Path(source)
    source_type = "text"

    if source.startswith("http://") or source.startswith("https://"):
        source_type = "url"
        return {
            "is_dataset": False,
            "source": source,
            "source_type": source_type,
            "columns": [],
            "dtypes": {},
            "num_rows": 0,
            "num_columns": 0,
            "sample_data": [],
            "raw_data_path": None,
            "confidence_score": 0.0,
            "rejection_reason": "URL scout fallback unavailable without agent support",
            "tables_found": 0,
            "timestamp": dt.datetime.now(timezone.utc).isoformat(),
        }

    if path.exists() and path.suffix.lower() == ".csv":
        source_type = "csv"
        df = pd.read_csv(path)
    elif path.exists() and path.suffix.lower() in {".xlsx", ".xls"}:
        source_type = "xlsx"
        df = pd.read_excel(path)
    elif path.exists() and path.suffix.lower() == ".json":
        source_type = "json"
        payload = _read_json(path)
        df = pd.json_normalize(payload) if isinstance(payload, dict) else pd.DataFrame(payload)
    else:
        source_type = "text"
        try:
            df = pd.read_csv(io.StringIO(source))
        except Exception:
            return {
                "is_dataset": False,
                "source": source,
                "source_type": source_type,
                "columns": [],
                "dtypes": {},
                "num_rows": 0,
                "num_columns": 0,
                "sample_data": [],
                "raw_data_path": None,
                "confidence_score": 0.0,
                "rejection_reason": "Source could not be parsed as tabular data",
                "tables_found": 0,
                "timestamp": dt.datetime.now(timezone.utc).isoformat(),
            }

    if df.empty:
        return {
            "is_dataset": False,
            "source": source,
            "source_type": source_type,
            "columns": [],
            "dtypes": {},
            "num_rows": 0,
            "num_columns": 0,
            "sample_data": [],
            "raw_data_path": None,
            "confidence_score": 0.0,
            "rejection_reason": "Parsed dataset is empty",
            "tables_found": 1,
            "timestamp": dt.datetime.now(timezone.utc).isoformat(),
        }

    # Save full DataFrame to raw_data.csv if run_dir provided
    raw_data_path: str | None = None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = run_dir / "raw_data.csv"
        df.to_csv(raw_csv, index=False)
        raw_data_path = str(raw_csv)
        print(f"[scout] Saved full dataset ({len(df)} rows) to {raw_csv}")

    dtypes = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            dtypes[str(col)] = "integer"
        elif pd.api.types.is_float_dtype(s):
            dtypes[str(col)] = "float"
        elif pd.api.types.is_bool_dtype(s):
            dtypes[str(col)] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            dtypes[str(col)] = "datetime"
        else:
            dtypes[str(col)] = "categorical"

    return {
        "is_dataset": True,
        "source": source,
        "source_type": source_type,
        "columns": [str(c) for c in df.columns],
        "dtypes": dtypes,
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "sample_data": df.head(5).astype(str).to_dict(orient="records"),
        "raw_data_path": raw_data_path,
        "confidence_score": 0.85,
        "rejection_reason": None,
        "tables_found": 1,
        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
    }


def _compat_labeler(run_dir: Path, scout_result: dict[str, Any]) -> None:
    """Fallback labeler when agent unavailable."""
    if not scout_result.get("is_dataset"):
        raise RuntimeError(scout_result.get("rejection_reason") or "Scout rejected input")

    # Priority 1: raw_data.csv in run_dir (full dataset written by scout)
    raw_csv = run_dir / "raw_data.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        print(f"[labeler] Loaded {len(df):,} rows from {raw_csv}")
    # Priority 2: raw_data_path from scout JSON
    elif scout_result.get("raw_data_path"):
        src_path = Path(scout_result["raw_data_path"])
        if src_path.exists():
            df = pd.read_csv(src_path)
            df.to_csv(raw_csv, index=False)
            print(f"[labeler] Loaded {len(df):,} rows from scout raw_data_path and copied to {raw_csv}")
        else:
            print(f"[labeler] Warning: raw_data_path {src_path} not found, falling back to sample_data")
            sample_data = scout_result.get("sample_data") or []
            if sample_data:
                df = pd.DataFrame(sample_data)
            else:
                cols = [str(c) for c in scout_result.get("columns", [])]
                df = pd.DataFrame(columns=cols)
    # Priority 3: sample_data fallback with warning
    else:
        print("[labeler] Warning: No raw_data.csv or raw_data_path found; using sample_data (only 5 rows)")
        sample_data = scout_result.get("sample_data") or []
        if sample_data:
            df = pd.DataFrame(sample_data)
        else:
            cols = [str(c) for c in scout_result.get("columns", [])]
            df = pd.DataFrame(columns=cols)

    if df.empty:
        raise RuntimeError("No rows available for cleaning")

    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.dropna(how="all").drop_duplicates()

    cleaned_csv = run_dir / "cleaned_data.csv"
    df.to_csv(cleaned_csv, index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        plt.hist(pd.to_numeric(df[num_cols[0]], errors="coerce").dropna(), bins=20, edgecolor="black")
        plt.title(f"Distribution of {num_cols[0]}")
        plt.xlabel(num_cols[0])
        plt.ylabel("Frequency")
    else:
        counts = df.iloc[:, 0].astype(str).value_counts().head(10)
        counts.plot(kind="bar")
        plt.title(f"Top values in {df.columns[0]}")
        plt.xlabel(df.columns[0])
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(run_dir / "visual_report.png", dpi=300)
    plt.close()

    phase2 = {
        "source": scout_result.get("source", "unknown"),
        "source_type": scout_result.get("source_type", "unknown"),
        "confidence_score": float(scout_result.get("confidence_score") or 0.0),
        "cleaned_data_csv": str(cleaned_csv),
        "visual_report_png": str(run_dir / "visual_report.png"),
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    _write_json(run_dir / "phase2_summary.json", phase2)


def _compat_analysis(run_dir: Path) -> None:
    """Fallback analysis when agent unavailable."""
    df = pd.read_csv(run_dir / "cleaned_data.csv")
    if df.empty:
        raise RuntimeError("cleaned_data.csv is empty")

    num_df = df.select_dtypes(include=[np.number])
    numeric_columns = [str(c) for c in num_df.columns]
    categorical_columns = [str(c) for c in df.columns if c not in num_df.columns]

    column_stats = []
    for col in df.columns:
        s = df[col]
        row: dict[str, Any] = {
            "column": str(col),
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "outlier_count": 0,
            "trend": None,
        }
        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce").dropna()
            q1 = float(sn.quantile(0.25)) if not sn.empty else 0.0
            q3 = float(sn.quantile(0.75)) if not sn.empty else 0.0
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            out = int(((s < lo) | (s > hi)).sum()) if not sn.empty else 0
            row.update(
                {
                    "mean": float(sn.mean()) if not sn.empty else None,
                    "median": float(sn.median()) if not sn.empty else None,
                    "std": float(sn.std()) if not sn.empty else None,
                    "min": float(sn.min()) if not sn.empty else None,
                    "max": float(sn.max()) if not sn.empty else None,
                    "q1": q1,
                    "q3": q3,
                    "outlier_count": out,
                    "trend": "flat",
                }
            )
        else:
            row.update({"mean": None, "median": None, "std": None, "min": None, "max": None, "q1": None, "q3": None})
        column_stats.append(row)

    correlations = []
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(corr.iloc[i, j])
                if np.isnan(r):
                    continue
                strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
                direction = "positive" if r >= 0 else "negative"
                correlations.append({
                    "col_a": str(cols[i]),
                    "col_b": str(cols[j]),
                    "pearson_r": round(r, 6),
                    "strength": strength,
                    "direction": direction,
                })

    anomalies = []
    outlier_rows: set[int] = set()
    for col in numeric_columns:
        sn = pd.to_numeric(df[col], errors="coerce")
        valid = sn.dropna()
        if valid.empty:
            continue
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        idx = sn[(sn < lo) | (sn > hi)].index.tolist()
        if idx:
            anomalies.append({
                "column": col,
                "row_indices": [int(i) for i in idx[:100]],
                "lower_fence": lo,
                "upper_fence": hi,
                "outlier_count": len(idx),
            })
            outlier_rows.update(int(i) for i in idx)

    key_insights = []
    if correlations:
        top = sorted(correlations, key=lambda c: abs(float(c["pearson_r"])), reverse=True)[0]
        key_insights.append(
            f"Strongest correlation: {top['col_a']} ↔ {top['col_b']} (r={top['pearson_r']:+.3f})."
        )
    if anomalies:
        top_a = sorted(anomalies, key=lambda a: int(a["outlier_count"]), reverse=True)[0]
        key_insights.append(f"Highest outlier count appears in {top_a['column']} ({top_a['outlier_count']}).")
    if not key_insights:
        key_insights.append("No strong anomalies or correlations detected.")

    analysis = {
        "source": str((_read_json(run_dir / "phase2_summary.json")).get("source", run_dir.name)) if (run_dir / "phase2_summary.json").exists() else run_dir.name,
        "source_type": str((_read_json(run_dir / "phase2_summary.json")).get("source_type", "unknown")) if (run_dir / "phase2_summary.json").exists() else "unknown",
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "column_stats": column_stats,
        "correlations": correlations,
        "categorical_insights": [],
        "anomalies": anomalies,
        "key_insights": key_insights,
        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
    }

    _write_json(run_dir / "analysis_result.json", analysis)
    report = [
        "# Analysis Report",
        "",
        f"- Rows: {analysis['num_rows']}",
        f"- Columns: {analysis['num_columns']}",
        "",
        "## Key Insights",
    ] + [f"- {x}" for x in key_insights]
    (run_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    enriched = df.copy()
    enriched["_is_outlier"] = enriched.index.isin(outlier_rows)
    enriched.to_csv(run_dir / "enriched_data.csv", index=False)

    phase3 = {
        "source": analysis["source"],
        "source_type": analysis["source_type"],
        "num_rows": analysis["num_rows"],
        "num_columns": analysis["num_columns"],
        "analysis_result_json": str(run_dir / "analysis_result.json"),
        "analysis_report_md": str(run_dir / "analysis_report.md"),
        "enriched_data_csv": str(run_dir / "enriched_data.csv"),
    }
    _write_json(run_dir / "phase3_summary.json", phase3)


def _compat_visualization(run_dir: Path) -> None:
    """Fallback visualization when agent unavailable."""
    df = pd.read_csv(run_dir / "enriched_data.csv")
    analysis = _read_json(run_dir / "analysis_result.json") if (run_dir / "analysis_result.json").exists() else {}

    charts = []
    sns.set_theme(style="whitegrid")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        out = run_dir / f"hist_{col}.png"
        plt.figure(figsize=(9, 6))
        plt.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=20, edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        charts.append({
            "chart_id": f"hist_{col}",
            "chart_type": "histogram",
            "title": f"Distribution of {col}",
            "columns_used": [str(col)],
            "png_path": str(out),
            "generated_by": "builtin",
        })

    if len(numeric_cols) >= 2:
        a, b = numeric_cols[0], numeric_cols[1]
        out = run_dir / f"scatter_{a}_{b}.png"
        plt.figure(figsize=(9, 6))
        plt.scatter(df[a], df[b], alpha=0.7)
        plt.title(f"{a} vs {b}")
        plt.xlabel(a)
        plt.ylabel(b)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        charts.append({
            "chart_id": f"scatter_{a}_{b}",
            "chart_type": "scatter",
            "title": f"{a} vs {b}",
            "columns_used": [str(a), str(b)],
            "png_path": str(out),
            "generated_by": "builtin",
        })

    if not charts:
        out = run_dir / "table_preview.png"
        plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.title("Dataset Preview")
        preview = df.head(10).astype(str)
        table = plt.table(cellText=preview.values, colLabels=preview.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        charts.append({
            "chart_id": "table_preview",
            "chart_type": "table",
            "title": "Dataset Preview",
            "columns_used": [str(c) for c in df.columns],
            "png_path": str(out),
            "generated_by": "builtin",
        })

    summary = {
        "source": analysis.get("source", run_dir.name),
        "source_type": analysis.get("source_type", "unknown"),
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "charts": charts,
        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "viz_summary.json", summary)



def scout_node(state: PipelineState) -> PipelineState:
    state["phase"] = "scout"
    state["status"] = "running"

    if not state["source"] or not state["source"].strip():
        return _state_error(state, "scout", "SCOUT_ERROR: source is empty")

    try:
        run_dir = _resolve_run_dir(state)

        # For uploaded files, skip actual scouting and create a synthetic scout result
        if state["source_type"] == "uploaded_file":
            path = Path(state["source"])
            if not path.exists():
                return _state_error(state, "scout", f"SCOUT_ERROR: Uploaded file not found: {state['source']}")
            
            # Read the file to determine basic info
            file_ext = path.suffix.lower()
            result: dict[str, Any]
            
            try:
                if file_ext == ".csv":
                    df = pd.read_csv(path)
                    result = {
                        "is_dataset": True,
                        "source": state["source"],
                        "source_type": "csv",
                        "columns": list(df.columns.astype(str)),
                        "dtypes": {col: ("categorical" if df[col].dtype == "object" else "numeric") for col in df.columns},
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "sample_data": df.head(5).astype(str).to_dict(orient="records"),
                        "raw_data_path": None,
                        "confidence_score": 1.0,
                        "rejection_reason": None,
                        "tables_found": 1,
                        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
                    }
                    df.to_csv(run_dir / "raw_data.csv", index=False)
                    print(f"[scout] Saved uploaded CSV ({len(df)} rows) to raw_data.csv")
                elif file_ext in {".xlsx", ".xls"}:
                    df = pd.read_excel(path)
                    result = {
                        "is_dataset": True,
                        "source": state["source"],
                        "source_type": "xlsx",
                        "columns": list(df.columns.astype(str)),
                        "dtypes": {col: ("categorical" if df[col].dtype == "object" else "numeric") for col in df.columns},
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "sample_data": df.head(5).astype(str).to_dict(orient="records"),
                        "raw_data_path": None,
                        "confidence_score": 1.0,
                        "rejection_reason": None,
                        "tables_found": 1,
                        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
                    }
                    df.to_csv(run_dir / "raw_data.csv", index=False)
                    print(f"[scout] Saved uploaded Excel ({len(df)} rows) to raw_data.csv")
                elif file_ext == ".json":
                    payload = _read_json(path)
                    df = pd.json_normalize(payload) if isinstance(payload, dict) else pd.DataFrame(payload)
                    result = {
                        "is_dataset": True,
                        "source": state["source"],
                        "source_type": "json",
                        "columns": list(df.columns.astype(str)),
                        "dtypes": {col: ("categorical" if df[col].dtype == "object" else "numeric") for col in df.columns},
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "sample_data": df.head(5).astype(str).to_dict(orient="records"),
                        "raw_data_path": None,
                        "confidence_score": 1.0,
                        "rejection_reason": None,
                        "tables_found": 1,
                        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
                    }
                    df.to_csv(run_dir / "raw_data.csv", index=False)
                    print(f"[scout] Saved uploaded JSON ({len(df)} rows) to raw_data.csv")
                else:
                    return _state_error(state, "scout", f"SCOUT_ERROR: Unsupported file format: {file_ext}")
            except Exception as e:
                return _state_error(state, "scout", f"SCOUT_ERROR: Failed to read uploaded file: {e}")
            
            _write_json(run_dir / "scout_result.json", result)
            state["scout_output_dir"] = str(run_dir)
            state["status"] = "success"
            return state

        # For non-uploaded files, proceed with normal scouting logic
        result: dict[str, Any]
        try:
            from .agents.scout import scout

            result = scout(state["source"].strip())
        except Exception:
            cmd = [sys.executable, str(Path(__file__).parent / "agents" / "scout.py"), state["source"].strip()]
            code, stdout, stderr = _run_cmd(cmd)
            if code == 0:
                result = _json_from_output(stdout)
            else:
                result = _compat_scout(state["source"].strip(), run_dir)
                if not result.get("is_dataset"):
                    return _state_error(state, "scout", f"SCOUT_ERROR: {stderr or stdout or result.get('rejection_reason')}" )

        _write_json(run_dir / "scout_result.json", result)

        if not bool(result.get("is_dataset", False)):
            compat = _compat_scout(state["source"].strip(), run_dir)
            if bool(compat.get("is_dataset", False)):
                result = compat
                _write_json(run_dir / "scout_result.json", result)
            else:
                reason = result.get("rejection_reason") or compat.get("rejection_reason") or "Source was not accepted as dataset"
                return _state_error(state, "scout", f"SCOUT_ERROR: {reason}")

        state["scout_output_dir"] = str(run_dir)
        state["status"] = "success"
        return state
    except Exception as e:
        return _state_error(state, "scout", f"SCOUT_ERROR: {e}")


def labeler_node(state: PipelineState) -> PipelineState:
    if state["status"] == "error":
        return state

    state["phase"] = "labeler"
    state["status"] = "running"

    if not state["scout_output_dir"]:
        return _state_error(state, "labeler", "LABELER_ERROR: Missing scout output directory")

    try:
        run_dir = Path(state["scout_output_dir"])
        scout_json_path = run_dir / "scout_result.json"
        if not scout_json_path.exists():
            return _state_error(state, "labeler", f"LABELER_ERROR: Missing file {scout_json_path}")

        completed = False
        try:
            from .agents.labeler import run_labeler

            run_labeler(
                scout_json_path=scout_json_path,
                model=os.getenv("OLLAMA_PHASE2_MODEL", "qwen2.5-coder:3b"),
                ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            )
            completed = True
        except Exception:
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "agents" / "labeler.py"),
                str(scout_json_path),
                "--model",
                os.getenv("OLLAMA_PHASE2_MODEL", "qwen2.5-coder:3b"),
                "--ollama-host",
                os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ]
            code, stdout, stderr = _run_cmd(cmd)
            completed = code == 0
            if not completed:
                scout_result = _read_json(scout_json_path)
                _compat_labeler(run_dir, scout_result)
                completed = True

        if not completed:
            return _state_error(state, "labeler", "LABELER_ERROR: Failed to complete labeler stage")

        _write_json(run_dir / "scout_input.json", _read_json(scout_json_path))

        cleaned_csv = run_dir / "cleaned_data.csv"
        if not cleaned_csv.exists() or cleaned_csv.stat().st_size == 0:
            return _state_error(state, "labeler", f"LABELER_ERROR: Missing cleaned output {cleaned_csv}")

        state["labeler_output_dir"] = str(run_dir)
        state["status"] = "success"
        return state
    except Exception as e:
        return _state_error(state, "labeler", f"LABELER_ERROR: {e}")


def analyst_node(state: PipelineState) -> PipelineState:
    if state["status"] == "error":
        return state

    state["phase"] = "analyst"
    state["status"] = "running"

    if not state["labeler_output_dir"]:
        return _state_error(state, "analyst", "ANALYST_ERROR: Missing labeler output directory")

    try:
        run_dir = Path(state["labeler_output_dir"])
        cleaned_csv = run_dir / "cleaned_data.csv"
        if not cleaned_csv.exists():
            return _state_error(state, "analyst", f"ANALYST_ERROR: Missing file {cleaned_csv}")

        completed = False
        try:
            from .agents.analyst import run_analysis

            run_analysis(
                cleaned_csv_path=cleaned_csv,
                out_dir=run_dir,
                model=os.getenv("OLLAMA_PHASE3_MODEL", "llama3.2:3b"),
                ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                use_llm=False,
            )
            completed = True
        except Exception:
            _compat_analysis(run_dir)
            completed = True

        if not completed:
            return _state_error(state, "analyst", "ANALYST_ERROR: Failed to complete analyst stage")

        if not (run_dir / "analysis_result.json").exists() or not (run_dir / "enriched_data.csv").exists():
            return _state_error(state, "analyst", "ANALYST_ERROR: Missing analysis_result.json or enriched_data.csv")

        state["analyst_output_dir"] = str(run_dir)
        state["status"] = "success"
        return state
    except Exception as e:
        return _state_error(state, "analyst", f"ANALYST_ERROR: {e}")


def artist_node(state: PipelineState) -> PipelineState:
    if state["status"] == "error":
        return state

    state["phase"] = "artist"
    state["status"] = "running"

    if not state["analyst_output_dir"]:
        return _state_error(state, "artist", "ARTIST_ERROR: Missing analyst output directory")

    try:
        run_dir = Path(state["analyst_output_dir"])
        enriched_csv = run_dir / "enriched_data.csv"
        analysis_json = run_dir / "analysis_result.json"

        if not enriched_csv.exists():
            return _state_error(state, "artist", f"ARTIST_ERROR: Missing file {enriched_csv}")

        completed = False
        try:
            from .agents.artist import run_visualization

            run_visualization(
                enriched_csv_path=enriched_csv,
                analysis_json_path=analysis_json if analysis_json.exists() else None,
                out_dir=run_dir,
                model=os.getenv("OLLAMA_PHASE4_MODEL", "qwen2.5-coder:3b"),
                ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                use_llm=False,
            )
            completed = True
        except Exception:
            _compat_visualization(run_dir)
            completed = True

        if not completed or not (run_dir / "viz_summary.json").exists():
            return _state_error(state, "artist", "ARTIST_ERROR: Missing viz_summary.json")

        state["artist_output_dir"] = str(run_dir)
        state["status"] = "success"
        return state
    except Exception as e:
        return _state_error(state, "artist", f"ARTIST_ERROR: {e}")


def validator_node(state: PipelineState) -> PipelineState:
    if state["status"] == "error":
        return state

    state["phase"] = "validator"
    state["status"] = "running"

    if not state["artist_output_dir"]:
        return _state_error(state, "validator", "VALIDATOR_ERROR: Missing artist output directory")

    try:
        run_dir = Path(state["artist_output_dir"])

        try:
            from .agents.validator import run_validation

            result = run_validation(run_dir=run_dir)
        except Exception:
            cmd = [sys.executable, str(Path(__file__).parent / "agents" / "validator.py"), str(run_dir)]
            code, stdout, stderr = _run_cmd(cmd)
            if code != 0:
                return _state_error(state, "validator", f"VALIDATOR_ERROR: {stderr or stdout}")
            result = _json_from_output(stdout)

        if result.get("status") == "REJECTED":
            report_path = run_dir / "validation_report.md"
            report_content = report_path.read_text() if report_path.exists() else "Validation report not available"
            print(f"[validator] Validation REJECTED with report:\n{report_content}")
            state["validator_output_dir"] = str(run_dir)
            state["phase"] = "complete"
            state["status"] = "rejected"
            return state

        if not (run_dir / "validation_result.json").exists() or not (run_dir / "validation_report.md").exists():
            return _state_error(state, "validator", "VALIDATOR_ERROR: Missing validation outputs")

        state["validator_output_dir"] = str(run_dir)
        state["phase"] = "complete"
        state["status"] = "success"
        return state
    except Exception as e:
        return _state_error(state, "validator", f"VALIDATOR_ERROR: {e}")


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("scout", scout_node)
    graph.add_node("labeler", labeler_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("artist", artist_node)
    graph.add_node("validator", validator_node)

    graph.add_edge(START, "scout")
    graph.add_edge("scout", "labeler")
    graph.add_edge("labeler", "analyst")
    graph.add_edge("analyst", "artist")
    graph.add_edge("artist", "validator")
    graph.add_edge("validator", END)
    return graph.compile()


_RUNS: dict[str, PipelineState] = {}


def create_run(source: str, source_type: str = "auto") -> str:
    """Create a new run. source_type can be 'auto', 'url', 'csv', 'xlsx', 'json', or 'uploaded_file'."""
    run_id = str(uuid.uuid4())[:8]
    
    # Auto-detect source_type if not specified
    if source_type == "auto":
        if source.startswith("http://") or source.startswith("https://"):
            source_type = "url"
        elif source.startswith(("backend/runs/uploads/", "/runs/uploads/")):
            source_type = "uploaded_file"
        elif source.lower().endswith(".csv"):
            source_type = "csv"
        elif source.lower().endswith((".xlsx", ".xls")):
            source_type = "xlsx"
        elif source.lower().endswith(".json"):
            source_type = "json"
        else:
            source_type = "url"  # Default to treating unknown as URL
    
    _RUNS[run_id] = {
        "run_id": run_id,
        "source": source,
        "phase": "pending",
        "status": "pending",
        "source_type": source_type,
        "error": None,
        "run_dir": None,
        "scout_output_dir": None,
        "labeler_output_dir": None,
        "analyst_output_dir": None,
        "artist_output_dir": None,
        "validator_output_dir": None,
    }
    return run_id


def get_run(run_id: str) -> PipelineState | None:
    return _RUNS.get(run_id)


def execute_run(run_id: str) -> None:
    state = _RUNS.get(run_id)
    if not state:
        return

    try:
        graph = build_graph()
        state_dict: PipelineState = dict(state)  # TypedDict to dict

        for output in graph.stream(state_dict):
            for _, node_state in output.items():
                if isinstance(node_state, dict):
                    _RUNS[run_id] = dict(node_state)  # Update stored state after each node
                else:
                    _RUNS[run_id] = node_state
    except Exception as e:
        # Persist the error state even on exception
        current_state = _RUNS.get(run_id, {})
        if isinstance(current_state, dict):
            current_state["status"] = "error"
            current_state["error"] = f"EXECUTION_ERROR: {e}"
            _RUNS[run_id] = current_state
        else:
            current_state["status"] = "error"
            current_state["error"] = f"EXECUTION_ERROR: {e}"
            _RUNS[run_id] = current_state
        print(f"[execute_run] Exception during graph execution: {e}")
        import traceback
        traceback.print_exc()


__all__ = ["PipelineState", "create_run", "get_run", "execute_run", "build_graph"]
