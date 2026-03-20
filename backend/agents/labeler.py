"""
Labeler Agent — Phase 2
Receives a ScoutResult JSON (from scout_agent.py), cleans the dataset,
and produces a seaborn visual report.

Input  : ScoutResult JSON  (is_dataset, columns, dtypes, sample_data,
                             num_rows, num_columns, source, source_type, …)
Output : cleaned_data.csv + visual_report.png + phase2_summary.json
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
from datetime import timezone
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Input schema — mirrors ScoutResult from scout_agent.py
# ---------------------------------------------------------------------------

class ScoutResult(BaseModel):
    is_dataset: bool
    source: str = Field(min_length=1)
    source_type: str                        # "url" | "csv" | "xlsx" | "json" | "text"
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    num_rows: int = 0
    num_columns: int = 0
    sample_data: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    rejection_reason: str | None = None
    tables_found: int = 0
    timestamp: str = ""

    # ------------------------------------------------------------------
    # Back-compat: accept old Phase1Metadata fields so existing run dirs
    # still work without re-scouting.
    # ------------------------------------------------------------------
    source_url: str | None = None           # old field → maps to source
    primary_topic: str | None = None
    publication_date: str | None = None
    raw_quantitative_stats: Any = None

    def effective_source(self) -> str:
        return self.source or self.source_url or "unknown"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame from the scout output using the best available data:
          1. Full source file for file-based inputs (csv/xlsx/json)
          2. sample_data  (list-of-dicts from HTML / file scraping)
          3. raw_quantitative_stats (old Phase1 path)
          4. Minimal metadata row as last resort
        """
        source_path = Path(self.source)

        # ---- Path 1: load full local dataset when available ----
        if self.source_type in {"csv", "xlsx", "json"} and source_path.exists():
            if self.source_type == "csv":
                df = pd.read_csv(source_path)
            elif self.source_type == "xlsx":
                df = pd.read_excel(source_path)
            else:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                df = pd.json_normalize(payload) if isinstance(payload, dict) else pd.DataFrame(payload)

            for col, dtype in self.dtypes.items():
                if col not in df.columns:
                    continue
                if dtype in ("integer", "float", "numeric"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "boolean":
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map({"true": True, "false": False})
                        .astype("boolean")
                    )
            return df

        # ---- Path 1: new scout sample_data ----
        if self.sample_data:
            df = pd.DataFrame(self.sample_data)
            # Apply the dtype hints the scout already inferred
            for col, dtype in self.dtypes.items():
                if col not in df.columns:
                    continue
                if dtype in ("integer", "float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "boolean":
                    df[col] = df[col].map(
                        {"True": True, "False": False, "true": True, "false": False}
                    ).astype("boolean")
            return df

        # ---- Path 2: old raw_quantitative_stats ----
        if self.raw_quantitative_stats is not None:
            stats = self.raw_quantitative_stats
            base = {
                "source": self.effective_source(),
                "publication_date": self.publication_date,
                "primary_topic": self.primary_topic,
            }
            if isinstance(stats, dict):
                flat = pd.json_normalize(stats, sep="__")
                for k, v in base.items():
                    flat[k] = v
                return flat
            return pd.DataFrame([{**base, "raw_quantitative_stats": stats}])

        # ---- Path 3: bare metadata row ----
        return pd.DataFrame([{
            "source": self.effective_source(),
            "source_type": self.source_type,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "confidence_score": self.confidence_score,
        }])


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:80] or "run"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_alive(host: str) -> bool:
    try:
        import requests as req
        return req.get(f"{host}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _generate_fallback_cleaning_code() -> str:
    return """
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')
        for c in df.columns
    ]
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def make_visual_report(df: pd.DataFrame, out_png_path: str) -> None:
    sns.set_theme(style='whitegrid')
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    n = min(len(numeric_cols), 4)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No numeric columns to plot',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    elif n == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[numeric_cols[0]].dropna(), bins=20, ax=ax, kde=True)
        ax.set_title(f'Distribution of {numeric_cols[0]}')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel('Frequency')
    else:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        for i, col in enumerate(numeric_cols[:n]):
            sns.histplot(df[col].dropna(), bins=20, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300)
    plt.close()
"""


def _ollama_generate(model: str, prompt: str, host: str) -> str:
    if not _ollama_alive(host):
        return _generate_fallback_cleaning_code()

    try:
        import ollama  # type: ignore
        client = ollama.Client(host=host)
        resp = client.generate(
            model=model, prompt=prompt, stream=False,
            options={"temperature": 0.1}, keep_alive=0,
        )
        text = str(resp.get("response", "")).strip()
        # Strip markdown fences if present
        if "```" in text:
            m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        return text or _generate_fallback_cleaning_code()
    except Exception:
        try:
            completed = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, check=True,
            )
            text = completed.stdout.decode(errors="replace").strip()
            return text or _generate_fallback_cleaning_code()
        except Exception:
            return _generate_fallback_cleaning_code()
    finally:
        subprocess.run(["ollama", "stop", model],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Code safety: AST validation
# ---------------------------------------------------------------------------

class UnsafeGeneratedCodeError(RuntimeError):
    pass


_DISALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
    ast.Try, ast.Raise, ast.Lambda, ast.Global, ast.Nonlocal,
)

_DISALLOWED_CALL_NAMES = {
    "open", "exec", "eval", "__import__", "compile", "input", "globals", "locals",
}


def _validate_generated_module(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_AST_NODES):
            raise UnsafeGeneratedCodeError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
                raise UnsafeGeneratedCodeError(f"Disallowed call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"os", "sys", "subprocess", "pathlib", "shutil"}:
                    raise UnsafeGeneratedCodeError(
                        f"Disallowed module usage: {node.func.value.id}"
                    )
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = {"clean_dataframe", "make_visual_report"} - fn_names
    if missing:
        raise UnsafeGeneratedCodeError(
            f"Missing required function(s): {', '.join(sorted(missing))}"
        )


def _strip_unsafe_lines(code: str) -> str:
    """Remove import / try-except lines that would fail AST validation."""
    lines = code.split("\n")
    filtered, skip_indent = [], None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            continue
        if stripped.startswith("try"):
            skip_indent = len(line) - len(line.lstrip())
            continue
        if skip_indent is not None:
            cur = len(line) - len(line.lstrip()) if line.strip() else skip_indent + 1
            if stripped.startswith(("except", "finally")):
                continue
            if line.strip() and cur <= skip_indent:
                skip_indent = None          # exited the try block
        if stripped == "pass" and skip_indent is not None:
            continue
        filtered.append(line)

    return "\n".join(filtered)


def _compile_generated_functions(
    code: str,
) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], Callable[[pd.DataFrame, str], None]]:
    code = _strip_unsafe_lines(code)
    Path("debug_generated_code.py").write_text(code)
    _validate_generated_module(code)

    safe_builtins = {
        "None": None, "True": True, "False": False,
        "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
        "sorted": sorted, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
        "float": float, "int": int, "str": str,
        "dict": dict, "list": list, "set": set, "tuple": tuple,
        "isinstance": isinstance, "print": print,
    }
    globals_ns: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd, "np": np, "sns": sns, "plt": plt, "re": re,
    }
    locals_ns: dict[str, Any] = {}

    exec(compile(code, "<llm_generated_phase2>", "exec"), globals_ns, locals_ns)

    clean_fn = locals_ns.get("clean_dataframe")
    viz_fn = locals_ns.get("make_visual_report")
    if not callable(clean_fn) or not callable(viz_fn):
        raise UnsafeGeneratedCodeError("Generated functions were not callable.")
    return clean_fn, viz_fn


# ---------------------------------------------------------------------------
# Prompt builder — now dtype-aware thanks to ScoutResult
# ---------------------------------------------------------------------------

def _build_phase2_prompt(
    df_preview_csv: str,
    model_name: str,
    dtypes: dict[str, str],
    source_type: str,
    num_rows: int,
) -> str:
    dtype_hint = (
        "\nColumn dtype hints from Scout Agent:\n"
        + "\n".join(f"  - {col}: {dtype}" for col, dtype in dtypes.items())
        if dtypes else ""
    )

    return (
        "You are a senior data engineer and scientific visualization specialist.\n"
        "Generate Python code ONLY. No markdown. No explanations.\n"
        "The code MUST define exactly two functions:\n\n"
        "def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    ...\n\n"
        "def make_visual_report(df: pd.DataFrame, out_png_path: str) -> None:\n"
        "    ...\n\n"
        "Cleaning requirements:\n"
        "- Only drop rows that are completely empty: df.dropna(how='all').\n"
        "- Standardize column names to snake_case (no spaces, no symbols).\n"
        "- Use the dtype hints below to cast columns correctly.\n"
        "- Return the cleaned DataFrame.\n"
        f"{dtype_hint}\n\n"
        "Visualization requirements:\n"
        "- Use seaborn and matplotlib.\n"
        "- Call sns.set_theme(style='whitegrid').\n"
        f"- The data came from source_type='{source_type}' with ~{num_rows} rows.\n"
        "- Choose the most informative plot(s) for the available columns and dtypes.\n"
        "- Label all axes professionally with a descriptive title.\n"
        "- Save high-DPI PNG to out_png_path (dpi>=300). Call plt.close().\n\n"
        "Strict safety constraints:\n"
        "- Do NOT import anything. pd, np, sns, plt, re are already available.\n"
        "- Do NOT use try/except, with, raise, lambda, import.\n"
        "- Do NOT access os, sys, subprocess, pathlib, open, exec, eval.\n\n"
        f"Model: {model_name}\n"
        "CSV preview of the DataFrame (first rows):\n"
        f"{df_preview_csv}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_labeler(
    scout_json_path: Path,
    model: str,
    ollama_host: str,
    out_dir: str = "runs",
) -> dict[str, Any]:
    """
    Core labeler logic — importable by Streamlit or other callers.
    Returns the phase2_summary dict.
    """
    raw_obj = _read_json(scout_json_path)

    # Accept both new ScoutResult and old Phase1Metadata shapes
    try:
        scout = ScoutResult.model_validate(raw_obj)
    except ValidationError:
        # Last-ditch: wrap old format into a minimal ScoutResult
        scout = ScoutResult(
            is_dataset=True,
            source=raw_obj.get("source_url", "unknown"),
            source_type="url",
            raw_quantitative_stats=raw_obj.get("raw_quantitative_stats"),
            primary_topic=raw_obj.get("primary_topic"),
            publication_date=raw_obj.get("publication_date"),
        )

    if not scout.is_dataset:
        raise ValueError(
            f"Scout rejected the input as not a dataset: {scout.rejection_reason}"
        )

    df = scout.to_dataframe()

    if df.empty:
        raise RuntimeError("Scout result produced an empty DataFrame — nothing to label.")

    df_preview_csv = df.head(8).to_csv(index=False)

    # Use scout's output dir (parent of JSON) or create new run dir
    run_dir = scout_json_path.parent
    _ensure_dir(run_dir)

    _write_json(run_dir / "scout_input.json", raw_obj)
    _write_text(run_dir / "df_preview.csv", df_preview_csv)

    prompt = _build_phase2_prompt(
        df_preview_csv=df_preview_csv,
        model_name=model,
        dtypes=scout.dtypes,
        source_type=scout.source_type,
        num_rows=scout.num_rows,
    )

    code = _ollama_generate(model=model, prompt=prompt, host=ollama_host)
    _write_text(run_dir / "llm_generated_phase2.py", code)

    try:
        clean_fn, viz_fn = _compile_generated_functions(code)
    except (UnsafeGeneratedCodeError, SyntaxError) as e:
        # Fall back to safe built-in functions
        print(f"[labeler] Code validation failed ({e}), using fallback functions.")
        clean_fn, viz_fn = _compile_generated_functions(_generate_fallback_cleaning_code())

    cleaned = clean_fn(df.copy())
    if not isinstance(cleaned, pd.DataFrame):
        raise RuntimeError("clean_dataframe() did not return a pandas DataFrame.")

    cleaned_csv_path = run_dir / "cleaned_data.csv"
    cleaned.to_csv(cleaned_csv_path, index=False)

    report_png_path = run_dir / "visual_report.png"
    sns.set_theme(style="whitegrid")
    viz_fn(cleaned, str(report_png_path))

    summary = {
        "source": scout.effective_source(),
        "source_type": scout.source_type,
        "confidence_score": scout.confidence_score,
        "cleaned_data_csv": str(cleaned_csv_path),
        "visual_report_png": str(report_png_path),
        "rows": int(cleaned.shape[0]),
        "columns": list(map(str, cleaned.columns)),
        "dtypes": {col: str(cleaned[col].dtype) for col in cleaned.columns},
    }
    _write_json(run_dir / "phase2_summary.json", summary)
    return summary


def main() -> int:
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Labeler Agent (Phase 2): clean + visualise a ScoutResult JSON."
    )
    parser.add_argument(
        "scout_json",
        help="Path to scout_agent output JSON (scout_result.json or extracted_metadata.json)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE2_MODEL", "qwen2.5-coder:3b"),
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    path = Path(args.scout_json)
    if not path.exists():
        raise FileNotFoundError(str(path))

    summary = run_labeler(
        scout_json_path=path,
        model=args.model,
        ollama_host=args.ollama_host,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValidationError, ValueError) as e:
        raise SystemExit(f"[labeler] Input error: {e}")