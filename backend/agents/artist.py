from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class UnsafeGeneratedCodeError(RuntimeError):
    pass


_DISALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
)

_DISALLOWED_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "__import__",
    "compile",
    "input",
    "globals",
    "locals",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:80] or "run"


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ollama_generate(model: str, prompt: str, host: str) -> str:
    """
    Prefer the `ollama` Python library; fall back to the CLI.
    keep_alive=0 requests immediate unload after completion.
    """
    try:
        import ollama  # type: ignore

        client = ollama.Client(host=host)
        resp = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.1},
            keep_alive=0,
        )
        return str(resp.get("response", "")).strip()
    except Exception:
        completed = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout.decode("utf-8", errors="replace").strip()
    finally:
        subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
                    raise UnsafeGeneratedCodeError(f"Disallowed module usage: {node.func.value.id}")

    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = {"make_distribution_plot", "make_correlation_heatmap"} - fn_names
    if missing:
        raise UnsafeGeneratedCodeError(f"Missing required function(s): {', '.join(sorted(missing))}")


def _compile_generated_functions(
    code: str,
) -> tuple[Callable[[pd.DataFrame, str, str], None], Callable[[pd.DataFrame, str], None]]:
    _validate_generated_module(code)

    safe_builtins = {
        "None": None,
        "True": True,
        "False": False,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "str": str,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
    }

    globals_ns: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "sns": sns,
        "plt": plt,
    }
    locals_ns: dict[str, Any] = {}

    compiled = compile(code, "<llm_generated_phase3>", "exec")
    exec(compiled, globals_ns, locals_ns)

    dist_fn = locals_ns.get("make_distribution_plot")
    heat_fn = locals_ns.get("make_correlation_heatmap")
    if not callable(dist_fn) or not callable(heat_fn):
        raise UnsafeGeneratedCodeError("Generated functions were not callable.")

    return dist_fn, heat_fn


def _select_primary_numeric_column(df: pd.DataFrame) -> str:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Prefer columns that look like measurements rather than IDs.
    deprioritize = {"id", "index", "source_url", "publication_date"}
    for c in numeric_cols:
        if str(c).lower() not in deprioritize:
            return str(c)
    if numeric_cols:
        return str(numeric_cols[0])
    raise RuntimeError("No numeric columns found for distribution plot.")


def _build_phase3_prompt(df_preview_csv: str, model_name: str, primary_col: str) -> str:
    return (
        "You are a scientific visualization agent.\n"
        "Generate Python code ONLY. No markdown. No explanations.\n"
        "The code MUST define exactly two functions with these signatures:\n\n"
        "def make_distribution_plot(df: pd.DataFrame, primary_col: str, out_png_path: str) -> None:\n"
        "    ...\n\n"
        "def make_correlation_heatmap(df: pd.DataFrame, out_png_path: str) -> None:\n"
        "    ...\n\n"
        "Plot requirements:\n"
        "- Use seaborn and matplotlib.\n"
        "- Use sns.set_context('paper') and sns.set_theme(style='whitegrid').\n"
        "- Call plt.tight_layout().\n"
        "- Save high-DPI PNGs (dpi>=300).\n"
        "- Professional axis labels and titles.\n\n"
        "Distribution plot:\n"
        f"- Create a distribution plot for primary_col='{primary_col}' (histogram with KDE if appropriate).\n\n"
        "Correlation heatmap:\n"
        "- Create a correlation heatmap of numeric label columns.\n"
        "- Exclude obviously non-label identifier columns if present (e.g., source_url).\n\n"
        "Safety constraints:\n"
        "- Do NOT import anything.\n"
        "- Do NOT read/write any files except saving figures to out_png_path.\n"
        "- Do NOT use os, sys, subprocess, pathlib, requests.\n"
        "- Do NOT call open(), exec(), eval().\n\n"
        f"Model context: {model_name}\n"
        "Here is a CSV preview of the DataFrame (first rows):\n"
        f"{df_preview_csv}\n"
    )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Phase 3 Artist: publication-ready plots via local Ollama.")
    parser.add_argument(
        "cleaned_csv",
        help="Path to cleaned CSV (e.g., cleaned_research_data.csv or cleaned_data.csv)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE3_MODEL", "qwen2.5-coder:3b"),
        help="Ollama model for Phase 3 (default: env OLLAMA_PHASE3_MODEL or qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL (default: env OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--out-dir",
        default="runs",
        help="Output directory under backend/ (default: runs)",
    )
    args = parser.parse_args()

    cleaned_path = Path(args.cleaned_csv)
    if not cleaned_path.exists():
        raise FileNotFoundError(str(cleaned_path))

    df = pd.read_csv(cleaned_path)
    if df.empty:
        raise RuntimeError("Cleaned CSV contained no rows.")

    primary_col = _select_primary_numeric_column(df)
    df_preview_csv = df.head(8).to_csv(index=False)

    run_ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("backend") / args.out_dir / f"{run_ts}_{_safe_slug(cleaned_path.stem)}"
    _ensure_dir(run_dir)

    _write_text(run_dir / "df_preview.csv", df_preview_csv)
    _write_text(run_dir / "primary_col.txt", primary_col)

    prompt = _build_phase3_prompt(df_preview_csv=df_preview_csv, model_name=args.model, primary_col=primary_col)
    code = _ollama_generate(model=args.model, prompt=prompt, host=args.ollama_host)
    _write_text(run_dir / "llm_generated_phase3.py", code)

    dist_fn, heat_fn = _compile_generated_functions(code)

    dist_png = run_dir / "distribution_primary_variable.png"
    heat_png = run_dir / "correlation_heatmap_labels.png"

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    dist_fn(df.copy(), primary_col, str(dist_png))

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    heat_fn(df.copy(), str(heat_png))

    summary = {
        "input_csv": str(cleaned_path),
        "primary_numeric_column": primary_col,
        "distribution_png": str(dist_png),
        "correlation_heatmap_png": str(heat_png),
        "rows": int(df.shape[0]),
        "columns": list(map(str, df.columns)),
    }
    _write_json(run_dir / "phase3_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
