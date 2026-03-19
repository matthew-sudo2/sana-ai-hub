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
from pydantic import BaseModel, Field, ValidationError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Phase1Metadata(BaseModel):
    source_url: str = Field(min_length=1)
    publication_date: str | None = None
    primary_topic: str = Field(min_length=1)
    raw_quantitative_stats: Any


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


def _validate_generated_module(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_AST_NODES):
            raise UnsafeGeneratedCodeError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            # Disallow direct calls to dangerous builtins.
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
                raise UnsafeGeneratedCodeError(f"Disallowed call: {node.func.id}")
            # Disallow obvious process and filesystem vectors.
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"os", "sys", "subprocess", "pathlib", "shutil"}:
                    raise UnsafeGeneratedCodeError(f"Disallowed module usage: {node.func.value.id}")

    # Require the two functions we expect.
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = {"clean_dataframe", "make_visual_report"} - fn_names
    if missing:
        raise UnsafeGeneratedCodeError(f"Missing required function(s): {', '.join(sorted(missing))}")


def _compile_generated_functions(code: str) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], Callable[[pd.DataFrame, str], None]]:
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

    compiled = compile(code, "<llm_generated_phase2>", "exec")
    exec(compiled, globals_ns, locals_ns)

    clean_fn = locals_ns.get("clean_dataframe")
    viz_fn = locals_ns.get("make_visual_report")
    if not callable(clean_fn) or not callable(viz_fn):
        raise UnsafeGeneratedCodeError("Generated functions were not callable.")

    return clean_fn, viz_fn


def _metadata_to_dataframe(meta: Phase1Metadata) -> pd.DataFrame:
    """
    Converts Phase 1 JSON into a DataFrame.
    If `raw_quantitative_stats` is a dict, flatten keys.
    If it's a list, keep as a column and let cleaning logic expand if needed.
    """
    base = {
        "source_url": meta.source_url,
        "publication_date": meta.publication_date,
        "primary_topic": meta.primary_topic,
    }

    stats = meta.raw_quantitative_stats
    if isinstance(stats, dict):
        flat = pd.json_normalize(stats, sep="__")
        for k in base:
            flat[k] = base[k]
        return flat

    df = pd.DataFrame([{**base, "raw_quantitative_stats": stats}])
    return df


def _build_phase2_prompt(df_preview_csv: str, model_name: str) -> str:
    return (
        "You are a senior data engineer and scientific visualization specialist.\n"
        "Generate Python code ONLY. No markdown. No explanations.\n"
        "The code MUST define exactly two functions with these signatures:\n\n"
        "def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    ...\n\n"
        "def make_visual_report(df: pd.DataFrame, out_png_path: str) -> None:\n"
        "    ...\n\n"
        "Cleaning requirements:\n"
        "- Handle null values deterministically (no random imputation).\n"
        "- Standardize column names into academic nomenclature: snake_case, no spaces, no symbols.\n"
        "- Where appropriate, convert numeric-looking strings to numeric dtypes.\n"
        "- Return a new cleaned DataFrame.\n\n"
        "Visualization requirements:\n"
        "- Use seaborn and matplotlib.\n"
        "- Call sns.set_theme(style='whitegrid').\n"
        "- Create at least one high-fidelity plot appropriate for the available columns.\n"
        "- Ensure all axes are labeled professionally and title is descriptive.\n"
        "- Save a high-DPI PNG to out_png_path (dpi>=300).\n\n"
        "Safety constraints:\n"
        "- Do NOT import anything.\n"
        "- Do NOT read/write any files except saving the plot to out_png_path.\n"
        "- Do NOT use os, sys, subprocess, pathlib, requests.\n"
        "- Do NOT call open(), exec(), eval().\n\n"
        f"Model context: {model_name}\n"
        "Here is a CSV preview of the current DataFrame (first rows):\n"
        f"{df_preview_csv}\n"
    )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Phase 2 Labeler & Artist: clean + seaborn report via local Ollama.")
    parser.add_argument(
        "phase1_json",
        help="Path to Phase 1 extracted_metadata.json",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE2_MODEL", "qwen2.5-coder:3b"),
        help="Ollama model for Phase 2 (default: env OLLAMA_PHASE2_MODEL or qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL (default: env OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--out-dir",
        default="runs",
        help="Output directory (default: runs)",
    )
    args = parser.parse_args()

    phase1_path = Path(args.phase1_json)
    if not phase1_path.exists():
        raise FileNotFoundError(str(phase1_path))

    raw_obj = _read_json(phase1_path)
    meta = Phase1Metadata.model_validate(raw_obj)

    df = _metadata_to_dataframe(meta)
    df_preview_csv = df.head(8).to_csv(index=False)

    run_ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("backend") / args.out_dir / f"{run_ts}_{_safe_slug(meta.primary_topic)}"
    _ensure_dir(run_dir)

    _write_json(run_dir / "phase1_input.json", raw_obj)
    _write_text(run_dir / "df_preview.csv", df_preview_csv)

    prompt = _build_phase2_prompt(df_preview_csv=df_preview_csv, model_name=args.model)
    code = _ollama_generate(model=args.model, prompt=prompt, host=args.ollama_host)
    _write_text(run_dir / "llm_generated_phase2.py", code)

    clean_fn, viz_fn = _compile_generated_functions(code)

    cleaned = clean_fn(df.copy())
    if not isinstance(cleaned, pd.DataFrame):
        raise RuntimeError("clean_dataframe(df) did not return a pandas DataFrame.")

    cleaned_csv_path = run_dir / "cleaned_data.csv"
    cleaned.to_csv(cleaned_csv_path, index=False)

    report_png_path = run_dir / "visual_report.png"
    # Ensure theme baseline even if generated code forgets; generated code may still set it too.
    sns.set_theme(style="whitegrid")
    viz_fn(cleaned, str(report_png_path))

    summary = {
        "cleaned_data_csv": str(cleaned_csv_path),
        "visual_report_png": str(report_png_path),
        "rows": int(cleaned.shape[0]),
        "columns": list(map(str, cleaned.columns)),
    }
    _write_json(run_dir / "phase2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValidationError as e:
        raise SystemExit(f"Invalid Phase 1 JSON: {e}")
