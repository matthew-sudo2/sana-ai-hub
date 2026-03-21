"""
Data Analysis Agent — Phase 3
Receives cleaned_data.csv (from Labeler/Cleaning Agent) and produces
analytical results, metrics, and an enriched summary for the Visualization Agent.

Responsibilities:
  1. Basic statistics  — mean, median, std, quartiles per numeric column
  2. Pattern / trend   — monotonicity, rolling-mean slope, value-frequency ranks
  3. Correlation       — Pearson matrix + ranked pairs
  4. Anomaly detection — IQR-based outlier flagging per numeric column
  5. Categorical insight — top-N value counts per categorical column
  6. Output            — analysis_result.json  (machine-readable, feeds Viz Agent)
                         analysis_report.md    (human-readable summary)
                         enriched_data.csv     (original rows + outlier flag column)

Fixes applied vs previous version:
  - _is_outlier column filtered out before analysis (prevents bleed-in on re-runs)
  - _iqr_outliers guards against all-NaN columns (was crashing on empty series)
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Output schemas  (consumed by Visualization Agent downstream)
# ---------------------------------------------------------------------------

CorrelationStrength = Literal["weak", "moderate", "strong"]
CorrelationDirection = Literal["positive", "negative"]


class ColumnStats(BaseModel):
    column: str
    dtype: str
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    q3: float | None = None
    missing_count: int = 0
    missing_pct: float = 0.0
    outlier_count: int = 0          # IQR-based
    trend: str | None = None        # "increasing" | "decreasing" | "flat" | None


class CorrelationPair(BaseModel):
    col_a: str
    col_b: str
    pearson_r: float
    strength: CorrelationStrength
    direction: CorrelationDirection


class CategoricalInsight(BaseModel):
    column: str
    unique_count: int
    top_values: list[dict[str, Any]]   # [{"value": x, "count": n, "pct": p}, …]


class Anomaly(BaseModel):
    column: str
    row_indices: list[int]
    lower_fence: float
    upper_fence: float
    outlier_count: int


class AnalysisResult(BaseModel):
    source: str
    source_type: str
    num_rows: int
    num_columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    column_stats: list[ColumnStats]
    correlations: list[CorrelationPair]
    categorical_insights: list[CategoricalInsight]
    anomalies: list[Anomaly]
    key_insights: list[str]         # bullet-ready plain-English findings
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:80] or "run"


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal columns that must never be analysed
# ---------------------------------------------------------------------------

_INTERNAL_COLS = {"_is_outlier"}


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pipeline-internal columns before analysis to prevent bleed-in."""
    to_drop = [c for c in df.columns if c in _INTERNAL_COLS]
    if to_drop:
        print(f"[analyst] Dropping internal columns before analysis: {to_drop}", flush=True)
        df = df.drop(columns=to_drop)
    return df


def _sanitize_for_markdown(value: str) -> str:
    """Escape special markdown characters for safe table rendering."""
    if value is None:
        return ""
    value = str(value)
    # Escape pipe characters for markdown tables
    value = value.replace("|", "•")
    # Escape leading characters that might cause markdown confusion
    if value.startswith(("#", "-", "*", ">")):
        value = " " + value
    return value


# ---------------------------------------------------------------------------
# Ollama helper (optional LLM insight enrichment)
# ---------------------------------------------------------------------------

def _ollama_alive(host: str) -> bool:
    try:
        import requests
        return requests.get(f"{host}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _ollama_generate(model: str, prompt: str, host: str) -> str:
    if not _ollama_alive(host):
        return ""
    try:
        import ollama  # type: ignore
        client = ollama.Client(host=host)
        resp = client.generate(
            model=model, prompt=prompt, stream=False,
            options={"temperature": 0.1, "num_ctx": 4096}, keep_alive=0,
        )
        return str(resp.get("response", "")).strip()
    except Exception:
        try:
            completed = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, check=True,
            )
            return completed.stdout.decode(errors="replace").strip()
        except Exception:
            return ""
    finally:
        subprocess.run(["ollama", "stop", model],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _strength(r: float) -> CorrelationStrength:
    ar = abs(r)
    if ar >= 0.7:
        return "strong"
    if ar >= 0.4:
        return "moderate"
    return "weak"


def _direction(r: float) -> CorrelationDirection:
    return "positive" if r >= 0 else "negative"


def _trend(series: pd.Series) -> str | None:
    """Detect monotonic trend via Spearman rank correlation with index."""
    s = series.dropna().reset_index(drop=True)
    if len(s) < 5:
        return None
    try:
        from scipy.stats import spearmanr  # type: ignore
        r, p = spearmanr(range(len(s)), s)
        if p > 0.05:
            return "flat"
        if r > 0.3:
            return "increasing"
        if r < -0.3:
            return "decreasing"
        return "flat"
    except ImportError:
        # Fallback: simple first/last half mean comparison
        first_half  = float(s[: len(s) // 2].mean())
        second_half = float(s[len(s) // 2 :].mean())
        diff = second_half - first_half
        if abs(diff) < 0.05 * (abs(first_half) + 1e-9):
            return "flat"
        return "increasing" if diff > 0 else "decreasing"


def _iqr_outliers(series: pd.Series) -> tuple[list[int], float, float]:
    """
    Return (outlier_row_indices, lower_fence, upper_fence).
    Returns empty list and (0.0, 0.0) if the series has fewer than 4 non-NaN values
    or if series is boolean/object dtype.
    """
    # Guard: skip boolean and object dtypes
    if series.dtype == 'bool' or series.dtype == 'object':
        return [], 0.0, 0.0
    
    s = series.dropna()

    # Guard: need at least 4 values to compute meaningful quartiles
    if len(s) < 4:
        return [], 0.0, 0.0

    try:
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        mask = (series < lo) | (series > hi)
        return list(series[mask].index.astype(int)), lo, hi
    except (TypeError, ValueError):
        # Fallback for any remaining type errors
        return [], 0.0, 0.0


def _compute_column_stats(df: pd.DataFrame) -> list[ColumnStats]:
    stats: list[ColumnStats] = []
    for col in df.columns:
        series = df[col]
        missing     = int(series.isna().sum())
        missing_pct = round(missing / len(df) * 100, 2) if len(df) else 0.0

        # Check if numeric but exclude boolean dtypes
        is_numeric = pd.api.types.is_numeric_dtype(series) and series.dtype != 'bool'
        
        if is_numeric:
            s = series.dropna()
            outlier_idx, lo, hi = _iqr_outliers(series)
            stats.append(ColumnStats(
                column=col,
                dtype=str(series.dtype),
                mean=round(float(s.mean()), 6)             if len(s) else None,
                median=round(float(s.median()), 6)         if len(s) else None,
                std=round(float(s.std()), 6)               if len(s) else None,
                min=round(float(s.min()), 6)               if len(s) else None,
                max=round(float(s.max()), 6)               if len(s) else None,
                q1=round(float(s.quantile(0.25)), 6)       if len(s) else None,
                q3=round(float(s.quantile(0.75)), 6)       if len(s) else None,
                missing_count=missing,
                missing_pct=missing_pct,
                outlier_count=len(outlier_idx),
                trend=_trend(series),
            ))
        else:
            stats.append(ColumnStats(
                column=col,
                dtype=str(series.dtype),
                missing_count=missing,
                missing_pct=missing_pct,
            ))
    return stats


def _compute_correlations(df: pd.DataFrame) -> list[CorrelationPair]:
    # Select numeric columns but exclude booleans
    num = df.select_dtypes(include=[np.number])
    num = num.select_dtypes(exclude=['bool'])  # Exclude boolean columns
    if num.shape[1] < 2:
        return []

    corr = num.corr(numeric_only=True)
    pairs: list[CorrelationPair] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if np.isnan(r):
                continue
            pairs.append(CorrelationPair(
                col_a=cols[i],
                col_b=cols[j],
                pearson_r=round(r, 6),
                strength=_strength(r),
                direction=_direction(r),
            ))

    pairs.sort(key=lambda p: abs(p.pearson_r), reverse=True)
    return pairs[:50]


def _compute_categorical_insights(df: pd.DataFrame, top_n: int = 10) -> list[CategoricalInsight]:
    insights: list[CategoricalInsight] = []
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        vc    = df[col].value_counts(dropna=False)
        total = len(df)
        top   = [
            {"value": str(v), "count": int(c), "pct": round(c / total * 100, 2)}
            for v, c in vc.head(top_n).items()
        ]
        insights.append(CategoricalInsight(
            column=col,
            unique_count=int(df[col].nunique(dropna=True)),
            top_values=top,
        ))
    return insights


def _compute_anomalies(df: pd.DataFrame) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        idx, lo, hi = _iqr_outliers(df[col])
        if idx:
            anomalies.append(Anomaly(
                column=col,
                row_indices=idx[:100],
                lower_fence=round(lo, 6),
                upper_fence=round(hi, 6),
                outlier_count=len(idx),
            ))
    return anomalies


def _derive_key_insights(
    col_stats: list[ColumnStats],
    correlations: list[CorrelationPair],
    anomalies: list[Anomaly],
    df: pd.DataFrame,
) -> list[str]:
    """Generate plain-English bullet insights from computed metrics."""
    insights: list[str] = []

    # Missing data
    high_missing = [s for s in col_stats if s.missing_pct > 20]
    if high_missing:
        cols_str = ", ".join(f"`{s.column}` ({s.missing_pct:.1f}%)" for s in high_missing[:5])
        insights.append(f"High missingness detected in: {cols_str}.")

    # Strong correlations
    strong = [c for c in correlations if c.strength == "strong"]
    if strong:
        top = strong[0]
        insights.append(
            f"Strongest correlation: `{top.col_a}` ↔ `{top.col_b}` "
            f"(r={top.pearson_r:+.3f}, {top.direction})."
        )
    if len(strong) > 1:
        insights.append(f"{len(strong)} strong correlation pair(s) found — potential multicollinearity.")

    # Trends
    increasing = [s.column for s in col_stats if s.trend == "increasing"]
    decreasing = [s.column for s in col_stats if s.trend == "decreasing"]
    if increasing:
        insights.append(f"Upward trend detected in: {', '.join(f'`{c}`' for c in increasing[:5])}.")
    if decreasing:
        insights.append(f"Downward trend detected in: {', '.join(f'`{c}`' for c in decreasing[:5])}.")

    # Outliers
    heavy_outliers = sorted(anomalies, key=lambda a: a.outlier_count, reverse=True)
    if heavy_outliers:
        top_a = heavy_outliers[0]
        insights.append(
            f"Most outlier-heavy column: `{top_a.column}` "
            f"({top_a.outlier_count} outlier(s) outside "
            f"[{top_a.lower_fence:.3g}, {top_a.upper_fence:.3g}])."
        )

    # Skewness
    for s in col_stats:
        if s.mean is not None and s.median is not None and s.std and s.std > 0:
            skew = (s.mean - s.median) / s.std
            if abs(skew) > 0.5:
                direction = "right" if skew > 0 else "left"
                insights.append(
                    f"`{s.column}` appears {direction}-skewed "
                    f"(mean−median / std ≈ {skew:+.2f})."
                )

    if not insights:
        insights.append("No strong patterns detected — data appears uniformly distributed.")

    return insights


# ---------------------------------------------------------------------------
# Optional LLM enrichment
# ---------------------------------------------------------------------------

def _llm_enrich_insights(
    key_insights: list[str],
    describe_md: str,
    model: str,
    host: str,
) -> list[str]:
    if not _ollama_alive(host):
        return key_insights

    prompt = (
        "You are a data analyst. Based on the statistics and existing insights below, "
        "add 2-3 additional plain-English bullet-point insights. "
        "Return ONLY a JSON array of strings. No markdown, no preamble.\n\n"
        "Existing insights:\n"
        + json.dumps(key_insights, ensure_ascii=False)
        + "\n\ndf.describe() (numeric columns):\n"
        + describe_md
    )

    raw = _ollama_generate(model=model, prompt=prompt, host=host)
    if not raw:
        return key_insights

    try:
        start = raw.find("[")
        end   = raw.rfind("]")
        if start != -1 and end != -1:
            extras = json.loads(raw[start: end + 1])
            if isinstance(extras, list):
                return key_insights + [str(x) for x in extras if isinstance(x, str)]
    except Exception:
        pass

    return key_insights


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_markdown_report(result: AnalysisResult) -> str:
    lines: list[str] = [
        "# Data Analysis Report",
        "",
        f"**Source:** {result.source}  ",
        f"**Type:** {result.source_type}  ",
        f"**Rows:** {result.num_rows:,}  |  **Columns:** {result.num_columns}  ",
        f"**Generated:** {result.timestamp}",
        "",
        "---",
        "",
        "## Key Insights",
        "",
    ]
    for insight in result.key_insights:
        lines.append(f"- {insight}")

    lines += ["", "---", "", "## Numeric Column Statistics", ""]
    num_stats = [s for s in result.column_stats if s.mean is not None]
    if num_stats:
        lines.append("| Column | Mean | Median | Std | Min | Max | Outliers | Trend |")
        lines.append("|--------|------|--------|-----|-----|-----|----------|-------|")
        for s in num_stats:
            lines.append(
                f"| `{s.column}` | {s.mean:.4g} | {s.median:.4g} | {s.std:.4g} "
                f"| {s.min:.4g} | {s.max:.4g} | {s.outlier_count} | {s.trend or '—'} |"
            )
    else:
        lines.append("_No numeric columns found._")

    if result.correlations:
        lines += ["", "---", "", "## Top Correlations", ""]
        lines.append("| Column A | Column B | Pearson r | Strength | Direction |")
        lines.append("|----------|----------|-----------|----------|-----------|")
        for c in result.correlations[:15]:
            lines.append(
                f"| `{c.col_a}` | `{c.col_b}` | {c.pearson_r:+.4f} "
                f"| {c.strength} | {c.direction} |"
            )

    if result.anomalies:
        lines += ["", "---", "", "## Anomalies (IQR Method)", ""]
        lines.append("| Column | Outlier Count | Lower Fence | Upper Fence |")
        lines.append("|--------|--------------|-------------|-------------|")
        for a in sorted(result.anomalies, key=lambda x: x.outlier_count, reverse=True):
            lines.append(
                f"| `{a.column}` | {a.outlier_count} "
                f"| {a.lower_fence:.4g} | {a.upper_fence:.4g} |"
            )

    if result.categorical_insights:
        lines += ["", "---", "", "## Categorical Columns", ""]
        for ci in result.categorical_insights:
            lines.append(f"### `{ci.column}` ({ci.unique_count} unique values)")
            lines.append("| Value | Count | % |")
            lines.append("|-------|-------|---|")
            for tv in ci.top_values[:5]:
                sanitized_value = _sanitize_for_markdown(tv['value'])
                lines.append(f"| {sanitized_value} | {tv['count']} | {tv['pct']}% |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Enriched DataFrame builder
# ---------------------------------------------------------------------------

def _enrich_dataframe(df: pd.DataFrame, anomalies: list[Anomaly]) -> pd.DataFrame:
    """Add a boolean `_is_outlier` column flagging any IQR-outlier row."""
    outlier_rows: set[int] = set()
    for a in anomalies:
        outlier_rows.update(a.row_indices)
    df = df.copy()
    df["_is_outlier"] = df.index.isin(outlier_rows)
    return df


# ---------------------------------------------------------------------------
# Core analysis runner  (importable by graph.py / Streamlit)
# ---------------------------------------------------------------------------

def run_analysis(
    cleaned_csv_path: Path,
    *,
    out_dir: Path | None = None,
    model: str = "llama3.2:3b",
    ollama_host: str = "http://127.0.0.1:11434",
    use_llm: bool = False,
    source: str = "",
    source_type: str = "csv",
) -> dict[str, Any]:
    """
    Main analysis pipeline. Returns the phase3_summary dict.

    Parameters
    ----------
    cleaned_csv_path : path to cleaned_data.csv from the Labeler Agent
    out_dir          : where to write outputs; defaults to cleaned_csv_path's parent
    model            : Ollama model for optional insight enrichment
    ollama_host      : Ollama base URL
    use_llm          : whether to call Ollama for extra insight bullets
    source           : original data source label (forwarded from scout)
    source_type      : original source type (forwarded from scout)
    """
    df = pd.read_csv(cleaned_csv_path)
    if df.empty:
        raise RuntimeError("Cleaned CSV is empty — nothing to analyse.")

    print(f"[analyst] Loaded {len(df):,} rows × {len(df.columns)} columns from cleaned_data.csv", flush=True)

    # ── Drop pipeline-internal columns before any analysis ───────────────
    df = _drop_internal_cols(df)

    run_dir = out_dir or cleaned_csv_path.parent
    _ensure_dir(run_dir)

    # ── Pull source info from phase2_summary if not supplied ─────────────
    if not source:
        summary_path = cleaned_csv_path.parent / "phase2_summary.json"
        if summary_path.exists():
            try:
                phase2 = _read_json(summary_path)
                source      = phase2.get("source", str(cleaned_csv_path))
                source_type = phase2.get("source_type", "csv")
            except Exception:
                source = str(cleaned_csv_path)

    numeric_cols     = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    col_stats    = _compute_column_stats(df)
    correlations = _compute_correlations(df)
    cat_insights = _compute_categorical_insights(df)
    anomalies    = _compute_anomalies(df)
    key_insights = _derive_key_insights(col_stats, correlations, anomalies, df)

    # ── Optional LLM enrichment ──────────────────────────────────────────
    if use_llm:
        num = df.select_dtypes(include=[np.number])
        describe_md = num.describe().T.to_markdown(floatfmt=".4g") if not num.empty else ""
        key_insights = _llm_enrich_insights(
            key_insights, describe_md, model=model, host=ollama_host
        )

    result = AnalysisResult(
        source=source or str(cleaned_csv_path),
        source_type=source_type,
        num_rows=len(df),
        num_columns=len(df.columns),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        column_stats=col_stats,
        correlations=correlations,
        categorical_insights=cat_insights,
        anomalies=anomalies,
        key_insights=key_insights,
    )

    # ── Write outputs ────────────────────────────────────────────────────
    result_dict = result.model_dump()
    _write_json(run_dir / "analysis_result.json", result_dict)
    _write_text(run_dir / "analysis_report.md", _build_markdown_report(result))

    enriched      = _enrich_dataframe(df, anomalies)
    enriched_path = run_dir / "enriched_data.csv"
    enriched.to_csv(enriched_path, index=False)

    summary = {
        "source": result.source,
        "source_type": result.source_type,
        "num_rows": result.num_rows,
        "num_columns": result.num_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "correlation_pairs_found": len(correlations),
        "anomaly_columns": [a.column for a in anomalies],
        "key_insights": key_insights,
        "analysis_result_json": str(run_dir / "analysis_result.json"),
        "analysis_report_md": str(run_dir / "analysis_report.md"),
        "enriched_data_csv": str(enriched_path),
    }
    _write_json(run_dir / "phase3_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Data Analysis Agent (Phase 3): statistics, correlations, anomalies."
    )
    parser.add_argument(
        "cleaned_csv",
        help="Path to cleaned_data.csv produced by the Labeler Agent.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE3_MODEL", "llama3.2:3b"),
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Enable Ollama for additional plain-English insight bullets.",
    )
    parser.add_argument("--source", default="")
    parser.add_argument("--source-type", default="csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.cleaned_csv)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    out_dir = Path(args.out_dir) if args.out_dir else None

    summary = run_analysis(
        cleaned_csv_path=csv_path,
        out_dir=out_dir,
        model=args.model,
        ollama_host=args.ollama_host,
        use_llm=args.use_llm,
        source=args.source,
        source_type=args.source_type,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)