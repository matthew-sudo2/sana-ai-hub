"""
Visualization Agent — Phase 4
Receives enriched_data.csv + analysis_result.json (from Analysis Agent)
and produces a suite of publication-ready charts auto-selected by data type.

Responsibilities:
  1. Chart-type selection  — bar, line, scatter, histogram, heatmap, box, pie
                             chosen automatically from column dtypes + correlations
  2. Plot generation       — matplotlib / seaborn, high-DPI PNG per chart
  3. Explanation JSON      — deterministic plain-English explanations per chart
                             derived from analysis metadata (no LLM required)
  4. Optional LLM pass     — Ollama writes bespoke chart code for edge cases
  5. Output                — individual PNGs + viz_summary.json + chart_explanations.json
                             consumed by Validation Agent and frontend UI
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
import warnings
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd

# ← FIX: Set non-interactive matplotlib backend FIRST to avoid tkinter threading errors
# This must happen before any plotting occurs
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, thread-safe)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Chart manifest schema  (consumed by Validation Agent + frontend)
# ---------------------------------------------------------------------------

class ChartRecord(BaseModel):
    chart_id: str
    chart_type: str        # "histogram" | "bar" | "line" | "scatter" | "heatmap" | "box" | "pie"
    title: str
    columns_used: list[str]
    png_path: str
    generated_by: str      # "builtin" | "llm"


class VizSummary(BaseModel):
    source: str
    source_type: str
    num_rows: int
    num_columns: int
    charts: list[ChartRecord]
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:60] or "col"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seaborn theme
# ---------------------------------------------------------------------------

def _apply_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


# ---------------------------------------------------------------------------
# Built-in chart generators  (no LLM needed)
# ---------------------------------------------------------------------------

def _plot_histogram(df: pd.DataFrame, col: str, out: Path, outlier_rows: list[int]) -> ChartRecord:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df[col].dropna()

    sns.histplot(data, bins="auto", kde=True, ax=ax, color="#4C72B0", edgecolor="white")

    # Overlay outlier rug if any
    if outlier_rows:
        outlier_vals = df.loc[df.index.isin(outlier_rows), col].dropna()
        if not outlier_vals.empty:
            sns.rugplot(outlier_vals, ax=ax, color="#C44E52", height=0.07,
                        label=f"Outliers (n={len(outlier_vals)})")
            ax.legend(fontsize=9)

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    _save_fig(fig, out)
    return ChartRecord(chart_id=f"hist_{_safe_slug(col)}", chart_type="histogram",
                       title=f"Distribution of {col}", columns_used=[col],
                       png_path=str(out), generated_by="builtin")


def _plot_bar(df: pd.DataFrame, col: str, out: Path, top_n: int = 15) -> ChartRecord:
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.6), 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                palette="Blues_d", edgecolor="white")
    ax.set_title(f"Top {top_n} Values — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out)
    return ChartRecord(chart_id=f"bar_{_safe_slug(col)}", chart_type="bar",
                       title=f"Top Values — {col}", columns_used=[col],
                       png_path=str(out), generated_by="builtin")


def _plot_pie(df: pd.DataFrame, col: str, out: Path, top_n: int = 8) -> ChartRecord:
    counts = df[col].value_counts().head(top_n)
    other = df[col].value_counts().iloc[top_n:].sum()
    if other > 0:
        counts = pd.concat([counts, pd.Series({"Other": other})])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%",
           startangle=140, colors=sns.color_palette("pastel", len(counts)))
    ax.set_title(f"Proportion — {col}")
    _save_fig(fig, out)
    return ChartRecord(chart_id=f"pie_{_safe_slug(col)}", chart_type="pie",
                       title=f"Proportion — {col}", columns_used=[col],
                       png_path=str(out), generated_by="builtin")


def _plot_line(df: pd.DataFrame, col: str, out: Path) -> ChartRecord:
    fig, ax = plt.subplots(figsize=(10, 4))
    series = df[col].dropna().reset_index(drop=True)
    ax.plot(series.index, series.values, linewidth=1.2, color="#4C72B0")
    ax.fill_between(series.index, series.values, alpha=0.12, color="#4C72B0")
    ax.set_title(f"Trend — {col}")
    ax.set_xlabel("Row index")
    ax.set_ylabel(col)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.2f}"))
    _save_fig(fig, out)
    return ChartRecord(chart_id=f"line_{_safe_slug(col)}", chart_type="line",
                       title=f"Trend — {col}", columns_used=[col],
                       png_path=str(out), generated_by="builtin")


def _plot_box(df: pd.DataFrame, cols: list[str], out: Path) -> ChartRecord:
    data = df[cols].dropna(how="all")
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 3 * n), 5), squeeze=False)
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col].dropna(), ax=axes[0][i], color="#4C72B0",
                    flierprops={"markerfacecolor": "#C44E52", "markersize": 4})
        axes[0][i].set_title(col, fontsize=10)
        axes[0][i].set_xlabel("")
    fig.suptitle("Box Plots — Numeric Columns", fontweight="bold")
    if n == 1:
        fig.set_figwidth(6)
    plt.tight_layout()
    _save_fig(fig, out)
    return ChartRecord(chart_id="boxplot_numeric", chart_type="box",
                       title="Box Plots — Numeric Columns", columns_used=cols,
                       png_path=str(out), generated_by="builtin")


def _plot_scatter(df: pd.DataFrame, col_a: str, col_b: str, out: Path,
                  outlier_rows: list[int]) -> ChartRecord:
    fig, ax = plt.subplots(figsize=(7, 6))
    normal = df[~df.index.isin(outlier_rows)]
    outliers = df[df.index.isin(outlier_rows)]

    ax.scatter(normal[col_a], normal[col_b], alpha=0.55, s=20,
               color="#4C72B0", label="Normal", edgecolors="none")
    if not outliers.empty:
        ax.scatter(outliers[col_a], outliers[col_b], alpha=0.8, s=30,
                   color="#C44E52", label="Outlier", edgecolors="none")
        ax.legend(fontsize=9)

    # Regression line
    try:
        valid = df[[col_a, col_b]].dropna()
        if len(valid) >= 2:
            m, b = np.polyfit(valid[col_a], valid[col_b], 1)
            x_line = np.linspace(float(df[col_a].min()), float(df[col_a].max()), 200)
            ax.plot(x_line, m * x_line + b, color="#DD8452", linewidth=1.4,
                    linestyle="--", label="Linear fit")
    except Exception:
        pass

    ax.set_title(f"Scatter: {col_a} vs {col_b}")
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)
    _save_fig(fig, out)
    return ChartRecord(chart_id=f"scatter_{_safe_slug(col_a)}_{_safe_slug(col_b)}",
                       chart_type="scatter",
                       title=f"Scatter: {col_a} vs {col_b}",
                       columns_used=[col_a, col_b],
                       png_path=str(out), generated_by="builtin")


def _plot_heatmap(df: pd.DataFrame, out: Path) -> ChartRecord | None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, num.shape[1]), max(5, num.shape[1] - 1)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5,
                annot_kws={"size": max(7, 10 - num.shape[1])})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    _save_fig(fig, out)
    return ChartRecord(chart_id="correlation_heatmap", chart_type="heatmap",
                       title="Correlation Heatmap",
                       columns_used=list(num.columns),
                       png_path=str(out), generated_by="builtin")


# ---------------------------------------------------------------------------
# Chart-type selection logic
# ---------------------------------------------------------------------------

def _all_outlier_rows(analysis: dict[str, Any]) -> list[int]:
    rows: set[int] = set()
    for anomaly in analysis.get("anomalies", []):
        rows.update(anomaly.get("row_indices", []))
    if not rows:
        print("[artist] Warning: No outlier rows found in analysis; red markers will not appear.")
    return list(rows)


def _select_charts(
    df: pd.DataFrame,
    analysis: dict[str, Any],
    run_dir: Path,
) -> list[ChartRecord]:
    """
    Decide which charts to produce based on column dtypes, cardinality,
    trend info, and top correlations from the Analysis Agent's output.
    Returns the list of ChartRecord objects created.
    """
    _apply_theme()
    charts: list[ChartRecord] = []
    outlier_rows = _all_outlier_rows(analysis)

    numeric_cols  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                     and c != "_is_outlier"]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols      = [c for c in df.columns
                     if not pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_datetime64_any_dtype(df[c])
                     and c != "_is_outlier"]

    col_stats_map: dict[str, dict] = {
        s["column"]: s for s in analysis.get("column_stats", [])
    }

    # ── 1. Histogram for each numeric column ─────────────────────────────
    for col in numeric_cols[:6]:   # cap at 6 to avoid sprawl
        out = run_dir / f"hist_{_safe_slug(col)}.png"
        charts.append(_plot_histogram(df, col, out, outlier_rows))

    # ── 2. Trend line for numeric cols flagged as increasing/decreasing ──
    trend_cols = [
        col for col in numeric_cols
        if col_stats_map.get(col, {}).get("trend") in ("increasing", "decreasing")
    ]
    for col in trend_cols[:3]:
        out = run_dir / f"line_{_safe_slug(col)}.png"
        charts.append(_plot_line(df, col, out))

    # ── 3. Scatter for top correlated pairs ──────────────────────────────
    top_corr = [
        c for c in analysis.get("correlations", [])
        if abs(c.get("pearson_r", 0)) >= 0.4
    ][:4]
    for pair in top_corr:
        a, b = pair["col_a"], pair["col_b"]
        if a in df.columns and b in df.columns:
            out = run_dir / f"scatter_{_safe_slug(a)}_{_safe_slug(b)}.png"
            charts.append(_plot_scatter(df, a, b, out, outlier_rows))

    # ── 4. Correlation heatmap (when ≥ 2 numeric cols) ───────────────────
    if len(numeric_cols) >= 2:
        out = run_dir / "correlation_heatmap.png"
        record = _plot_heatmap(df, out)
        if record:
            charts.append(record)

    # ── 5. Box plots for numeric columns ─────────────────────────────────
    if numeric_cols:
        out = run_dir / "boxplot_numeric.png"
        charts.append(_plot_box(df, numeric_cols[:8], out))

    # ── 6. Categorical columns ───────────────────────────────────────────
    for col in cat_cols[:4]:
        unique = df[col].nunique(dropna=True)
        if unique == 0:
            continue
        if unique <= 8:
            # Few categories → pie chart
            out = run_dir / f"pie_{_safe_slug(col)}.png"
            charts.append(_plot_pie(df, col, out))
        elif unique <= 30:
            # Medium cardinality → bar chart
            out = run_dir / f"bar_{_safe_slug(col)}.png"
            charts.append(_plot_bar(df, col, out))
        # High-cardinality categoricals skipped (not informative)

    # ── 7. Datetime columns → line chart using first numeric col ─────────
    for dt_col in datetime_cols[:2]:
        for num_col in numeric_cols[:2]:
            try:
                tmp = df[[dt_col, num_col]].dropna().sort_values(dt_col)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(tmp[dt_col], tmp[num_col], linewidth=1.2, color="#4C72B0")
                ax.set_title(f"{num_col} over time ({dt_col})")
                ax.set_xlabel(dt_col)
                ax.set_ylabel(num_col)
                fig.autofmt_xdate()
                out = run_dir / f"timeseries_{_safe_slug(dt_col)}_{_safe_slug(num_col)}.png"
                _save_fig(fig, out)
                charts.append(ChartRecord(
                    chart_id=f"ts_{_safe_slug(dt_col)}_{_safe_slug(num_col)}",
                    chart_type="line",
                    title=f"{num_col} over {dt_col}",
                    columns_used=[dt_col, num_col],
                    png_path=str(out),
                    generated_by="builtin",
                ))
            except Exception as e:
                print(f"[artist] Warning: Failed to create timeseries chart for {dt_col} vs {num_col}: {e}")
                continue

    # ── CRITICAL FIX: return the charts list ─────────────────────────────
    return charts


# ---------------------------------------------------------------------------
# Chart Explanation Generator  (deterministic — no LLM)
# ---------------------------------------------------------------------------

def _generate_chart_explanations(
    charts: list[ChartRecord],
    analysis: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """
    Generate plain-English explanations for every chart by combining:
      - chart metadata (type, columns)
      - column statistics from the Analysis Agent (mean, std, outliers, trend …)
      - correlation pairs from the Analysis Agent
      - key insights already identified

    Writes chart_explanations.json to run_dir.
    Returns the explanations dict.
    """
    # Build lookup maps from analysis JSON
    col_stats_map: dict[str, dict] = {
        s["column"]: s
        for s in analysis.get("column_stats", [])
        if isinstance(s, dict) and "column" in s
    }

    # Bidirectional correlation lookup: (col_a, col_b) → corr_record
    corr_map: dict[tuple[str, str], dict] = {}
    for c in analysis.get("correlations", []):
        if isinstance(c, dict) and "col_a" in c and "col_b" in c:
            corr_map[(c["col_a"], c["col_b"])] = c
            corr_map[(c["col_b"], c["col_a"])] = c

    key_insights: list[str] = analysis.get("key_insights", [])

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fmt(v: float | None, decimals: int = 2) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{v:,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    def _explain_chart(chart: ChartRecord) -> str:
        ct = chart.chart_type
        cols = chart.columns_used

        # ── Histogram ────────────────────────────────────────────────────
        if ct == "histogram" and cols:
            col = cols[0]
            s = col_stats_map.get(col, {})
            parts: list[str] = [f"This histogram shows the frequency distribution of '{col}'."]
            if s.get("mean") is not None:
                parts.append(
                    f"The average value is {_fmt(s['mean'])} "
                    f"with a standard deviation of {_fmt(s.get('std'))}."
                )
            if s.get("min") is not None and s.get("max") is not None:
                parts.append(f"Values range from {_fmt(s['min'])} to {_fmt(s['max'])}.")
            trend = s.get("trend")
            if trend and trend != "flat":
                parts.append(f"The column shows a {trend} trend across the dataset.")
            oc = s.get("outlier_count", 0) or 0
            if oc:
                parts.append(
                    f"{oc} outlier(s) detected via IQR method (shown as a red rug at the bottom)."
                )
            mp = s.get("missing_pct", 0) or 0
            if mp > 5:
                parts.append(f"Note: {mp:.1f}% of values were missing before cleaning.")
            return " ".join(parts)

        # ── Scatter ──────────────────────────────────────────────────────
        if ct == "scatter" and len(cols) >= 2:
            a, b = cols[0], cols[1]
            corr = corr_map.get((a, b)) or corr_map.get((b, a))
            parts = [f"This scatter plot reveals the relationship between '{a}' and '{b}'."]
            if corr:
                r = corr.get("pearson_r", 0)
                strength = corr.get("strength", "")
                direction = corr.get("direction", "")
                parts.append(
                    f"Pearson r = {r:+.3f}, indicating a {strength} {direction} linear correlation."
                )
                if abs(r) >= 0.7:
                    parts.append(
                        "The dashed orange line shows the linear regression fit — "
                        "this is a strong predictor relationship worth investigating further."
                    )
                elif abs(r) >= 0.4:
                    parts.append(
                        "A moderate association exists; other factors likely also influence this relationship."
                    )
            else:
                parts.append(
                    "A linear regression line is overlaid to indicate the general trend direction. "
                    "Red points are IQR-detected outliers."
                )
            return " ".join(parts)

        # ── Heatmap ──────────────────────────────────────────────────────
        if ct == "heatmap":
            num_cols = len(cols)
            strong_pairs = [
                f"'{c['col_a']}' ↔ '{c['col_b']}' (r={c['pearson_r']:+.2f})"
                for c in analysis.get("correlations", [])
                if c.get("strength") == "strong"
            ][:3]
            parts = [
                f"This correlation heatmap shows pairwise Pearson coefficients across "
                f"{num_cols} numeric column(s).",
                "Warm reds indicate strong positive correlation; cool blues indicate negative correlation.",
                "Values near zero (white) suggest no linear relationship.",
            ]
            if strong_pairs:
                parts.append(f"Strongest pairs: {', '.join(strong_pairs)}.")
            return " ".join(parts)

        # ── Bar ──────────────────────────────────────────────────────────
        if ct == "bar" and cols:
            col = cols[0]
            ci = next(
                (x for x in analysis.get("categorical_insights", []) if x.get("column") == col),
                {},
            )
            unique = ci.get("unique_count", "?")
            parts = [
                f"This bar chart shows the frequency of the most common values in '{col}' "
                f"({unique} unique value(s) total).",
                "Taller bars represent more frequently occurring categories.",
            ]
            top = (ci.get("top_values") or [{}])[:1]
            if top and top[0].get("value"):
                parts.append(
                    f"The most common value is '{top[0]['value']}' "
                    f"({top[0].get('pct', '?')}% of records)."
                )
            return " ".join(parts)

        # ── Pie ──────────────────────────────────────────────────────────
        if ct == "pie" and cols:
            col = cols[0]
            ci = next(
                (x for x in analysis.get("categorical_insights", []) if x.get("column") == col),
                {},
            )
            parts = [
                f"This pie chart shows the proportional breakdown of '{col}'.",
                "Each slice represents a category's share of the total record count.",
            ]
            top = (ci.get("top_values") or [{}])[:1]
            if top and top[0].get("value"):
                parts.append(
                    f"The dominant category is '{top[0]['value']}' at {top[0].get('pct', '?')}%."
                )
            return " ".join(parts)

        # ── Line / timeseries ────────────────────────────────────────────
        if ct == "line" and cols:
            col = cols[0]
            s = col_stats_map.get(col, {})
            trend = s.get("trend") or "stable"
            parts = [
                f"This line chart tracks '{col}' across dataset rows, "
                f"showing a {trend} trend.",
            ]
            if s.get("mean") is not None:
                parts.append(
                    f"The mean is {_fmt(s['mean'])} (range: {_fmt(s.get('min'))} – {_fmt(s.get('max'))})."
                )
            if len(cols) >= 2:
                parts[0] = (
                    f"This time-series line chart plots '{cols[1]}' over '{cols[0]}', "
                    f"revealing a {trend} trend."
                )
            return " ".join(parts)

        # ── Box ──────────────────────────────────────────────────────────
        if ct == "box":
            outlier_cols = [
                f"'{a.get('column')}' ({a.get('outlier_count')})"
                for a in analysis.get("anomalies", [])
                if a.get("column") in cols and a.get("outlier_count", 0) > 0
            ]
            parts = [
                f"This box plot summarises the spread of {len(cols)} numeric column(s).",
                "Each box spans the interquartile range (Q1–Q3); the line marks the median.",
                "Red diamond markers indicate IQR-detected outliers.",
            ]
            if outlier_cols:
                parts.append(f"Columns with notable outliers: {', '.join(outlier_cols)}.")
            return " ".join(parts)

        # ── Fallback ─────────────────────────────────────────────────────
        col_list = ", ".join(f"'{c}'" for c in cols) if cols else "the dataset"
        return f"This {ct} chart visualises {col_list}."

    def _stats_for_chart(chart: ChartRecord) -> dict[str, Any]:
        """Extract relevant numeric stats for the primary column of a chart."""
        if not chart.columns_used:
            return {}
        col = chart.columns_used[0]
        s = col_stats_map.get(col, {})
        wanted = ["mean", "median", "std", "min", "max", "outlier_count", "trend",
                  "missing_count", "missing_pct"]
        return {k: s[k] for k in wanted if k in s}

    # ── Build explanations list ───────────────────────────────────────────
    explanations: list[dict[str, Any]] = []
    for chart in charts:
        explanations.append({
            "filename": Path(chart.png_path).name,
            "chart_id": chart.chart_id,
            "chart_type": chart.chart_type,
            "columns_used": chart.columns_used,
            "title": chart.title,
            "generated_by": chart.generated_by,
            "explanation": _explain_chart(chart),
            "stats": _stats_for_chart(chart),
        })

    result: dict[str, Any] = {
        "timestamp": dt.datetime.now(timezone.utc).isoformat(),
        "total_charts": len(charts),
        "key_insights": key_insights,
        "explanations": explanations,
    }

    _write_json(run_dir / "chart_explanations.json", result)
    print(f"[artist] chart_explanations.json written ({len(explanations)} entries)", flush=True)
    return result


# ---------------------------------------------------------------------------
# Optional LLM chart (edge-case bespoke plot)
# ---------------------------------------------------------------------------

_DISALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
    ast.Try, ast.Raise, ast.Lambda, ast.Global, ast.Nonlocal,
)
_DISALLOWED_CALL_NAMES = {
    "open", "exec", "eval", "__import__", "compile", "input", "globals", "locals",
}


class UnsafeGeneratedCodeError(RuntimeError):
    pass


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
        resp = client.generate(model=model, prompt=prompt, stream=False,
                               options={"temperature": 0.1}, keep_alive=0)
        text = str(resp.get("response", "")).strip()
        if "```" in text:
            m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        return text
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


def _strip_unsafe_lines(code: str) -> str:
    lines = code.split("\n")
    filtered, skip_indent = [], None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            continue
        if stripped.startswith("try:"):
            skip_indent = len(line) - len(line.lstrip())
            continue
        if skip_indent is not None:
            cur = len(line) - len(line.lstrip()) if line.strip() else skip_indent + 1
            if stripped.startswith(("except", "finally")):
                continue
            if line.strip() and cur <= skip_indent:
                skip_indent = None
        if stripped == "pass" and skip_indent is not None:
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _validate_llm_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_AST_NODES):
            raise UnsafeGeneratedCodeError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
                raise UnsafeGeneratedCodeError(f"Disallowed call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"os", "sys", "subprocess", "pathlib", "shutil"}:
                    raise UnsafeGeneratedCodeError(f"Disallowed module: {node.func.value.id}")
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "make_bespoke_chart" not in fn_names:
        raise UnsafeGeneratedCodeError("Missing required function: make_bespoke_chart")


def _run_llm_chart(
    df: pd.DataFrame,
    analysis: dict[str, Any],
    run_dir: Path,
    model: str,
    host: str,
) -> ChartRecord | None:
    """Ask Ollama to generate one bespoke chart based on key_insights."""
    insights = analysis.get("key_insights", [])
    if not insights:
        return None

    prompt = (
        "You are a scientific visualization agent.\n"
        "Generate Python code ONLY. No markdown. No explanations.\n"
        "Define exactly one function:\n\n"
        "def make_bespoke_chart(df: pd.DataFrame, out_png_path: str) -> None:\n"
        "    ...\n\n"
        "Requirements:\n"
        "- Use seaborn and matplotlib.\n"
        "- sns.set_theme(style='whitegrid') must be called first.\n"
        "- Create ONE chart that best illustrates the most important insight below.\n"
        "- Professional labels, title, tight_layout(), savefig(out_png_path, dpi=300), plt.close().\n\n"
        "Safety constraints:\n"
        "- Do NOT import anything. pd, np, sns, plt are already available.\n"
        "- Do NOT use try/except, with, raise, lambda, import.\n"
        "- Do NOT use os, sys, subprocess, open, exec, eval.\n\n"
        f"Key insights from Analysis Agent:\n{json.dumps(insights, ensure_ascii=False)}\n\n"
        f"df.columns: {list(df.columns)}\n"
        f"CSV preview:\n{df.head(6).to_csv(index=False)}\n"
    )

    raw = _ollama_generate(model=model, prompt=prompt, host=host)
    if not raw:
        return None

    try:
        code = _strip_unsafe_lines(raw)
        _write_text(run_dir / "llm_generated_bespoke.py", code)
        _validate_llm_code(code)

        safe_builtins = {
            k: v for k, v in {
                "None": None, "True": True, "False": False,
                "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
                "sorted": sorted, "range": range, "enumerate": enumerate,
                "zip": zip, "float": float, "int": int, "str": str,
                "dict": dict, "list": list, "set": set, "tuple": tuple,
                "isinstance": isinstance, "any": any, "all": all,
            }.items()
        }
        g: dict[str, Any] = {"__builtins__": safe_builtins,
                              "pd": pd, "np": np, "sns": sns, "plt": plt}
        loc: dict[str, Any] = {}
        exec(compile(code, "<llm_bespoke>", "exec"), g, loc)

        fn = loc.get("make_bespoke_chart")
        if not callable(fn):
            return None

        out = run_dir / "bespoke_llm_chart.png"
        _apply_theme()
        fn(df.copy(), str(out))

        return ChartRecord(
            chart_id="bespoke_llm",
            chart_type="bespoke",
            title="LLM-Selected Insight Chart",
            columns_used=list(df.columns),
            png_path=str(out),
            generated_by="llm",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core runner  (importable by graph.py / Validation Agent)
# ---------------------------------------------------------------------------

def run_visualization(
    enriched_csv_path: Path,
    *,
    analysis_json_path: Path | None = None,
    out_dir: Path | None = None,
    model: str = "qwen2.5-coder:3b",
    ollama_host: str = "http://127.0.0.1:11434",
    use_llm: bool = False,
) -> dict[str, Any]:
    """
    Main visualization pipeline. Returns the viz_summary dict.

    Pipeline:
      1. Generate charts via _select_charts (builtin, deterministic)
      2. Extract per-chart metadata + column stats from analysis
      3. Write chart_explanations.json (used by API + frontend)
      4. Optionally generate one LLM bespoke chart
      5. Write viz_summary.json

    Parameters
    ----------
    enriched_csv_path  : enriched_data.csv from Analysis Agent
    analysis_json_path : analysis_result.json from Analysis Agent
                         (auto-discovered from same dir if omitted)
    out_dir            : where to write PNGs; defaults to CSV's parent dir
    model              : Ollama model for optional bespoke chart
    ollama_host        : Ollama base URL
    use_llm            : generate one extra LLM-authored chart
    """
    df = pd.read_csv(enriched_csv_path)
    if df.empty:
        raise RuntimeError("Enriched CSV is empty — nothing to visualise.")

    run_dir = out_dir or enriched_csv_path.parent
    _ensure_dir(run_dir)

    # Auto-discover analysis_result.json
    if analysis_json_path is None:
        candidate = enriched_csv_path.parent / "analysis_result.json"
        analysis_json_path = candidate if candidate.exists() else None

    analysis: dict[str, Any] = {}
    if analysis_json_path and analysis_json_path.exists():
        analysis = _read_json(analysis_json_path)

    # Pull source metadata from analysis or phase3_summary
    source = analysis.get("source", str(enriched_csv_path))
    source_type = analysis.get("source_type", "csv")
    if not source:
        phase3 = enriched_csv_path.parent / "phase3_summary.json"
        if phase3.exists():
            p3 = _read_json(phase3)
            source = p3.get("source", str(enriched_csv_path))
            source_type = p3.get("source_type", "csv")

    # ── Step 1: Generate built-in charts ─────────────────────────────────
    print("[artist] Generating charts...", flush=True)
    charts = _select_charts(df, analysis, run_dir)
    print(f"[artist] {len(charts)} chart(s) generated.", flush=True)

    # ── Step 2: Optional LLM bespoke chart ───────────────────────────────
    if use_llm:
        bespoke = _run_llm_chart(df, analysis, run_dir, model=model, host=ollama_host)
        if bespoke:
            charts.append(bespoke)
        else:
            print("[artist] Warning: LLM bespoke chart generation failed or returned nothing.")

    # ── Step 3: Extract metadata + write chart_explanations.json ─────────
    print("[artist] Generating chart explanations...", flush=True)
    _generate_chart_explanations(charts, analysis, run_dir)

    # ── Step 4: Write viz_summary.json ───────────────────────────────────
    summary = VizSummary(
        source=source,
        source_type=source_type,
        num_rows=len(df),
        num_columns=len(df.columns),
        charts=charts,
    )

    summary_dict = summary.model_dump()
    _write_json(run_dir / "viz_summary.json", summary_dict)
    return summary_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(
        description="Visualization Agent (Phase 4): auto-select and render charts."
    )
    parser.add_argument(
        "enriched_csv",
        help="Path to enriched_data.csv from the Analysis Agent.",
    )
    parser.add_argument(
        "--analysis-json", default=None,
        help="Path to analysis_result.json (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE4_MODEL", "qwen2.5-coder:3b"),
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Generate one extra bespoke chart via Ollama.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.enriched_csv)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    analysis_path = Path(args.analysis_json) if args.analysis_json else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    result = run_visualization(
        enriched_csv_path=csv_path,
        analysis_json_path=analysis_path,
        out_dir=out_dir,
        model=args.model,
        ollama_host=args.ollama_host,
        use_llm=args.use_llm,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)