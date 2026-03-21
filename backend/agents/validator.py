from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field


Severity = Literal["info", "warning", "error"]
Status = Literal["APPROVED", "APPROVED_WITH_WARNINGS", "REJECTED"]


class CheckResult(BaseModel):
    check_id: str
    category: str
    description: str
    passed: bool
    severity: Severity
    detail: str = ""


class DimensionScore(BaseModel):
    dimension: str
    score: float
    weight: float
    checks_passed: int
    checks_total: int


class ValidationResult(BaseModel):
    status: Status
    overall_confidence: float
    dimension_scores: list[DimensionScore]
    checks: list[CheckResult]
    errors: list[str]
    warnings: list[str]
    infos: list[str]
    source: str
    source_type: str
    num_rows: int
    num_columns: int
    charts_validated: int
    timestamp: str = Field(default_factory=lambda: dt.datetime.now(timezone.utc).isoformat())


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_detail(detail: str) -> str:
    """Sanitize detail text for safe markdown rendering."""
    if not detail:
        return ""
    # Convert Windows paths to forward slashes for better markdown rendering
    detail = detail.replace("\\", "/")
    # Remove special markdown characters or escape them
    detail = detail.replace("|", "•")  # Replace pipes with bullets
    detail = detail.replace("[", "(").replace("]", ")")  # Replace brackets with parens
    # Clean up Python list/dict representations
    detail = detail.replace("'", "")  # Remove single quotes from list repr
    return detail.strip()


def _check(
    check_id: str,
    category: str,
    description: str,
    passed: bool,
    severity: Severity,
    detail: str = "",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        category=category,
        description=description,
        passed=passed,
        severity=severity,
        detail=_sanitize_detail(detail),
    )


def _check_completeness(run_dir: Path) -> list[CheckResult]:
    required: list[tuple[str, str, Severity]] = [
        ("scout_input.json", "Scout output JSON", "error"),
        ("cleaned_data.csv", "Labeler cleaned CSV", "error"),
        ("analysis_result.json", "Analysis result JSON", "error"),
        ("analysis_report.md", "Analysis markdown report", "warning"),
        ("enriched_data.csv", "Enriched CSV", "warning"),
        ("viz_summary.json", "Visualization summary JSON", "error"),
        ("phase2_summary.json", "Labeler summary", "warning"),
        ("phase3_summary.json", "Analysis summary", "warning"),
    ]

    checks: list[CheckResult] = []
    for filename, label, severity in required:
        path = run_dir / filename
        exists = path.exists() and path.stat().st_size > 0
        checks.append(
            _check(
                check_id=f"file_exists_{filename.replace('.', '_')}",
                category="completeness",
                description=f"{label} present and non-empty",
                passed=exists,
                severity=severity,
                detail=str(path) if not exists else "",
            )
        )
    return checks


def _check_sanity(df: pd.DataFrame, analysis: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    reported_rows = int(analysis.get("num_rows", 0) or 0)
    reported_cols = int(analysis.get("num_columns", 0) or 0)
    actual_rows = len(df)
    actual_cols = len(df.columns)

    checks.append(_check("row_count_match", "sanity", "Row count matches analysis", reported_rows == actual_rows, "error", f"analysis={reported_rows}, actual={actual_rows}" if reported_rows != actual_rows else ""))
    checks.append(_check("col_count_match", "sanity", "Column count matches analysis", reported_cols == actual_cols, "warning", f"analysis={reported_cols}, actual={actual_cols}" if reported_cols != actual_cols else ""))

    all_na_rows = int(df.isna().all(axis=1).sum())
    checks.append(_check("no_all_na_rows", "sanity", "No fully-empty rows", all_na_rows == 0, "warning", f"{all_na_rows} fully-empty rows" if all_na_rows else ""))

    all_na_cols = [c for c in df.columns if df[c].isna().all()]
    checks.append(_check("no_all_na_cols", "sanity", "No fully-empty columns", len(all_na_cols) == 0, "warning", f"{all_na_cols}" if all_na_cols else ""))

    dup = int(df.duplicated().sum())
    checks.append(_check("no_duplicates", "sanity", "No duplicate rows", dup == 0, "info", f"{dup} duplicate rows" if dup else ""))

    count_pattern = re.compile(r"(count|total|freq|num_|n_)", re.I)
    pct_pattern = re.compile(r"(pct|percent|rate|ratio)", re.I)

    for col in numeric_cols:
        if count_pattern.search(col):
            neg = int((pd.to_numeric(df[col], errors="coerce").dropna() < 0).sum())
            checks.append(_check(f"non_negative_{col}", "sanity", f"Count-like `{col}` has no negatives", neg == 0, "error", f"{neg} negatives" if neg else ""))
        if pct_pattern.search(col):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            out = int(((s < 0) | (s > 100)).sum())
            checks.append(_check(f"pct_range_{col}", "sanity", f"Percent-like `{col}` in [0,100]", out == 0, "warning", f"{out} out-of-range values" if out else ""))

    checks.append(_check("min_row_count", "sanity", "At least 3 rows", actual_rows >= 3, "error", f"rows={actual_rows}" if actual_rows < 3 else ""))
    checks.append(_check("min_col_count", "sanity", "At least 2 columns", actual_cols >= 2, "error", f"cols={actual_cols}" if actual_cols < 2 else ""))
    return checks


def _check_consistency(df: pd.DataFrame, analysis: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    col_stats_map = {s.get("column"): s for s in analysis.get("column_stats", []) if isinstance(s, dict) and s.get("column")}
    tol = 1e-3

    for col, stats in col_stats_map.items():
        if col not in df.columns:
            checks.append(_check(f"col_exists_{col}", "consistency", f"Column `{col}` exists", False, "error", "missing in CSV"))
            continue
        if stats.get("mean") is None:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        actual_mean = float(s.mean())
        rep_mean = float(stats.get("mean", actual_mean))
        mean_ok = abs(actual_mean - rep_mean) <= tol * (abs(rep_mean) + 1e-9)
        checks.append(_check(f"mean_match_{col}", "consistency", f"Mean for `{col}` matches", mean_ok, "warning", f"reported={rep_mean:.6g}, actual={actual_mean:.6g}" if not mean_ok else ""))

        actual_std = float(s.std())
        rep_std = float(stats.get("std", actual_std))
        std_ok = abs(actual_std - rep_std) <= tol * (abs(rep_std) + 1e-9)
        checks.append(_check(f"std_match_{col}", "consistency", f"Std for `{col}` matches", std_ok, "info", f"reported={rep_std:.6g}, actual={actual_std:.6g}" if not std_ok else ""))

        actual_min = float(s.min())
        rep_min = float(stats.get("min", actual_min))
        min_ok = abs(actual_min - rep_min) <= tol * (abs(rep_min) + 1e-9)
        checks.append(_check(f"min_match_{col}", "consistency", f"Min for `{col}` matches", min_ok, "info", f"reported={rep_min:.6g}, actual={actual_min:.6g}" if not min_ok else ""))

        actual_max = float(s.max())
        rep_max = float(stats.get("max", actual_max))
        max_ok = abs(actual_max - rep_max) <= tol * (abs(rep_max) + 1e-9)
        checks.append(_check(f"max_match_{col}", "consistency", f"Max for `{col}` matches", max_ok, "info", f"reported={rep_max:.6g}, actual={actual_max:.6g}" if not max_ok else ""))

    corr_pairs = analysis.get("correlations", [])
    num_df = df.select_dtypes(include=[np.number])
    for pair in corr_pairs[:10]:
        a = pair.get("col_a")
        b = pair.get("col_b")
        if a not in num_df.columns or b not in num_df.columns:
            continue
        actual_r = float(num_df[a].corr(num_df[b]))
        rep_r = float(pair.get("pearson_r", 0))
        ok = (actual_r >= 0) == (rep_r >= 0)
        checks.append(_check(f"corr_direction_{a}_{b}", "consistency", f"Correlation direction `{a}` <-> `{b}`", ok, "warning", f"reported={rep_r:+.4f}, actual={actual_r:+.4f}" if not ok else ""))

    return checks


def _check_visualizations(viz_summary: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    charts = viz_summary.get("charts", [])
    checks.append(_check("chart_count_nonzero", "visualization", "At least one chart produced", len(charts) > 0, "error", "No charts in viz_summary.json" if not charts else ""))

    for chart in charts:
        cid = str(chart.get("chart_id", "unknown"))
        path = Path(str(chart.get("png_path", "")))

        exists = path.exists() and path.stat().st_size > 0
        checks.append(_check(f"chart_file_exists_{cid}", "visualization", f"Chart `{cid}` file exists", exists, "error", str(path) if not exists else ""))
        if not exists:
            continue

        try:
            with Image.open(path) as img:
                w, h = img.size
            renderable = True
        except Exception as e:
            renderable = False
            w, h = 0, 0
            err = str(e)

        checks.append(_check(f"chart_renderable_{cid}", "visualization", f"Chart `{cid}` renderable", renderable, "error", err if not renderable else ""))
        if renderable:
            checks.append(_check(f"chart_resolution_{cid}", "visualization", f"Chart `{cid}` has min size", max(w, h) >= 600, "info", f"size={w}x{h}" if max(w, h) < 600 else ""))

    return checks


_DIMENSION_WEIGHTS = {
    "completeness": 0.25,
    "sanity": 0.35,
    "consistency": 0.25,
    "visualization": 0.15,
}

_SEVERITY_WEIGHT = {
    "error": 1.0,
    "warning": 0.5,
    "info": 0.1,
}


def _score_dimension(checks: list[CheckResult], dimension: str) -> DimensionScore:
    dim_checks = [c for c in checks if c.category == dimension]
    if not dim_checks:
        return DimensionScore(dimension=dimension, score=1.0, weight=_DIMENSION_WEIGHTS[dimension], checks_passed=0, checks_total=0)

    total = sum(_SEVERITY_WEIGHT[c.severity] for c in dim_checks)
    penalty = sum(_SEVERITY_WEIGHT[c.severity] for c in dim_checks if not c.passed)
    score = 1.0 - (penalty / total) if total > 0 else 1.0

    return DimensionScore(
        dimension=dimension,
        score=round(max(0.0, score), 4),
        weight=_DIMENSION_WEIGHTS[dimension],
        checks_passed=sum(1 for c in dim_checks if c.passed),
        checks_total=len(dim_checks),
    )


def _overall_confidence(dim_scores: list[DimensionScore]) -> float:
    total_weight = sum(d.weight for d in dim_scores)
    if total_weight == 0:
        return 0.0
    return round(sum(d.score * d.weight for d in dim_scores) / total_weight, 4)


def _derive_status(confidence: float, checks: list[CheckResult]) -> Status:
    has_blocking_error = any((not c.passed) and c.severity == "error" for c in checks)
    if has_blocking_error or confidence < 0.50:
        return "REJECTED"
    if confidence < 0.75:
        return "APPROVED_WITH_WARNINGS"
    return "APPROVED"


def _build_report(result: ValidationResult) -> str:
    """
    Build a clean, compact validation report in markdown format.
    Focus on readability with only essential information highlighted.
    """
    lines = [
        "# Validation Report",
        "",
        f"**Status:** {result.status} | **Confidence:** {result.overall_confidence:.1%}",
        "",
    ]
    
    # Summary metrics in a concise format
    lines += [
        "## Dataset Information",
        f"- Rows: {result.num_rows:,} | Columns: {result.num_columns}",
        f"- Source: {result.source}",
        f"- Charts Validated: {result.charts_validated}",
        "",
    ]
    
    # Quality scores by dimension
    lines += ["## Quality Scores"]
    for d in result.dimension_scores:
        status_icon = "✓" if d.score >= 0.9 else "⚠" if d.score >= 0.75 else "✗"
        lines.append(f"**{status_icon} {d.dimension.title()}:** {d.score:.1%} ({d.checks_passed}/{d.checks_total} passed)")
    lines.append("")
    
    # Group checks by category and show only failures
    failed_by_category = {}
    for c in result.checks:
        if not c.passed:
            if c.category not in failed_by_category:
                failed_by_category[c.category] = []
            failed_by_category[c.category].append(c)
    
    if failed_by_category:
        lines += ["## Issues Found"]
        for category in sorted(failed_by_category.keys()):
            checks = failed_by_category[category]
            lines.append(f"### {category.title()} ({len(checks)} issue{'s' if len(checks) != 1 else ''})")
            for check in checks:
                severity = "🔴" if check.severity == "error" else "🟡" if check.severity == "warning" else "ℹ️"
                detail = f" — {check.detail}" if check.detail else ""
                lines.append(f"- {severity} {check.description}{detail}")
            lines.append("")
    else:
        lines += ["", "✅ All validation checks passed!"]
    
    # Summary statistics
    total_checks = len(result.checks)
    passed_checks = sum(1 for c in result.checks if c.passed)
    lines += [
        "",
        "---",
        f"**Validation Complete:** {passed_checks}/{total_checks} checks passed",
        f"**Generated:** {result.timestamp}",
    ]

    return "\n".join(lines)


def run_validation(
    run_dir: Path,
    *,
    cleaned_csv_path: Path | None = None,
    analysis_json_path: Path | None = None,
    viz_summary_path: Path | None = None,
) -> dict[str, Any]:
    cleaned_csv = cleaned_csv_path or run_dir / "cleaned_data.csv"
    analysis_json = analysis_json_path or run_dir / "analysis_result.json"
    viz_json = viz_summary_path or run_dir / "viz_summary.json"

    df = pd.read_csv(cleaned_csv) if cleaned_csv.exists() else pd.DataFrame()
    analysis = _read_json(analysis_json) if analysis_json.exists() else {}
    viz_summary = _read_json(viz_json) if viz_json.exists() else {}

    checks: list[CheckResult] = []
    checks += _check_completeness(run_dir)
    if not df.empty:
        checks += _check_sanity(df, analysis)
        checks += _check_consistency(df, analysis)
    checks += _check_visualizations(viz_summary)

    dimensions = ["completeness", "sanity", "consistency", "visualization"]
    dim_scores = [_score_dimension(checks, d) for d in dimensions]
    confidence = _overall_confidence(dim_scores)
    status = _derive_status(confidence, checks)

    errors = [c.detail for c in checks if (not c.passed) and c.severity == "error" and c.detail]
    warnings = [c.detail for c in checks if (not c.passed) and c.severity == "warning" and c.detail]
    infos = [c.detail for c in checks if (not c.passed) and c.severity == "info" and c.detail]

    result = ValidationResult(
        status=status,
        overall_confidence=confidence,
        dimension_scores=dim_scores,
        checks=checks,
        errors=errors,
        warnings=warnings,
        infos=infos,
        source=str(analysis.get("source") or viz_summary.get("source") or run_dir),
        source_type=str(analysis.get("source_type") or viz_summary.get("source_type") or "unknown"),
        num_rows=len(df),
        num_columns=len(df.columns),
        charts_validated=len(viz_summary.get("charts", [])),
    )

    result_dict = result.model_dump()
    _write_json(run_dir / "validation_result.json", result_dict)
    _write_text(run_dir / "validation_report.md", _build_report(result))
    return result_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation Agent (Phase 5)")
    parser.add_argument("run_dir", help="Shared run directory")
    parser.add_argument("--cleaned-csv", default=None)
    parser.add_argument("--analysis-json", default=None)
    parser.add_argument("--viz-summary", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    result = run_validation(
        run_dir=run_dir,
        cleaned_csv_path=Path(args.cleaned_csv) if args.cleaned_csv else None,
        analysis_json_path=Path(args.analysis_json) if args.analysis_json else None,
        viz_summary_path=Path(args.viz_summary) if args.viz_summary else None,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("APPROVED", "APPROVED_WITH_WARNINGS") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
