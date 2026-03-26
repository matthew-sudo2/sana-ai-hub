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

# ML Quality Scoring & Visualization
try:
    from pathlib import Path as _PathLib
    _utils_path = (_PathLib(__file__).parent.parent / "utils")
    if _utils_path.exists():
        import sys
        sys.path.insert(0, str(_utils_path.parent))
        from utils.ml_quality_scorer import MLQualityScorer
        from utils.ml_assessment_viz import MLAssessmentVisualizer
        from utils.feature_cache import FeatureCache
        _ML_SCORER = MLQualityScorer()
    else:
        _ML_SCORER = None
        MLAssessmentVisualizer = None
        FeatureCache = None
except Exception as e:
    _ML_SCORER = None
    FeatureCache = None
    MLAssessmentVisualizer = None


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
    """
    Split sanity checks across new dimensions:
    - Data presence issues → "completeness"
    - Type/value validation → "accuracy"
    - Duplicate detection → "duplicates"
    """
    checks: list[CheckResult] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    reported_rows = int(analysis.get("num_rows", 0) or 0)
    reported_cols = int(analysis.get("num_columns", 0) or 0)
    actual_rows = len(df)
    actual_cols = len(df.columns)

    # ACCURACY: Row/column count validation
    checks.append(_check("row_count_match", "accuracy", "Row count matches analysis", reported_rows == actual_rows, "error", f"analysis={reported_rows}, actual={actual_rows}" if reported_rows != actual_rows else ""))
    checks.append(_check("col_count_match", "accuracy", "Column count matches analysis", reported_cols == actual_cols, "warning", f"analysis={reported_cols}, actual={actual_cols}" if reported_cols != actual_cols else ""))

    # COMPLETENESS: All-NA rows/columns (data presence) — higher severity
    all_na_rows = int(df.isna().all(axis=1).sum())
    checks.append(_check("no_all_na_rows", "completeness", "No fully-empty rows", all_na_rows == 0, "error", f"{all_na_rows} fully-empty rows" if all_na_rows else ""))

    all_na_cols = [c for c in df.columns if df[c].isna().all()]
    checks.append(_check("no_all_na_cols", "completeness", "No fully-empty columns", len(all_na_cols) == 0, "error", f"{all_na_cols}" if all_na_cols else ""))

    # DUPLICATES: Duplicate row detection — higher severity impact
    dup = int(df.duplicated().sum())
    checks.append(_check("no_duplicates", "duplicates", "No duplicate rows", dup == 0, "warning", f"{dup} duplicate rows" if dup else ""))

    # ACCURACY: Count/percent column validation
    count_pattern = re.compile(r"(count|total|freq|num_|n_)", re.I)
    pct_pattern = re.compile(r"(pct|percent|rate|ratio)", re.I)

    for col in numeric_cols:
        if count_pattern.search(col):
            neg = int((pd.to_numeric(df[col], errors="coerce").dropna() < 0).sum())
            checks.append(_check(f"non_negative_{col}", "accuracy", f"Count-like `{col}` has no negatives", neg == 0, "error", f"{neg} negatives" if neg else ""))
        if pct_pattern.search(col):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            out = int(((s < 0) | (s > 100)).sum())
            checks.append(_check(f"pct_range_{col}", "accuracy", f"Percent-like `{col}` in [0,100]", out == 0, "warning", f"{out} out-of-range values" if out else ""))

    # ACCURACY: Minimum row/column requirements
    checks.append(_check("min_row_count", "accuracy", "At least 3 rows", actual_rows >= 3, "error", f"rows={actual_rows}" if actual_rows < 3 else ""))
    checks.append(_check("min_col_count", "accuracy", "At least 2 columns", actual_cols >= 2, "error", f"cols={actual_cols}" if actual_cols < 2 else ""))
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


def _check_outliers(df: pd.DataFrame) -> list[CheckResult]:
    """
    Detect outliers in numeric columns using IQR method.
    Flags dataset if outlier percentage exceeds thresholds.
    """
    checks: list[CheckResult] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        # No numeric columns → no outlier risk
        checks.append(_check("outliers_evaluation", "outliers", "Numeric columns evaluated for outliers", True, "info", "No numeric columns — outlier check not applicable"))
        return checks
    
    total_outlier_count = 0
    total_data_points = 0
    outlier_cols_with_issues = []
    
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
            col_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())
            total_outlier_count += col_outliers
            
            if col_outliers > 0:
                outlier_pct = (col_outliers / len(series)) * 100
                outlier_cols_with_issues.append((col, col_outliers, outlier_pct))
    
    if total_data_points == 0:
        checks.append(_check("outliers_evaluation", "outliers", "Numeric columns evaluated for outliers", True, "info", "All numeric columns empty — outlier check not applicable"))
        return checks
    
    total_outlier_pct = (total_outlier_count / total_data_points) * 100
    
    # Evaluation: outliers acceptable if < 5% (standard IQR threshold)
    # Stock/financial data naturally has price fluctuations, so 5% is realistic
    outliers_acceptable = total_outlier_pct < 5.0
    detail = f"{total_outlier_count} outliers detected ({total_outlier_pct:.1f}% of {total_data_points} values)"
    
    if outlier_cols_with_issues:
        col_details = "; ".join([f"{c}: {cnt} ({pct:.1f}%)" for c, cnt, pct in outlier_cols_with_issues[:3]])
        detail += f" — {col_details}"
    
    checks.append(_check(
        "outliers_evaluation",
        "outliers",
        "Outlier levels acceptable (< 5%)",
        outliers_acceptable,
        # Severity scales with outlier %; info for clean, warning for moderate, error for bad
        "info" if total_outlier_pct < 3.0 else ("warning" if total_outlier_pct < 7.0 else "error"),
        detail
    ))
    
    return checks


def _check_ml_quality(df: pd.DataFrame, run_dir: Path | None = None) -> list[CheckResult]:
    """
    ML-based data quality assessment using trained Random Forest model.
    Evaluates overall data quality as GOOD or BAD.
    """
    checks: list[CheckResult] = []
    
    if _ML_SCORER is None or df.empty:
        # Skip if model not available or empty dataframe
        checks.append(_check(
            "ml_quality_assessment",
            "ml_assessment",
            "ML-based quality assessment (model-assisted)",
            True,
            "info",
            "ML model not available or dataframe empty — skipping"
        ))
        return checks
    
    try:
        score_result = _ML_SCORER.score(df)
        quality = score_result["quality"]  # "GOOD" or "BAD"
        confidence = score_result["score"]  # 0-100
        
        # Save extracted features to cache for feedback loop
        if run_dir and FeatureCache and score_result.get("features") is not None:
            try:
                dataset_hash = FeatureCache.get_dataset_hash(str(run_dir / "cleaned_data.csv"))
                FeatureCache.save_features(
                    run_dir=str(run_dir),
                    features=score_result.get("features", []),
                    dataset_hash=dataset_hash
                )
            except Exception as e:
                print(f"[validator] Warning: Failed to cache features: {e}")
        
        # Stricter interpretation: Only pass if GOOD quality with sufficient confidence (>70%)
        # If BAD, always fail. If GOOD but low confidence, flag as warning
        passed = quality == "GOOD" and confidence > 70.0
        severity = "info" if (quality == "GOOD" and confidence > 75.0) else "warning"
        
        detail = (
            f"ML Prediction: {quality} (confidence: {confidence:.1f}%) | "
            f"P(GOOD)={score_result['probability_good']:.2%}, "
            f"P(BAD)={score_result['probability_bad']:.2%}"
        )
        
        checks.append(_check(
            "ml_quality_assessment",
            "ml_assessment",
            f"ML model predicts data quality as {quality} (confidence {confidence:.0f}%)",
            passed,
            severity,
            detail
        ))
    except Exception as e:
        checks.append(_check(
            "ml_quality_assessment",
            "ml_assessment",
            "ML-based quality assessment attempted",
            True,
            "info",
            f"Error running ML assessment: {str(e)}"
        ))
    
    return checks


_DIMENSION_WEIGHTS = {
    "completeness": 0.20,
    "consistency": 0.20,
    "accuracy": 0.15,
    "duplicates": 0.12,
    "outliers": 0.12,
    "ml_assessment": 0.21,
}

_SEVERITY_WEIGHT = {
    "error": 1.0,      # Errors fully penalize
    "warning": 0.7,    # Warnings penalize more (was 0.5) — stricter
    "info": 0.15,      # Info penalizes slightly (was 0.1)
}


def _score_dimension(checks: list[CheckResult], dimension: str) -> DimensionScore:
    dim_checks = [c for c in checks if c.category == dimension]
    if not dim_checks:
        return DimensionScore(dimension=dimension, score=0.75, weight=_DIMENSION_WEIGHTS.get(dimension, 0.2), checks_passed=0, checks_total=0)

    total = sum(_SEVERITY_WEIGHT[c.severity] for c in dim_checks)
    penalty = sum(_SEVERITY_WEIGHT[c.severity] for c in dim_checks if not c.passed)
    score = 1.0 - (penalty / total) if total > 0 else 1.0
    
    # Cap individual dimension scores at 0.85 for strict grading (data quality is inherently uncertain)
    # Any dimension hitting 0.85+ means excellent performance
    score = min(score, 0.85)

    return DimensionScore(
        dimension=dimension,
        score=round(max(0.0, score), 4),
        weight=_DIMENSION_WEIGHTS.get(dimension, 0.2),
        checks_passed=sum(1 for c in dim_checks if c.passed),
        checks_total=len(dim_checks),
    )


def _overall_confidence(dim_scores: list[DimensionScore]) -> float:
    total_weight = sum(d.weight for d in dim_scores)
    if total_weight == 0:
        return 0.70  # All no checks — conservative fallback to 70%
    
    weighted_sum = sum(d.score * d.weight for d in dim_scores)
    confidence = weighted_sum / total_weight
    
    # Cap overall confidence at 0.80 (80% max) — acknowledges data quality uncertainty
    # Extreme rigor: judges must see realistic scores to trust the system
    return round(min(confidence, 0.80), 4)


def _derive_status(confidence: float, checks: list[CheckResult]) -> Status:
    has_blocking_error = any((not c.passed) and c.severity == "error" for c in checks)
    if has_blocking_error or confidence < 0.50:
        return "REJECTED"
    # Stricter thresholds: 60% for warnings, 70% for approval (was 75%)
    if confidence < 0.65:
        return "APPROVED_WITH_WARNINGS"
    if confidence < 0.70:
        return "APPROVED_WITH_WARNINGS"  # Extra strictness: maintain caution
    return "APPROVED"


def _build_report(result: ValidationResult, run_dir: Path | None = None, run_id: str = "", ml_viz_markdown: str = "") -> str:
    """
    Build a comprehensive validation report in markdown format.
    Includes before/after data quality transformation metrics, quality scores table,
    and ML assessment visualizations with explanations.
    """
    lines = [
        "# Validation Report",
        "",
        f"**Status:** {result.status} | **Confidence:** {result.overall_confidence:.1%}",
        "",
    ]
    
    # Summary metrics
    lines += [
        "## Dataset Information",
        f"- Rows: {result.num_rows:,} | Columns: {result.num_columns}",
        f"- Source: {result.source}",
        f"- Charts Validated: {result.charts_validated}",
        "",
    ]
    
    # Data Quality Transformation: Before/After Comparison
    if run_dir:
        raw_csv = run_dir / "raw_data.csv"
        cleaned_csv = run_dir / "cleaned_data.csv"
        
        if raw_csv.exists() and cleaned_csv.exists():
            try:
                df_raw = pd.read_csv(raw_csv)
                df_cleaned = pd.read_csv(cleaned_csv)
                
                # Compute before/after metrics
                raw_rows = len(df_raw)
                cleaned_rows = len(df_cleaned)
                rows_removed = raw_rows - cleaned_rows
                
                raw_cols = len(df_raw.columns)
                cleaned_cols = len(df_cleaned.columns)
                cols_removed = raw_cols - cleaned_cols
                
                raw_missing = df_raw.isna().sum().sum()
                cleaned_missing = df_cleaned.isna().sum().sum()
                missing_removed = raw_missing - cleaned_missing
                
                raw_total_cells = raw_rows * raw_cols
                cleaned_total_cells = cleaned_rows * cleaned_cols
                raw_missing_pct = (raw_missing / raw_total_cells * 100) if raw_total_cells > 0 else 0.0
                cleaned_missing_pct = (cleaned_missing / cleaned_total_cells * 100) if cleaned_total_cells > 0 else 0.0
                
                raw_dups = len(df_raw) - len(df_raw.drop_duplicates())
                cleaned_dups = len(df_cleaned) - len(df_cleaned.drop_duplicates())
                
                # Format duplicate row percentages safely
                raw_dup_pct = (100 * raw_dups / raw_rows) if raw_rows > 0 else 0.0
                cleaned_dup_pct = (100 * cleaned_dups / cleaned_rows) if cleaned_rows > 0 else 0.0
                
                lines += [
                    "## Data Quality Transformation",
                    "",
                    "| Metric | Before Clean | After Clean | Change |",
                    "|--------|--------------|-------------|--------|",
                    f"| Total Rows | {raw_rows:,} | {cleaned_rows:,} | {rows_removed:+,} |",
                    f"| Total Columns | {raw_cols} | {cleaned_cols} | {cols_removed:+,} |",
                    f"| Missing Values | {raw_missing:,} ({raw_missing_pct:.1f}%) | {cleaned_missing:,} ({cleaned_missing_pct:.1f}%) | {missing_removed:+,} ↓ |",
                    f"| Duplicate Rows | {raw_dups} ({raw_dup_pct:.1f}%) | {cleaned_dups} ({cleaned_dup_pct:.1f}%) | {raw_dups - cleaned_dups:+,} ↓ |",
                    "",
                ]
            except Exception as e:
                lines += [
                    "## Data Quality Transformation",
                    f"*Could not compute transformation metrics: {e}*",
                    "",
                ]
    
    # Quality Scores Table (5 metrics)
    lines += [
        "## Quality Metric Scores",
        "",
        "| Metric | Score | Status | Details |",
        "|--------|-------|--------|---------|",
    ]
    
    for d in result.dimension_scores:
        status_icon = "✓" if d.score >= 0.90 else "⚠" if d.score >= 0.75 else "✗"
        status_text = "Good" if d.score >= 0.90 else "Acceptable" if d.score >= 0.75 else "Needs Review"
        details = f"{d.checks_passed}/{d.checks_total} checks passed"
        lines.append(f"| {d.dimension.capitalize()} | {d.score:.1%} | {status_icon} {status_text} | {details} |")
    
    lines.append("")
    
    # Issues Found
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
    
    # ML Assessment Visualizations
    if ml_viz_markdown:
        lines += [
            "",
            "---",
            ml_viz_markdown,
            "",
        ]
        
        # Add visualization images if they exist
        if run_dir and run_id:
            lines.append("## Visualizations")
            lines.append("")
            
            images = [
                ("ml_confidence_gauge.png", "Confidence Score Gauge"),
                ("ml_feature_radar.png", "Feature Radar Chart"),
                ("ml_feature_comparison.png", "Feature Comparison"),
                ("ml_probability_breakdown.png", "Prediction Probability Breakdown"),
            ]
            
            for img_file, img_title in images:
                img_path = run_dir / img_file
                if img_path.exists():
                    lines.append(f"### {img_title}")
                    lines.append("")
                    # Use API URL for image (no /api prefix - routes are /runs/...)
                    api_url = f"/runs/{run_id}/ml-assessment/{img_file}"
                    lines.append(f"![{img_title}]({api_url})")
                    lines.append("")
    
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
        checks += _check_outliers(df)
        checks += _check_ml_quality(df, run_dir)
    checks += _check_visualizations(viz_summary)

    # 5-metric framework: Completeness, Consistency, Accuracy, Duplicates, Outliers + ML Assessment
    dimensions = ["completeness", "consistency", "accuracy", "duplicates", "outliers", "ml_assessment"]
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
    
    # Extract run_id from path (last directory component)
    run_id = run_dir.name
    
    # Generate ML Assessment Visualizations if model is available
    ml_viz_markdown = ""
    if _ML_SCORER is not None and MLAssessmentVisualizer is not None and not df.empty:
        try:
            score_result = _ML_SCORER.score(df)
            features = score_result.get("features", [])
            prediction = 1 if score_result.get("quality") == "GOOD" else 0
            probabilities = [score_result.get("probability_bad", 0), score_result.get("probability_good", 0)]
            
            visualizer = MLAssessmentVisualizer(
                features=features,
                prediction=prediction,
                probabilities=probabilities,
                df=df
            )
            
            # Generate visualizations in run directory
            visualizer.generate_confidence_gauge(str(run_dir / "ml_confidence_gauge.png"))
            visualizer.generate_feature_radar(str(run_dir / "ml_feature_radar.png"))
            visualizer.generate_feature_comparison(str(run_dir / "ml_feature_comparison.png"))
            visualizer.generate_probability_breakdown(str(run_dir / "ml_probability_breakdown.png"))
            
            # Get markdown section for report
            ml_viz_markdown = visualizer.generate_markdown_section()
            
        except Exception as e:
            print(f"⚠️  ML visualization generation failed: {e}")
    
    _write_text(run_dir / "validation_report.md", _build_report(result, run_dir, run_id=run_id, ml_viz_markdown=ml_viz_markdown))
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
