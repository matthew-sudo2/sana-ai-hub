"""
FastAPI server for Sana All May Label pipeline.
Exposes HTTP endpoints for frontend integration.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fastapi
import matplotlib
# ← FIX: Set non-interactive matplotlib backend FIRST to avoid tkinter threading errors
matplotlib.use('Agg')  # Use Agg backend (non-interactive, thread-safe)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from backend/.env (not root)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

from .graph import create_run, get_run, execute_run, PipelineState, _RUNS
from .agents.analyst import (
    _generate_visualization_explanation,
    _load_cached_explanation,
    _save_cached_explanation,
    VisualizationExplanation,
)
from .utils.ml_quality_scorer import MLQualityScorer
from .utils.feedback_db import FeedbackDB
from .utils.feature_cache import FeatureCache
from .utils.continuous_learner import ContinuousLearner


# ============================================================================
# Utility Functions
# ============================================================================

def _calculate_quality_score_decay(hours_since_retrain: float) -> float:
    """
    Calculate quality score decay factor based on time since last model retrain.
    
    Decay formula:
    - 0 days: 1.0 (100% confidence)
    - 7 days: 0.95 (95% confidence)
    - 30 days: 0.85 (85% confidence)
    - 90+ days: 0.70 (70% confidence - should retrain)
    
    Args:
        hours_since_retrain: Hours since model was last retrained
    
    Returns:
        Decay factor (0.70 to 1.0) to multiply with confidence score
    """
    days_since_retrain = hours_since_retrain / 24.0
    
    # Linear decay over 90 days: 1.0 → 0.70
    if days_since_retrain <= 0:
        return 1.0
    elif days_since_retrain >= 90:
        return 0.70
    else:
        # Linear interpolation: (1.0 - 0.70) * (1 - days/90) + 0.70
        decay = 1.0 - (days_since_retrain / 90.0) * 0.30
        return max(0.70, decay)


app = fastapi.FastAPI(
    title="Sana API",
    description="Sequential multi-agent pipeline orchestrator",
    version="1.0.0",
)

# CORS for frontend (Vite dev server runs on localhost:8080/8081)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================

class RunRequest(BaseModel):
    source: str | None = None
    url: str | None = None


class RunResponse(BaseModel):
    run_id: str
    phase: str
    status: str
    error: str | None = None
    output_dir: str | None = None


class CustomChartRequest(BaseModel):
    instruction: str


class MLQualityAssessment(BaseModel):
    quality: str  # "GOOD" or "BAD"
    score: float  # 0-100
    probability_good: float
    probability_bad: float
    features: list[float]


class QualityCheckResponse(BaseModel):
    run_id: str
    ml_assessment: MLQualityAssessment
    validation_status: str
    confidence: float


class FeedbackRequest(BaseModel):
    dataset_hash: str
    predicted_score: float
    actual_quality: int = Field(ge=0, le=3)  # Only allow 0-3 (poor to excellent)
    features: list[float] | None = None


class FeedbackResponse(BaseModel):
    status: str  # "stored" or "retrained"
    feedback_count: int
    cv_score: float | None = None  # Current cv score after retraining
    previous_cv_score: float | None = None  # Previous cv score
    improvement: float | None = None  # Percentage improvement
    model_version: int | None = None  # Model version number
    next_retrain_at: int | None = None
    message: str | None = None


class FeedbackStatsResponse(BaseModel):
    total_feedbacks: int
    models_trained: int
    current_cv_score: float | None
    latest_retrain_at: str | None
    improvement_percentage: float | None


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/run")
async def start_pipeline(
    file: UploadFile | None = File(None),
    source: str | None = Form(None),
) -> RunResponse:
    """
    Start a new pipeline run.

    Accepts either:
    - Multipart form data with 'file' field (CSV/XLSX/JSON)
    - Form data with 'source' field (URL/path)

    Returns run_id for polling.
    Pipeline runs in background; use GET /runs/{run_id} to check progress.
    """
    input_source = None
    source_type = "auto"

    # Handle file upload
    if file:
        try:
            # Save file to temporary location
            temp_dir = Path(__file__).parent / "runs" / "uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)

            file_path = temp_dir / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            input_source = str(file_path)
            source_type = "uploaded_file"  # Mark as uploaded file so scout is skipped
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to save uploaded file: {e}"},
            )

    # Handle form data source
    elif source:
        input_source = source.strip()

    if not input_source:
        return JSONResponse(
            status_code=400,
            content={"error": "file upload or source (URL/path) is required"},
        )

    # Create run with source_type info
    run_id = create_run(input_source, source_type=source_type)

    # Execute in background
    asyncio.create_task(_execute_async(run_id))

    state = get_run(run_id)
    return RunResponse(
        run_id=run_id,
        phase=state["phase"] if state else None,
        status=state["status"] if state else None,
        error=state["error"] if state else None,
    )


async def _execute_async(run_id: str) -> None:
    """Execute pipeline asynchronously."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, execute_run, run_id)


@app.post("/run/json")
async def start_pipeline_json(request: RunRequest) -> RunResponse:
    """
    Start a new pipeline run from JSON body.

    Accepts JSON with 'source' or 'url' field.

    Returns run_id for polling.
    Pipeline runs in background; use GET /runs/{run_id} to check progress.
    """
    input_source = (request.source or request.url or "").strip()

    if not input_source:
        return JSONResponse(
            status_code=400,
            content={"error": "source or url is required"},
        )

    # Create run (auto-detects source_type)
    run_id = create_run(input_source, source_type="auto")

    # Execute in background
    asyncio.create_task(_execute_async(run_id))

    state = get_run(run_id)
    return RunResponse(
        run_id=run_id,
        phase=state["phase"] if state else None,
        status=state["status"] if state else None,
        error=state["error"] if state else None,
    )


@app.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    """
    Get current status of a pipeline run.

    Returns: phase (scout|labeler|analyst|artist|validator|complete), status, error if any.
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    return RunResponse(
        run_id=run_id,
        phase=state["phase"],
        status=state["status"],
        error=state["error"],
        output_dir=(
            state["validator_output_dir"]
            or state["artist_output_dir"]
            or state["analyst_output_dir"]
            or state["labeler_output_dir"]
            or state["scout_output_dir"]
        ),
    )


@app.get("/runs/{run_id}/data.csv", response_model=None)
async def download_csv(run_id: str):
    """
    Download cleaned_data.csv from Phase 2+ output.

    Returns 404 if run not found or phase hasn't completed data cleaning.
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    if not state["labeler_output_dir"]:
        return JSONResponse(
            status_code=202,  # Accepted, not ready yet
            content={"error": "Labeler phase not yet complete"},
        )

    csv_path = Path(state["labeler_output_dir"]) / "cleaned_data.csv"
    if not csv_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"CSV not found at {csv_path}"},
        )

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="cleaned_data.csv",
    )


@app.get("/runs/{run_id}/images")
async def list_images(run_id: str):
    """
    List all PNG images from Phase 3 output (excluding ML assessment images).

    Returns: {images: [filename1.png, filename2.png, ...]}
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    if not state["artist_output_dir"]:
        return JSONResponse(
            status_code=202,
            content={"images": [], "message": "Artist phase not yet complete"},
        )

    output_dir = Path(state["artist_output_dir"])
    # Filter out ML assessment images (ml_*)  - they go to validation report only
    pngs = sorted([f.name for f in output_dir.glob("*.png") if not f.name.startswith("ml_")])

    return JSONResponse(
        {
            "images": pngs,
            "count": len(pngs),
        }
    )


@app.get("/runs/{run_id}/images/{filename}", response_model=None)
async def download_image(run_id: str, filename: str):
    """
    Download a specific PNG image from Phase 3 output.

    Prevents path traversal attacks by validating filename.
    """
    # Validate filename (prevent path traversal)
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid filename"},
        )

    state = get_run(run_id)
    if not state or not state["artist_output_dir"]:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} or output not found"},
        )

    image_path = Path(state["artist_output_dir"]) / filename
    if not image_path.exists() or not image_path.suffix.lower() == ".png":
        return JSONResponse(
            status_code=404,
            content={"error": f"Image not found: {filename}"},
        )

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=filename,
    )


@app.get("/runs/{run_id}/report")
async def get_report(run_id: str):
    """
    Get markdown report from final outputs.

    Returns: {content: "markdown content as string"}
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    output_dir = (
        state["validator_output_dir"]
        or state["artist_output_dir"]
        or state["analyst_output_dir"]
        or state["labeler_output_dir"]
        or state["scout_output_dir"]
    )
    if not output_dir:
        return JSONResponse(
            status_code=202,
            content={"error": "Pipeline output not yet available"},
        )

    out_dir = Path(output_dir)
    report_path = out_dir / "validation_report.md"
    filename = "validation_report.md"
    if not report_path.exists():
        report_path = out_dir / "analysis_report.md"
        filename = "analysis_report.md"

    if not report_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No report found (expected validation_report.md or analysis_report.md)"},
        )

    content = report_path.read_text(encoding="utf-8")
    return JSONResponse(
        {
            "content": content,
            "filename": filename,
        }
    )


@app.get("/runs/{run_id}/chart-explanations")
async def get_chart_explanations(run_id: str):
    """
    Get per-chart plain-English explanations generated by the Artist Agent.

    Returns chart_explanations.json which contains:
      - explanations[]: list of {filename, chart_type, columns_used, title,
                                  explanation, stats, generated_by}
      - key_insights[]: high-level dataset insights from the Analyst Agent
      - total_charts: int
      - timestamp: ISO datetime

    Available once the artist phase completes.
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    output_dir = _resolve_pipeline_output_dir(state)
    if not output_dir:
        return JSONResponse(
            status_code=202,
            content={"error": "Artist phase not yet complete — explanations not available"},
        )

    explanations_path = output_dir / "chart_explanations.json"
    if not explanations_path.exists():
        # Attempt to generate on-the-fly from existing artifacts
        viz_summary_path = output_dir / "viz_summary.json"
        analysis_json_path = output_dir / "analysis_result.json"
        if viz_summary_path.exists() and analysis_json_path.exists():
            try:
                from .agents.artist import _generate_chart_explanations, ChartRecord
                viz = _read_json_file(viz_summary_path)
                analysis = _read_json_file(analysis_json_path)
                charts = [ChartRecord(**c) for c in viz.get("charts", [])]
                _generate_chart_explanations(charts, analysis, output_dir)
            except Exception as e:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"chart_explanations.json not found and could not be generated: {e}"},
                )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "chart_explanations.json not found — artist phase may not have completed"},
            )

    return JSONResponse(_read_json_file(explanations_path))


@app.get("/runs/{run_id}/validation")
async def get_validation_result(run_id: str):
    """Get validation_result.json for a completed run."""
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    output_dir = (
        state["validator_output_dir"]
        or state["artist_output_dir"]
        or state["analyst_output_dir"]
        or state["labeler_output_dir"]
        or state["scout_output_dir"]
    )
    if not output_dir:
        return JSONResponse(
            status_code=202,
            content={"error": "Validation not yet available"},
        )

    result_path = Path(output_dir) / "validation_result.json"
    if not result_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "validation_result.json not found"},
        )

    return JSONResponse(_read_json_file(result_path))


@app.get("/runs/{run_id}/ml-assessment/{image_file}", response_model=None)
async def get_ml_assessment_image(run_id: str, image_file: str):
    """
    Get ML assessment visualization images (confidence gauge, radar, comparison, probability).
    Serves PNG images from validator output directory.
    
    Windows path limitation (260 chars) can truncate directory names. This endpoint:
    1. First tries UUID (short) lookup in _RUNS state
    2. Then searches file system for directory prefix match
    3. Returns image file by searching recursively through runs/
    """
    # Validate filename - strict security check
    if ".." in image_file or "/" in image_file or "\\" in image_file:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    
    if not image_file.startswith("ml_") or not image_file.endswith(".png"):
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})
    
    runs_base = Path(__file__).parent / "runs"
    
    # Strategy 1: Try UUID lookup first (if run_id is short UUID)
    state = get_run(run_id)
    if state and state.get("validator_output_dir"):
        candidate = Path(state["validator_output_dir"]) / image_file
        if candidate.exists() and candidate.is_file():
            try:
                return FileResponse(candidate, media_type="image/png")
            except Exception:
                pass
    
    # Strategy 2: Search file system for matching directory
    # The run_id might be truncated due to Windows 260-char path limit
    # Try to find any directory that starts with the given run_id
    if runs_base.exists():
        for run_dir in runs_base.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Check if directory name matches (exact or prefix)
            dir_name = run_dir.name
            if dir_name == run_id or dir_name.startswith(run_id):
                image_path = run_dir / image_file
                if image_path.exists() and image_path.is_file():
                    try:
                        return FileResponse(image_path, media_type="image/png")
                    except Exception:
                        pass
    
    # Strategy 3: If still not found, search for the image file anywhere in runs/
    # This handles cases where the entire directory name is truncated
    if runs_base.exists():
        for run_dir in runs_base.iterdir():
            if not run_dir.is_dir():
                continue
            image_path = run_dir / image_file
            if image_path.exists() and image_path.is_file():
                # Return the first match found
                try:
                    return FileResponse(image_path, media_type="image/png")
                except Exception:
                    pass
    
    # Not found
    return JSONResponse(status_code=404, content={"error": f"Image not found: {image_file}"})


@app.get("/runs/{run_id}/validation-report")
async def get_validation_report_markdown(run_id: str):
    """Get validation_report.md with full markdown content."""
    state = get_run(run_id)
    if not state:
        return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})
    
    output_dir = (
        state["validator_output_dir"]
        or state["artist_output_dir"]
        or state["analyst_output_dir"]
        or state["labeler_output_dir"]
        or state["scout_output_dir"]
    )
    
    if not output_dir:
        return JSONResponse(status_code=202, content={"error": "Validation not yet available"})
    
    report_path = Path(output_dir) / "validation_report.md"
    if not report_path.exists():
        return JSONResponse(status_code=404, content={"error": "validation_report.md not found"})
    
    try:
        report_content = report_path.read_text(encoding="utf-8")
        return JSONResponse({"content": report_content, "run_id": run_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/runs/{run_id}/cleaning-metrics")
async def get_cleaning_metrics(run_id: str):
    """Get cleaning metrics from phase2_summary.json (labeler phase output)."""
    state = get_run(run_id)
    if not state:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"},
        )

    if not state["labeler_output_dir"]:
        return JSONResponse(
            status_code=202,
            content={"error": "Labeler phase not yet complete"},
        )

    summary_path = Path(state["labeler_output_dir"]) / "phase2_summary.json"
    if not summary_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "phase2_summary.json not found"},
        )

    summary = _read_json_file(summary_path)
    # Extract just the cleaning_metrics field if present
    metrics = summary.get("cleaning_metrics", {})
    return JSONResponse(metrics)


@app.get("/runs/{run_id}/quality-assessment")
async def get_quality_assessment(run_id: str) -> dict[str, Any]:
    """
    Get ML-based quality assessment for a run's cleaned data.
    Includes quality score decay based on time since last model retraining.
    """
    try:
        state = get_run(run_id)
        if not state:
            return JSONResponse(
                status_code=404,
                content={"error": f"Run {run_id} not found"}
            )
        
        output_dir = (
            state["validator_output_dir"]
            or state["artist_output_dir"]
            or state["analyst_output_dir"]
            or state["labeler_output_dir"]
        )
        
        if not output_dir:
            return JSONResponse(
                status_code=202,
                content={"error": "Cleaning phase not yet complete"}
            )
        
        cleaned_csv = Path(output_dir) / "cleaned_data.csv"
        
        if not cleaned_csv.exists():
            return JSONResponse(
                status_code=400,
                content={"error": f"Cleaned data not found for run {run_id}"}
            )
        
        # Load and score
        df = pd.read_csv(cleaned_csv)
        scorer = MLQualityScorer()
        assessment = scorer.score(df)
        
        # Get validation status
        validation_result_path = Path(output_dir) / "validation_result.json"
        validation_status = "unknown"
        confidence = 0.5
        
        if validation_result_path.exists():
            validation_data = json.loads(validation_result_path.read_text())
            validation_status = validation_data.get("status", "unknown")
            confidence = validation_data.get("overall_confidence", 0.5)
        
        # Calculate quality score decay based on time since retrain
        decay_factor = 1.0
        hours_since_retrain = 0.0
        retrain_timestamp = None
        
        try:
            learner = ContinuousLearner()
            history = learner.get_model_history(max_records=1)
            
            if history and "timestamp" in history[0]:
                retrain_timestamp = history[0]["timestamp"]
                # Parse ISO format timestamp
                last_retrain = dt.datetime.fromisoformat(retrain_timestamp)
                now = dt.datetime.now(dt.timezone.utc)
                
                # Handle timezone-aware and naive datetimes
                if last_retrain.tzinfo is None:
                    last_retrain = last_retrain.replace(tzinfo=dt.timezone.utc)
                
                time_diff = now - last_retrain
                hours_since_retrain = time_diff.total_seconds() / 3600.0
                decay_factor = _calculate_quality_score_decay(hours_since_retrain)
        except Exception as e:
            print(f"[API] Warning: Could not calculate decay factor: {e}")
        
        return JSONResponse({
            "run_id": run_id,
            "ml_assessment": assessment,
            "validation_status": validation_status,
            "confidence": confidence,
            "decay_factor": round(decay_factor, 2),
            "hours_since_retrain": round(hours_since_retrain, 1),
            "last_retrain_timestamp": retrain_timestamp,
            "effective_confidence": round(confidence * decay_factor, 2),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get quality assessment: {str(e)}"}
        )


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_pipeline_output_dir(state: PipelineState) -> Path | None:
    output_dir = (
        state["validator_output_dir"]
        or state["artist_output_dir"]
        or state["analyst_output_dir"]
        or state["labeler_output_dir"]
        or state["scout_output_dir"]
    )
    return Path(output_dir) if output_dir else None


def _resolve_cleaned_csv_path(state: PipelineState) -> Path | None:
    if state["labeler_output_dir"]:
        candidate = Path(state["labeler_output_dir"]) / "cleaned_data.csv"
        if candidate.exists():
            return candidate

    out_dir = _resolve_pipeline_output_dir(state)
    if out_dir:
        candidate = out_dir / "cleaned_data.csv"
        if candidate.exists():
            return candidate

    return None


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _extract_columns_from_instruction(instruction: str, columns: list[str]) -> list[str]:
    """
    Extract matching column names from instruction using fuzzy matching.
    Tries multiple matching strategies for better coverage.
    """
    instruction_norm = _normalize_token(instruction)
    instruction_words = set(re.findall(r'\b\w+\b', instruction.lower()))

    matched: list[str] = []
    scored_matches: dict[str, int] = {}

    for col in columns:
        col_norm = _normalize_token(col)
        col_words = set(re.findall(r'\b\w+\b', col.lower()))

        if not col_norm:
            continue

        score = 0

        # Strategy 1: Exact substring in normalized version (high confidence)
        if col_norm in instruction_norm:
            score += 10

        # Strategy 2: Word match in original text (medium confidence)
        for word in col_words:
            if word in instruction_words:
                score += 5

        # Strategy 3: Partial substring match (lower confidence, but catches typos)
        if any(col_norm.startswith(word) or word.startswith(col_norm[:3])
               for word in instruction_words if len(word) >= 3):
            score += 2

        if score > 0:
            scored_matches[col] = score

    # Return columns sorted by score (highest first)
    if scored_matches:
        matched = sorted(scored_matches.items(), key=lambda x: x[1], reverse=True)
        matched = [col for col, _ in matched]

    return matched


@app.get("/runs/{run_id}/stats")
async def get_descriptive_stats(run_id: str):
    """
    Compute descriptive statistics for numeric columns in cleaned_data.csv.

    Includes standard deviation, max, min, variance, range, median, and mode.
    """
    state = get_run(run_id)
    if not state:
        return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})

    csv_path = _resolve_cleaned_csv_path(state)
    if not csv_path:
        return JSONResponse(status_code=404, content={"error": "cleaned_data.csv not found"})

    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    stats: list[dict[str, Any]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        mode_series = series.mode(dropna=True)
        mode_value = float(mode_series.iloc[0]) if not mode_series.empty else None
        min_value = float(series.min())
        max_value = float(series.max())
        stats.append(
            {
                "column": col,
                "standard_deviation": float(series.std(ddof=1)),
                "max": max_value,
                "min": min_value,
                "variance": float(series.var(ddof=1)),
                "range": float(max_value - min_value),
                "median": float(series.median()),
                "mode": mode_value,
            }
        )

    return JSONResponse(
        {
            "run_id": run_id,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_columns": len(stats),
            "stats": stats,
        }
    )


# ============================================================================
# Async Explanation Generation (non-blocking)
# ============================================================================

async def _generate_explanation_async(
    chart_type: str,
    title: str,
    x: str,
    y: str,
    dataset: str,
    output_dir: Path,
) -> VisualizationExplanation | None:
    """
    Generate visualization explanation in background without blocking.
    
    Runs in executor to avoid blocking the event loop.
    Includes caching to avoid redundant LLM calls.
    """
    try:
        import os
        
        # Check cache first
        cache_dir = output_dir / ".explanation_cache"
        cached = _load_cached_explanation(cache_dir, chart_type, x, y, dataset)
        if cached:
            return cached
        
        # Generate new explanation via LLM
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_PHASE3_MODEL", "llama3.2:3b")
        template_path = Path(__file__).parent / "prompts" / "analyst_prompt.txt"
        
        explanation = _generate_visualization_explanation(
            chart_type=chart_type,
            title=title,
            x=x,
            y=y,
            dataset=dataset,
            model=model,
            ollama_host=ollama_host,
            template_path=template_path,
        )
        
        # Cache the result if successful
        if explanation:
            _save_cached_explanation(cache_dir, explanation, chart_type, x, y, dataset)
        
        return explanation
    except Exception as e:
        print(f"[api] Explanation generation failed: {e}", flush=True)
        return None


@app.post("/runs/{run_id}/charts/custom")
async def create_custom_chart(run_id: str, request: CustomChartRequest):
    """
    Generate a custom chart from a natural-language instruction.

    Supports: histogram, scatter, line, bar charts.
    Uses fuzzy column matching to find relevant data.
    """
    try:
        state = get_run(run_id)
        if not state:
            return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})

        csv_path = _resolve_cleaned_csv_path(state)
        if not csv_path:
            return JSONResponse(status_code=404, content={"error": "cleaned_data.csv not found"})

        output_dir = _resolve_pipeline_output_dir(state)
        if not output_dir:
            return JSONResponse(status_code=404, content={"error": "No output directory found for run"})

        instruction = (request.instruction or "").strip()
        if not instruction:
            return JSONResponse(status_code=400, content={"error": "instruction is required"})

        # Load data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {str(e)}"})

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "cleaned_data.csv is empty"})

        sns.set_theme(style="whitegrid")

        columns = list(df.columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        matched_cols = _extract_columns_from_instruction(instruction, columns)
        instruction_lc = instruction.lower()

        # Determine chart type from instruction keywords
        if "hist" in instruction_lc or "distribution" in instruction_lc:
            chart_type = "histogram"
        elif "scatter" in instruction_lc:
            chart_type = "scatter"
        elif "line" in instruction_lc:
            chart_type = "line"
        elif "bar" in instruction_lc or "count" in instruction_lc:
            chart_type = "bar"
        else:
            # Default: scatter if 2+ numeric, else histogram
            chart_type = "scatter" if len(numeric_cols) >= 2 else "histogram"

        timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"custom_{chart_type}_{timestamp}.png"
        output_path = output_dir / filename

        # Generate chart based on type
        try:
            if chart_type == "histogram":
                # Select numeric column for histogram
                if matched_cols:
                    col = next((c for c in matched_cols if c in numeric_cols), None)
                    if not col and numeric_cols:
                        col = numeric_cols[0]
                elif numeric_cols:
                    col = numeric_cols[0]
                else:
                    return JSONResponse(status_code=400,
                                        content={"error": "No numeric columns available for histogram. Got: " + ", ".join(columns[:5])})

                # Extract numeric data
                try:
                    data = pd.to_numeric(df[col], errors="coerce").dropna()
                except Exception as e:
                    return JSONResponse(status_code=400,
                                        content={"error": f"Column '{col}' conversion failed: {str(e)}"})

                if data.empty or len(data) == 0:
                    return JSONResponse(status_code=400,
                                        content={"error": f"Column '{col}' has no numeric values ({len(df)} rows, {len(data)} numeric)"})

                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(data, bins="auto", kde=True, ax=ax)
                    ax.set_title(f"Histogram: {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    plt.tight_layout()
                    fig.savefig(output_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    return JSONResponse(status_code=500,
                                        content={"error": f"Failed to render histogram: {str(e)}"})

                columns_used = [col]
                title = f"Histogram: {col}"

            elif chart_type in {"scatter", "line"}:
                # Need 2 columns for scatter/line
                if len(matched_cols) >= 2:
                    x_col, y_col = matched_cols[0], matched_cols[1]
                elif len(matched_cols) == 1 and len(numeric_cols) >= 2:
                    x_col = matched_cols[0]
                    y_col = next((c for c in numeric_cols if c != x_col), numeric_cols[1] if len(numeric_cols) > 1 else None)
                elif len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                else:
                    available = ", ".join(columns[:10]) + ("..." if len(columns) > 10 else "")
                    return JSONResponse(status_code=400,
                                        content={"error": f"Need at least 2 columns for {chart_type}. Available: {available}"})

                # Prepare data
                try:
                    plot_df = df[[x_col, y_col]].copy()
                    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
                    plot_df = plot_df.dropna()
                except Exception as e:
                    return JSONResponse(status_code=400,
                                        content={"error": f"Data prep failed for {x_col}/{y_col}: {str(e)}"})

                if plot_df.empty or len(plot_df) == 0:
                    return JSONResponse(status_code=400,
                                        content={"error": f"No valid data for {x_col} vs {y_col} ({len(df)} rows, {len(plot_df)} after cleaning)"})

                try:
                    fig, ax = plt.subplots(figsize=(10, 5.5))
                    if chart_type == "scatter":
                        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.6, s=50)
                        ax.set_title(f"Scatter: {x_col} vs {y_col}")
                    else:  # line
                        ax.plot(plot_df[x_col], plot_df[y_col], linewidth=2, marker='o', markersize=4)
                        ax.set_title(f"Line Chart: {y_col} over {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    plt.tight_layout()
                    fig.savefig(output_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    return JSONResponse(status_code=500,
                                        content={"error": f"Failed to render {chart_type}: {str(e)}"})

                columns_used = [x_col, y_col]
                title = f"{chart_type.title()}: {x_col} and {y_col}"

            else:  # bar chart
                try:
                    if len(matched_cols) >= 2:
                        x_col, y_col = matched_cols[0], matched_cols[1]
                        bar_df = df[[x_col, y_col]].copy()
                        bar_df[y_col] = pd.to_numeric(bar_df[y_col], errors="coerce")
                        grouped = bar_df.dropna().groupby(x_col, as_index=False)[y_col].mean()

                        if len(grouped) > 20:
                            grouped = grouped.nlargest(20, y_col)

                        if grouped.empty or len(grouped) == 0:
                            return JSONResponse(status_code=400,
                                                content={"error": f"No valid data for bar chart: {x_col}, {y_col}"})

                        fig, ax = plt.subplots(figsize=(10, 5.5))
                        sns.barplot(data=grouped, x=x_col, y=y_col, ax=ax)
                        ax.set_title(f"Bar Chart: mean {y_col} by {x_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                        columns_used = [x_col, y_col]
                        title = f"Bar: {y_col} by {x_col}"
                    else:
                        # Single column bar - show value counts
                        x_col = matched_cols[0] if matched_cols else columns[0]
                        counts = df[x_col].astype(str).value_counts().head(20)

                        if counts.empty or len(counts) == 0:
                            return JSONResponse(status_code=400,
                                                content={"error": f"No data for bar chart in column: {x_col}"})

                        fig, ax = plt.subplots(figsize=(10, 5.5))
                        sns.barplot(x=counts.index, y=counts.values, ax=ax)
                        ax.set_title(f"Bar Chart: top values in {x_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel("Count")
                        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                        columns_used = [x_col]
                        title = f"Bar: top values in {x_col}"

                    plt.tight_layout()
                    fig.savefig(output_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    return JSONResponse(status_code=500,
                                        content={"error": f"Failed to render bar chart: {str(e)}"})

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Chart generation failed: {str(e)}"})

        # Update viz_summary.json (create if doesn't exist)
        try:
            viz_summary_path = output_dir / "viz_summary.json"
            if viz_summary_path.exists():
                summary = _read_json_file(viz_summary_path)
            else:
                summary = {"charts": [], "timestamp": dt.datetime.utcnow().isoformat()}

            charts = summary.get("charts", [])
            charts.append(
                {
                    "chart_id": output_path.stem,
                    "chart_type": chart_type,
                    "title": title,
                    "columns_used": columns_used,
                    "png_path": str(output_path),
                    "generated_by": "custom_instruction",
                }
            )
            summary["charts"] = charts
            summary["timestamp"] = dt.datetime.utcnow().isoformat()
            viz_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            # Also update chart_explanations.json with a simple entry for the new chart
            explanations_path = output_dir / "chart_explanations.json"
            if explanations_path.exists():
                try:
                    exp_data = _read_json_file(explanations_path)
                    col_list = ", ".join(f"'{c}'" for c in columns_used)
                    exp_data["explanations"].append({
                        "filename": filename,
                        "chart_id": output_path.stem,
                        "chart_type": chart_type,
                        "columns_used": columns_used,
                        "title": title,
                        "generated_by": "custom_instruction",
                        "explanation": (
                            f"Custom {chart_type} chart created from the instruction: "
                            f"\"{instruction}\". "
                            f"Columns used: {col_list}."
                        ),
                        "stats": {},
                    })
                    exp_data["total_charts"] = len(exp_data["explanations"])
                    exp_data["timestamp"] = dt.datetime.utcnow().isoformat()
                    explanations_path.write_text(
                        json.dumps(exp_data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass  # Non-critical — chart was created successfully

        except Exception as e:
            # Log but don't fail - chart was created successfully
            print(f"[WARNING] Failed to update viz_summary.json: {e}", flush=True)

        # Generate explanation asynchronously without blocking chart response
        dataset_name = Path(csv_path).stem if csv_path else "dataset"
        asyncio.create_task(
            _generate_explanation_async(
                chart_type=chart_type,
                title=title,
                x=columns_used[0] if len(columns_used) > 0 else "",
                y=columns_used[1] if len(columns_used) > 1 else columns_used[0] if columns_used else "",
                dataset=dataset_name,
                output_dir=output_dir,
            )
        )

        return JSONResponse(
            {
                "run_id": run_id,
                "filename": filename,
                "chart_type": chart_type,
                "columns_used": columns_used,
                "title": title,
                "instruction": instruction,
                "explanation": None,  # Explanation will be generated asynchronously
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})


@app.post("/runs/{run_id}/visualize-natural-language")
async def visualize_natural_language(run_id: str, request: CustomChartRequest):
    """
    Generate visualization from natural language request using LLM intent parser.

    Supports two modes:
    1. VISUALIZATION MODE - User wants an actual chart
       Request: {"instruction": "Show how prices changed over time"}
       Response: {"mode": "visualization", "chart": {...}, "plot_spec": {...}}

    2. GUIDANCE MODE - User asks for chart recommendations
       Request: {"instruction": "How should I visualize my data?"}
       Response: {"mode": "guidance", "guidance": {...}}

    Uses the Artist Agent's natural language parser to understand intent,
    then executes appropriate visualization based on PlotSpec.
    """
    try:
        state = get_run(run_id)
        if not state:
            return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})

        csv_path = _resolve_cleaned_csv_path(state)
        if not csv_path:
            return JSONResponse(status_code=404, content={"error": "cleaned_data.csv not found"})

        output_dir = _resolve_pipeline_output_dir(state)
        if not output_dir:
            return JSONResponse(status_code=404, content={"error": "No output directory found for run"})

        prompt = (request.instruction or "").strip()
        if not prompt:
            return JSONResponse(status_code=400, content={"error": "instruction is required"})

        # Load data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {str(e)}"})

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "cleaned_data.csv is empty"})

        # Call the artist agent's natural language visualizer
        from .agents.artist import generate_from_natural_language
        
        result = await generate_from_natural_language(
            df=df,
            user_prompt=prompt,
            out_dir=output_dir,
            model="llama3.2:3b",
            ollama_host="http://127.0.0.1:11434",
        )

        # Handle different response modes
        if result["mode"] == "visualization":
            # Update viz_summary.json with the new chart
            try:
                viz_summary_path = output_dir / "viz_summary.json"
                if viz_summary_path.exists():
                    summary = _read_json_file(viz_summary_path)
                else:
                    summary = {
                        "source": str(csv_path),
                        "source_type": "csv",
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "charts": [],
                        "timestamp": dt.datetime.utcnow().isoformat(),
                    }

                # Add the new chart to the summary
                if "chart" in result:
                    summary["charts"].append(result["chart"])
                    summary["timestamp"] = dt.datetime.utcnow().isoformat()
                    viz_summary_path.write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to update viz_summary.json: {e}", flush=True)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Natural language visualization failed: {str(e)}"}
        )


@app.get("/runs/{run_id}/charts/{chart_type}/explanation")
async def get_chart_explanation(
    run_id: str,
    chart_type: str,
    x: str = "",
    y: str = "",
    dataset: str = "dataset",
):
    """
    Fetch cached explanation for a visualization.
    
    Query parameters:
    - x: x-axis column name (or empty for single-column charts)
    - y: y-axis column name
    - dataset: dataset name/identifier
    
    Returns:
    {
        "explanation": {
            "summary": "...",
            "insight": "...",
            "interpretation": "...",
            "suggestion": "..."
        }
    }
    
    Returns 202 (Accepted) if explanation is still being generated.
    Returns 404 if chart or explanation not found.
    """
    try:
        state = get_run(run_id)
        if not state:
            return JSONResponse(
                status_code=404,
                content={"error": f"Run {run_id} not found"},
            )
        
        output_dir = _resolve_pipeline_output_dir(state)
        if not output_dir:
            return JSONResponse(
                status_code=404,
                content={"error": "No output directory found for run"},
            )
        
        # Load cached explanation
        cache_dir = output_dir / ".explanation_cache"
        explanation = _load_cached_explanation(cache_dir, chart_type, x, y, dataset)
        
        if explanation:
            return JSONResponse(
                {
                    "explanation": explanation.model_dump(),
                }
            )
        else:
            # Explanation not yet available (still generating or not requested)
            return JSONResponse(
                status_code=202,
                content={"message": "Explanation is being generated. Please try again in a moment."},
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch explanation: {str(e)}"},
        )


# ============================================================================
# Continuous Learning Feedback Loop
# ============================================================================

@app.post("/api/feedback")
async def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Record user feedback for continuous model improvement.
    
    User provides feedback on whether the quality score was accurate.
    Features are extracted during validation and stored for retraining.
    
    Auto-retrains every 20 feedback samples.
    
    Args:
        dataset_hash: MD5 hash of the dataset (for deduplication)
        predicted_score: Model's quality prediction (0-100)
        actual_quality: User feedback (0=poor, 1=fair, 2=good, 3=excellent)
        features: List of 8 extracted features (optional, can be fetched from cache)
    
    Returns:
        FeedbackResponse with status and next retrain count
    """
    try:
        feedback_db = FeedbackDB()
        
        # Store feedback
        feedback_db.save(
            dataset_hash=request.dataset_hash,
            predicted_score=request.predicted_score,
            actual_label=request.actual_quality,
            features=request.features or []
        )
        
        count = feedback_db.count()
        
        # Check if it's time to retrain: after 1st feedback, then every 5 feedbacks
        should_retrain = (count == 1) or (count >= 5 and count % 5 == 0)
        
        if should_retrain:
            print(f"\n[API] Feedback count: {count} - triggering auto-retrain...")
            
            try:
                learner = ContinuousLearner()
                result = learner.retrain()
                
                if result["success"]:
                    # Clean old feedback records after successful retrain (keep last 100)
                    deleted = feedback_db.clear_feedback(keep_last=100)
                    if deleted > 0:
                        print(f"[API] Cleaned {deleted} old feedback records after retrain")
                    
                    # Reload model into memory (important for stateful scoring)
                    print("[API] Reloading retrained model...")
                    scorer = MLQualityScorer()
                    scorer.reload_model()
                    print("[API] ✓ Model reloaded successfully")
                    
                    # Get model version and calculate improvement
                    model_history = learner.get_model_history(max_records=2)
                    model_version = len(model_history) if model_history else 1
                    
                    previous_cv_score = None
                    improvement_pct = None
                    
                    if len(model_history) >= 2:
                        previous_cv_score = model_history[1].get("cv_score")
                        current_cv_score = result["cv_score"]
                        if previous_cv_score and previous_cv_score > 0:
                            improvement_pct = ((current_cv_score - previous_cv_score) / previous_cv_score) * 100
                    
                    return FeedbackResponse(
                        status="retrained",
                        feedback_count=count,
                        cv_score=result["cv_score"],
                        previous_cv_score=previous_cv_score,
                        improvement=improvement_pct,
                        model_version=model_version,
                        message="✓ Thank you! Your feedback helps us improve our quality assessments."
                    )
                else:
                    next_retrain = 5 if count == 1 else (5 - (count % 5))
                    return FeedbackResponse(
                        status="stored",
                        feedback_count=count,
                        message=f"Feedback stored, but retrain failed: {result.get('error', 'Unknown error')}. Next retrain in {next_retrain} feedbacks.",
                        next_retrain_at=next_retrain
                    )
            except Exception as e:
                print(f"[API] Retrain error: {e}")
                next_retrain = 5 if count == 1 else (5 - (count % 5))
                return FeedbackResponse(
                    status="stored",
                    feedback_count=count,
                    message=f"Feedback stored. Retrain failed: {str(e)}",
                    next_retrain_at=next_retrain
                )
        
        # Not time to retrain yet - calculate feedbacks remaining
        if count == 0:
            next_retrain = 1
        elif count == 1:
            next_retrain = 4  # Next retrain at count=5
        else:
            next_retrain = 5 - (count % 5)
        
        return FeedbackResponse(
            status="stored",
            feedback_count=count,
            next_retrain_at=next_retrain,
            message=f"✓ Feedback stored. {next_retrain} more feedbacks until next retrain."
        )
    
    except Exception as e:
        print(f"[API] Feedback error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to store feedback: {str(e)}"}
        )


@app.get("/api/feedback/stats")
async def get_feedback_stats() -> dict[str, Any]:
    """
    Get feedback loop statistics and model improvement metrics.
    Includes model decay status and recommendations for retraining.
    
    Returns:
        Dictionary with:
        - total_feedbacks: Number of feedback records
        - models_trained: Number of retraining events  
        - current_cv_score: Latest model's cross-val score
        - latest_retrain_at: Timestamp of last retrain
        - improvement_percentage: Score improvement (if multiple retrains)
        - hours_since_retrain: Hours since last model retraining
        - decay_factor: Current quality score decay factor
        - model_status: "fresh" (0-7 days), "good" (7-30 days), "aging" (30-90 days), "stale" (90+ days)
        - next_retrain_recommended_at: Hours remaining before retraining recommended
    """
    try:
        feedback_db = FeedbackDB()
        learner = ContinuousLearner()
        
        total_feedbacks = feedback_db.count()
        history = learner.get_model_history(max_records=10)
        
        models_trained = len(history)
        current_cv_score = None
        latest_retrain_at = None
        improvement_percentage = None
        hours_since_retrain = 0.0
        decay_factor = 1.0
        model_status = "unknown"
        next_retrain_recommended_at = None
        
        if history:
            latest = history[-1]
            current_cv_score = latest.get("cv_score", 0.0)
            latest_retrain_at = latest.get("timestamp")
            
            # Calculate improvement if we have history
            if len(history) > 1:
                first = history[0]
                first_score = first.get("cv_score", 0.0)
                improvement_percentage = ((current_cv_score - first_score) / first_score * 100) if first_score > 0 else None
            
            # Calculate time since retrain and decay
            if latest_retrain_at:
                try:
                    last_retrain = dt.datetime.fromisoformat(latest_retrain_at)
                    now = dt.datetime.now(dt.timezone.utc)
                    
                    # Handle timezone-aware and naive datetimes
                    if last_retrain.tzinfo is None:
                        last_retrain = last_retrain.replace(tzinfo=dt.timezone.utc)
                    
                    time_diff = now - last_retrain
                    hours_since_retrain = time_diff.total_seconds() / 3600.0
                    decay_factor = _calculate_quality_score_decay(hours_since_retrain)
                    
                    # Determine model status based on age
                    days_old = hours_since_retrain / 24.0
                    if days_old < 7:
                        model_status = "fresh"
                    elif days_old < 30:
                        model_status = "good"
                    elif days_old < 90:
                        model_status = "aging"
                    else:
                        model_status = "stale"
                    
                    # Calculate hours until retraining is recommended (at 30 days)
                    hours_until_recommended_retrain = max(0, (30 * 24) - hours_since_retrain)
                    next_retrain_recommended_at = hours_until_recommended_retrain
                    
                except Exception as e:
                    print(f"[API] Error calculating decay: {e}")
        
        return {
            "total_feedbacks": total_feedbacks,
            "models_trained": models_trained,
            "current_cv_score": current_cv_score,
            "latest_retrain_at": latest_retrain_at,
            "improvement_percentage": improvement_percentage,
            "hours_since_retrain": round(hours_since_retrain, 1),
            "decay_factor": round(decay_factor, 2),
            "model_status": model_status,
            "next_retrain_recommended_at": round(next_retrain_recommended_at, 1) if next_retrain_recommended_at else None
        }
    
    except Exception as e:
        print(f"[API] Stats error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get feedback stats: {str(e)}"}
        )


@app.get("/api/features/{run_id}")
async def get_cached_features(run_id: str) -> dict:
    """
    Retrieve cached ML features for a specific run.
    
    Features are extracted by MLQualityScorer during validation.
    Returns the 8 statistical features needed for feedback retraining.
    
    Args:
        run_id: The pipeline run ID
    
    Returns:
        dict with:
        - features: List of 8 floats [missing_ratio, duplicate_ratio, numeric_ratio, ...]
        - dataset_hash: MD5 hash of the cleaned CSV
        - feature_names: Names of the 8 features
        - error: Error message if features not found
    """
    try:
        # Get run metadata
        run_data = _RUNS.get(run_id, {})
        if not run_data:
            return JSONResponse(
                status_code=404,
                content={"error": "Run not found", "features": [], "dataset_hash": ""}
            )
        
        # Get output directory path
        output_dir = run_data.get("output_dir")
        if not output_dir:
            return JSONResponse(
                status_code=404,
                content={"error": "Output directory not found", "features": [], "dataset_hash": ""}
            )
        
        # Check for cached features file
        features_file = Path(output_dir) / "features.json"
        if not features_file.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Features not cached for this run", "features": [], "dataset_hash": ""}
            )
        
        # Load features
        features_data = json.loads(features_file.read_text())
        return {
            "features": features_data.get("features", []),
            "dataset_hash": features_data.get("dataset_hash", ""),
            "feature_names": features_data.get("feature_names", [])
        }
        
    except Exception as e:
        print(f"[api] Error retrieving features: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "features": [], "dataset_hash": ""}
        )


@app.post("/api/data-hash")
async def compute_data_hash(file: UploadFile = File(...)) -> dict:
    """
    Compute MD5 hash of uploaded dataset for deduplication.
    
    Used to ensure consistent hashing between frontend and backend.
    Same dataset always produces same hash.
    
    Args:
        file: CSV file to hash
    
    Returns:
        dict with:
        - hash: MD5 hash of file contents (hex string)
        - filename: Original filename
    
    Example:
        POST /api/data-hash
        file: (binary CSV data)
        
        Returns:
        {
          "hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
          "filename": "data.csv"
        }
    """
    try:
        contents = await file.read()
        data_hash = hashlib.md5(contents).hexdigest()
        return {
            "hash": data_hash,
            "filename": file.filename
        }
    except Exception as e:
        print(f"[api] Hash computation error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/feedback/health")
async def feedback_db_health() -> dict:
    """
    Check feedback database integrity and health.
    
    Returns:
        dict with:
        - total_feedbacks: Total records
        - valid_with_features: Records with valid 8-feature arrays
        - invalid_empty_features: Records with missing/invalid features
        - health_percentage: Percentage of valid records
        - status: "healthy" | "degraded" | "poor" | "no_data"
    """
    try:
        feedback_db = FeedbackDB()
        total = feedback_db.count()
        
        if total == 0:
            return {
                "total_feedbacks": 0,
                "valid_with_features": 0,
                "invalid_empty_features": 0,
                "health_percentage": 100.0,
                "status": "no_data"
            }
        
        # Count valid records
        valid_features, valid_labels = feedback_db.get_feedback_for_retraining()
        valid_count = len(valid_features)
        invalid_count = total - valid_count
        health = (valid_count / total * 100) if total > 0 else 0
        
        return {
            "total_feedbacks": total,
            "valid_with_features": valid_count,
            "invalid_empty_features": invalid_count,
            "health_percentage": round(health, 1),
            "status": "healthy" if health > 90 else "degraded" if health > 70 else "poor"
        }
    except Exception as e:
        print(f"[api] Feedback health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return JSONResponse(
        {
            "name": "Sana All May Label API",
            "version": "1.0.0",
            "endpoints": {
                "POST /run": "Start a new pipeline (body: {source}, url also accepted)",
                "GET /runs/{run_id}": "Check pipeline status",
                "GET /runs/{run_id}/data.csv": "Download cleaned CSV",
                "GET /runs/{run_id}/images": "List PNG images",
                "GET /runs/{run_id}/images/{filename}": "Download PNG image",
                "GET /runs/{run_id}/report": "Get markdown report",
                "GET /runs/{run_id}/validation": "Get validation_result.json",
                "GET /runs/{run_id}/stats": "Get descriptive stats for cleaned CSV",
                "GET /runs/{run_id}/chart-explanations": "Get per-chart explanations JSON",
                "POST /runs/{run_id}/charts/custom": "Generate custom chart from instruction",
                "GET /runs/{run_id}/charts/{chart_type}/explanation": "Get AI-generated explanation for chart (query: x, y, dataset)",
                "POST /api/feedback": "Record feedback on ML quality prediction (0-3 rating). Triggers auto-retrain every 5 feedbacks",
                "GET /api/feedback/stats": "Get feedback statistics: total feedbacks, models trained, improvement %",
                "GET /api/feedback/health": "Check feedback database integrity and data quality",
                "GET /api/features/{run_id}": "Retrieve cached ML features from a run (for feedback submission)",
                "POST /api/data-hash": "Compute MD5 hash of dataset for consistent deduplication",
                "GET /health": "Health check",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )