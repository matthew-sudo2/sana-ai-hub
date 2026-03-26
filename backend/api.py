"""
FastAPI server for Sana All May Label pipeline.
Exposes HTTP endpoints for frontend integration.
"""

from __future__ import annotations

import asyncio
import datetime as dt
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
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from backend/.env (not root)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

from .graph import create_run, get_run, execute_run, PipelineState
from .agents.analyst import (
    _generate_visualization_explanation,
    _load_cached_explanation,
    _save_cached_explanation,
    VisualizationExplanation,
)


app = fastapi.FastAPI(
    title="Sana API",
    description="Sequential multi-agent pipeline orchestrator",
    version="1.0.0",
)

# CORS for frontend (Vite dev server runs on localhost:8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
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
    List all PNG images from Phase 3 output.
    
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
    pngs = sorted([f.name for f in output_dir.glob("*.png")])
    
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
                "POST /runs/{run_id}/charts/custom": "Generate custom chart from instruction",
                "GET /runs/{run_id}/charts/{chart_type}/explanation": "Get AI-generated explanation for chart (query: x, y, dataset)",
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
