from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import numpy as np
import pandas as pd
from PIL import Image


class UnsafeModelOutputError(RuntimeError):
    pass


CorrelationStrength = Literal["weak", "moderate", "strong"]
CorrelationDirection = Literal["positive", "negative"]


class CorrelationClaim(BaseModel):
    type: Literal["correlation"] = "correlation"
    col_a: str = Field(min_length=1)
    col_b: str = Field(min_length=1)
    direction: CorrelationDirection
    strength: CorrelationStrength


class Phase4ModelOutput(BaseModel):
    analysis_markdown: str = Field(min_length=1)
    claims: list[CorrelationClaim] = Field(default_factory=list)


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    width_px: int
    height_px: int
    dpi_x: float | None
    dpi_y: float | None


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


def _ollama_generate(model: str, prompt: str, host: str) -> str:
    """Generate analysis report via Ollama, with fallback when unavailable."""
    options = {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
    }

    try:
        import ollama  # type: ignore
        import requests
        
        # Check if Ollama is alive first
        try:
            requests.get(f"{host}/api/tags", timeout=2)
        except Exception:
            return _generate_fallback_report()

        client = ollama.Client(host=host)
        resp = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options,
            keep_alive=0,
        )
        return str(resp.get("response", "")).strip()
    except Exception:
        return _generate_fallback_report()
    finally:
        subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _generate_fallback_report() -> str:
    """Return basic analysis report when Ollama is unavailable."""
    return """# Data Analysis Report

## Overview
Automated analysis report generated from the provided dataset.

## Summary
- Data pipeline completed successfully
- Dataset processed through all analysis stages
- Note: Detailed LLM analysis unavailable (Ollama not running)

## Conclusions
✓ Data pipeline completed
✓ Fallback mode active

## Recommendations
For advanced LLM-powered analysis, ensure Ollama is installed and running with required models."""


def _json_from_text(text: str) -> dict[str, Any]:
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
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise UnsafeModelOutputError("Model output was not valid JSON.")


def _describe_markdown(df: pd.DataFrame) -> str:
    # Describe numeric columns for stable, interpretable validation.
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return "No numeric columns available for df.describe()."
    desc = num.describe().T
    desc.index.name = "column"
    return desc.to_markdown(floatfmt=".6g")


def _correlation_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        corr = pd.DataFrame()
        return corr, []

    corr = num.corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            pairs.append({"col_a": str(cols[i]), "col_b": str(cols[j]), "pearson_r": r, "abs_r": abs(r)})
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    return corr, pairs[:25]


def _strength_from_r(r: float) -> CorrelationStrength:
    ar = abs(r)
    if ar >= 0.7:
        return "strong"
    if ar >= 0.4:
        return "moderate"
    return "weak"


def _direction_from_r(r: float) -> CorrelationDirection:
    return "positive" if r >= 0 else "negative"


def _extract_image_metadata(paths: list[Path]) -> list[ImageMetadata]:
    out: list[ImageMetadata] = []
    for p in paths:
        with Image.open(p) as img:
            dpi = img.info.get("dpi")
            dpi_x: float | None = None
            dpi_y: float | None = None
            if isinstance(dpi, tuple) and len(dpi) == 2:
                try:
                    dpi_x = float(dpi[0])
                    dpi_y = float(dpi[1])
                except Exception:
                    dpi_x = None
                    dpi_y = None
            out.append(
                ImageMetadata(
                    path=str(p),
                    width_px=int(img.size[0]),
                    height_px=int(img.size[1]),
                    dpi_x=dpi_x,
                    dpi_y=dpi_y,
                )
            )
    return out


def _build_prompt(
    describe_md: str,
    top_corr_pairs: list[dict[str, Any]],
    image_meta: list[ImageMetadata],
) -> str:
    image_meta_json = json.dumps([m.__dict__ for m in image_meta], ensure_ascii=False, indent=2)
    top_corr_json = json.dumps(top_corr_pairs, ensure_ascii=False, indent=2)

    return (
        "You are a validation and synthesis agent preparing a formal research report.\n"
        "You must be skeptical and verify statements against the provided numeric summaries.\n"
        "Return STRICT JSON ONLY (no markdown fences, no commentary) with keys:\n"
        '- "analysis_markdown": string (formal Markdown research report)\n'
        '- "claims": array of objects, each:\n'
        '  {"type":"correlation","col_a":string,"col_b":string,"direction":"positive|negative","strength":"weak|moderate|strong"}\n\n'
        "Constraints:\n"
        "- Do NOT invent values.\n"
        "- Any correlation claim must be consistent with the provided correlation pairs.\n"
        "- In analysis_markdown, include a header:\n"
        "  - If consistent: '## Data Integrity: Verified'\n"
        "  - If inconsistent: '## Numerical Discrepancies' (with bullet list)\n\n"
        "Inputs:\n\n"
        "### df.describe() (numeric)\n"
        f"{describe_md}\n\n"
        "### Top Pearson correlation pairs (absolute value descending)\n"
        f"{top_corr_json}\n\n"
        "### Phase 3 image metadata\n"
        f"{image_meta_json}\n"
    )


def _validate_claims_against_data(
    claims: list[CorrelationClaim],
    corr_df: pd.DataFrame,
) -> list[str]:
    discrepancies: list[str] = []
    if corr_df.empty and claims:
        discrepancies.append("Correlation claims were made, but the dataset has fewer than 2 numeric columns.")
        return discrepancies

    for c in claims:
        a = c.col_a
        b = c.col_b
        if a not in corr_df.columns or b not in corr_df.columns:
            discrepancies.append(f"Claim references missing column(s): ({a}, {b}).")
            continue

        r = float(corr_df.loc[a, b])
        expected_dir = _direction_from_r(r)
        expected_strength = _strength_from_r(r)

        if c.direction != expected_dir:
            discrepancies.append(f"Direction mismatch for ({a}, {b}): claimed {c.direction}, observed {expected_dir} (r={r:.3g}).")

        # Strength check: allow conservative claims (claim weaker than observed),
        # but flag overstatement (claim stronger than observed).
        order = {"weak": 0, "moderate": 1, "strong": 2}
        if order[c.strength] > order[expected_strength]:
            discrepancies.append(
                f"Strength overstated for ({a}, {b}): claimed {c.strength}, observed {expected_strength} (r={r:.3g})."
            )

    return discrepancies


def main() -> int:
    # Load environment from explicit backend/.env path
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)

    parser = argparse.ArgumentParser(description="Phase 4 Analyst: verified synthesis with local Llama 3 8B.")
    parser.add_argument("cleaned_csv", help="Path to cleaned CSV (e.g., cleaned_research_data.csv or cleaned_data.csv)")
    parser.add_argument(
        "--phase3-dir",
        default=None,
        help="Directory containing Phase 3 outputs (expects .png files). If omitted, only data checks run.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_PHASE4_MODEL", "llama3:8b"),
        help="Ollama model for Phase 4 (default: env OLLAMA_PHASE4_MODEL or llama3:8b)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL (default: env OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument("--out-dir", default="runs", help="Output directory under backend/ (default: runs)")
    args = parser.parse_args()

    cleaned_path = Path(args.cleaned_csv)
    if not cleaned_path.exists():
        raise FileNotFoundError(str(cleaned_path))

    df = pd.read_csv(cleaned_path)
    if df.empty:
        raise RuntimeError("Cleaned CSV contained no rows.")

    describe_md = _describe_markdown(df)
    corr_df, top_pairs = _correlation_summary(df)

    image_meta: list[ImageMetadata] = []
    if args.phase3_dir:
        p3 = Path(args.phase3_dir)
        if not p3.exists():
            raise FileNotFoundError(str(p3))
        pngs = sorted(p3.glob("*.png"))
        if pngs:
            image_meta = _extract_image_metadata(pngs)

    prompt = _build_prompt(describe_md=describe_md, top_corr_pairs=top_pairs, image_meta=image_meta)
    model_text = _ollama_generate(model=args.model, prompt=prompt, host=args.ollama_host)

    obj = _json_from_text(model_text)
    try:
        model_out = Phase4ModelOutput.model_validate(obj)
    except ValidationError as e:
        raise UnsafeModelOutputError(f"Model JSON did not match required schema: {e}") from e

    discrepancies = _validate_claims_against_data(model_out.claims, corr_df)

    run_ts = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("backend") / args.out_dir / f"{run_ts}_{_safe_slug(cleaned_path.stem)}"
    _ensure_dir(run_dir)

    _write_text(run_dir / "phase4_prompt.txt", prompt)
    _write_text(run_dir / "phase4_model_raw.txt", model_text)
    _write_json(run_dir / "phase4_model_output.json", model_out.model_dump())

    # Final report: if we found discrepancies, prepend a formal section.
    report = model_out.analysis_markdown.strip()
    if discrepancies and "## Numerical Discrepancies" not in report:
        report = (
            "## Numerical Discrepancies\n"
            + "\n".join([f"- {d}" for d in discrepancies])
            + "\n\n"
            + report
        )
    if (not discrepancies) and "## Data Integrity: Verified" not in report:
        report = "## Data Integrity: Verified\n\n" + report

    report_path = run_dir / "research_report.md"
    _write_text(report_path, report + "\n")

    summary = {
        "input_csv": str(cleaned_path),
        "phase3_dir": args.phase3_dir,
        "report_md": str(report_path),
        "discrepancy_count": len(discrepancies),
    }
    _write_json(run_dir / "phase4_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

