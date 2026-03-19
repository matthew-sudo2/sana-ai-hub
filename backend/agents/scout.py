from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ExtractedMetadata(BaseModel):
    source_url: str = Field(min_length=1)
    publication_date: Optional[str] = None
    primary_topic: str = Field(min_length=1)
    raw_quantitative_stats: Any


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:80] or "run"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def crawl_with_spider(source_url: str, limit: int = 5, return_format: str = "markdown") -> Any:
    """
    Uses the `spider_client` package's API surface shown in the user's snippet:
      # pip install spider_client
      from spider import Spider
      app = Spider(api_key=...)
      result = app.crawl_url(url, params={...})
    """
    spider_api_key = _require_env("SPIDER_API_KEY")

    try:
        from spider import Spider  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import Spider client. Ensure you installed `spider_client` "
            "into your current Python environment."
        ) from e

    app = Spider(api_key=spider_api_key)
    result = app.crawl_url(
        source_url,
        params={"limit": int(limit), "return_format": str(return_format)},
    )
    return result


def _json_from_text(text: str) -> dict[str, Any]:
    """
    Robust JSON extraction for LLM outputs:
    - Prefer full-string JSON
    - Otherwise try to extract the first {...} block
    """
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

    raise ValueError("Model output was not valid JSON.")


def extract_entities_with_ollama(source_url: str, crawled_payload: Any) -> ExtractedMetadata:
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

    prompt = (
        "You are an information extraction system.\n"
        "Task: Extract relevant metadata from the provided crawled web content.\n"
        "Return STRICT JSON ONLY (no markdown, no commentary).\n\n"
        "Required JSON keys:\n"
        '- "source_url": string (must equal the provided URL)\n'
        '- "publication_date": string or null (ISO 8601 preferred if available)\n'
        '- "primary_topic": string (concise)\n'
        '- "raw_quantitative_stats": object or array (numbers and units as found; do not invent)\n\n'
        f"Provided source_url: {source_url}\n\n"
        "Crawled content (may be markdown/HTML or structured):\n"
        f"{crawled_payload}\n"
    )

    # Prefer the official Python client; fall back to the CLI if needed.
    # Sequential Loading: keep single call and explicitly request unload after.
    text_out: str
    try:
        import ollama  # type: ignore

        client = ollama.Client(host=ollama_host)
        resp = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.1},
            keep_alive=0,
        )
        text_out = str(resp.get("response", "")).strip()
    except Exception:
        completed = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        text_out = completed.stdout.decode("utf-8", errors="replace").strip()

    try:
        obj = _json_from_text(text_out)
        obj["source_url"] = source_url
        extracted = ExtractedMetadata.model_validate(obj)
        return extracted
    except (ValueError, ValidationError) as e:
        raise RuntimeError(f"Entity extraction failed: {e}") from e
    finally:
        # Best-effort explicit unload to free VRAM for the next phase.
        subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Phase 1 Scout: crawl URL + extract metadata via local Ollama.")
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--limit", type=int, default=5, help="Spider crawl limit (default: 5)")
    parser.add_argument("--return-format", default="markdown", help="Spider return_format (default: markdown)")
    parser.add_argument("--out-dir", default="runs", help="Output directory (default: runs)")
    args = parser.parse_args()

    source_url = args.url.strip()
    if not source_url:
        raise RuntimeError("URL must be non-empty.")

    run_ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("backend") / args.out_dir / f"{run_ts}_{_safe_slug(source_url)}"
    _ensure_dir(run_dir)

    crawled = crawl_with_spider(source_url, limit=args.limit, return_format=args.return_format)
    _write_json(run_dir / "crawl_raw.json", crawled)

    extracted = extract_entities_with_ollama(source_url, crawled)
    _write_json(run_dir / "extracted_metadata.json", extracted.model_dump())

    print(json.dumps(extracted.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
