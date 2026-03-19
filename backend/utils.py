from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return (slug[:max_len] or "run").strip("_") or "run"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ollama_stop(model: str) -> None:
    subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name)
    if v is None or not v.strip():
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return v.strip()

