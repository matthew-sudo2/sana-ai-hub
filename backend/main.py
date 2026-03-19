from __future__ import annotations

"""
Orchestrator entrypoint (sequential, VRAM-safe).

This provides a single CLI to run phases in order while preserving the
phase-specific scripts and the requested folder structure.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Sana All May Label: sequential phase orchestrator.")
    p.add_argument("--phase1-url", default=None, help="Run Phase 1 with this URL.")
    p.add_argument("--phase1-out", default=None, help="Optional: Phase 1 output directory.")
    p.add_argument("--phase1-json", default=None, help="Path to Phase 1 extracted_metadata.json for Phase 2.")
    p.add_argument("--phase2-out", default=None, help="Optional: Phase 2 output directory.")
    p.add_argument("--phase2-csv", default=None, help="Path to Phase 2 cleaned CSV for Phase 3/4.")
    p.add_argument("--phase3-dir", default=None, help="Path to Phase 3 run dir (pngs) for Phase 4.")
    args = p.parse_args()

    # Note: This is a lightweight orchestrator. Your UI can call each phase directly.
    if args.phase1_url:
        cmd = [sys.executable, str(Path("backend") / "agents" / "scout.py"), args.phase1_url]
        if args.phase1_out:
            cmd += ["--out-dir", args.phase1_out]
        rc = _run(cmd)
        if rc != 0:
            return rc

    if args.phase1_json:
        cmd = [sys.executable, str(Path("backend") / "agents" / "labeler.py"), args.phase1_json]
        if args.phase2_out:
            cmd += ["--out-dir", args.phase2_out]
        rc = _run(cmd)
        if rc != 0:
            return rc

    if args.phase2_csv:
        cmd = [sys.executable, str(Path("backend") / "agents" / "artist.py"), args.phase2_csv]
        rc = _run(cmd)
        if rc != 0:
            return rc

    if args.phase2_csv and args.phase3_dir:
        cmd = [
            sys.executable,
            str(Path("backend") / "agents" / "analyst.py"),
            args.phase2_csv,
            "--phase3-dir",
            args.phase3_dir,
        ]
        rc = _run(cmd)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

