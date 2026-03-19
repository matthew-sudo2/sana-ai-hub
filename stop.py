from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIDS_PATH = ROOT / ".sana" / "pids.json"


def _taskkill(pid: int) -> None:
    # /T kills child processes; /F forces termination (dev servers often hold open handles).
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    if not PIDS_PATH.exists():
        print("[info] No PID file found. Nothing to stop.")
        return 0

    payload = json.loads(PIDS_PATH.read_text(encoding="utf-8"))
    frontend_pid = payload.get("frontend_pid")

    if isinstance(frontend_pid, int) and frontend_pid > 0:
        if os.name == "nt":
            _taskkill(frontend_pid)
        else:
            try:
                os.kill(frontend_pid, 15)
            except Exception:
                pass
        print(f"[ok] Stop signal sent to frontend (pid={frontend_pid}).")

    try:
        PIDS_PATH.unlink()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

