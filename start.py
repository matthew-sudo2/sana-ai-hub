from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
BACKEND_DIR = ROOT / "backend"
STATE_DIR = ROOT / ".sana"
PIDS_PATH = STATE_DIR / "pids.json"

def _load_backend_env_into_process() -> None:
    """
    Minimal .env loader so `start.py` can honor backend/.env without requiring python-dotenv.
    Only sets variables that are not already set in the process environment.
    """
    for env_name in (".env", ".env.example"):
        env_path = BACKEND_DIR / env_name
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and (k not in os.environ or not os.environ.get(k, "").strip()):
                os.environ[k] = v


REQUIRED_OLLAMA_MODELS = [
    os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    os.getenv("OLLAMA_PHASE2_MODEL", "qwen2.5-coder:3b"),
    os.getenv("OLLAMA_PHASE3_MODEL", "qwen2.5-coder:3b"),
    os.getenv("OLLAMA_PHASE4_MODEL", "llama3:8b"),
]


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _read_env_key() -> str | None:
    for env_name in (".env", ".env.example"):
        env_path = BACKEND_DIR / env_name
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SPIDER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def preflight_python_imports() -> list[str]:
    missing: list[str] = []
    required = [
        ("spider", "spider_client"),
        ("ollama", "ollama"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("PIL", "pillow"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "pydantic"),
    ]
    for module, pip_name in required:
        try:
            __import__(module)
        except Exception:
            missing.append(pip_name)
    return missing


def preflight_node_tools() -> list[str]:
    missing: list[str] = []
    if _which("node") is None:
        missing.append("node")
    if _which("npm") is None:
        missing.append("npm")
    return missing


def preflight_spider_key() -> str | None:
    key = os.getenv("SPIDER_API_KEY") or _read_env_key()
    if not key or key == "REPLACE_ME":
        return None
    return key


def ollama_list_models() -> set[str]:
    if _which("ollama") is None:
        raise RuntimeError("Ollama CLI not found in PATH. Install Ollama and ensure `ollama` is available.")
    cp = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Ollama not reachable. Start Ollama, then retry.\n{cp.stderr.strip()}")

    models: set[str] = set()
    # Format typically: NAME  ID  SIZE  MODIFIED
    for line in cp.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0].strip())
    return models


def _model_is_satisfied(required: str, installed: set[str]) -> bool:
    """
    Treat `base:tag` as satisfied if an installed model exists with the same base name.
    Example: required `llama3.2:3b` is satisfied by installed `llama3.2:latest`.
    """
    if required in installed:
        return True
    base = required.split(":", 1)[0]
    return any(m == base or m.startswith(base + ":") for m in installed)


def ollama_pull(model: str) -> None:
    print(f"[preflight] Pulling Ollama model: {model}")
    subprocess.check_call(["ollama", "pull", model])


def ensure_frontend_deps(auto_install: bool) -> None:
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        return
    if not auto_install:
        raise RuntimeError("Frontend dependencies not installed. Run `cd frontend && npm install` (or rerun start.py --install).")
    print("[preflight] Installing frontend dependencies (npm install)")
    npm_exe = "npm.cmd" if os.name == "nt" and _which("npm.cmd") else "npm"
    subprocess.check_call([npm_exe, "install"], cwd=str(FRONTEND_DIR))


def start_frontend() -> int:
    """Start frontend dev server on fixed port 8080."""
    # New process group allows reliable stop via taskkill /T on Windows.
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    npm_exe = "npm.cmd" if os.name == "nt" and _which("npm.cmd") else "npm"

    # Pass port explicitly to ensure 8080
    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = "http://localhost:8000"
    
    proc = subprocess.Popen(
        [npm_exe, "run", "dev", "--", "--port", "8080"],
        cwd=str(FRONTEND_DIR),
        creationflags=creationflags,
        env=env,
    )
    return int(proc.pid)


def start_backend() -> int:
    """Start backend API server on fixed port 8000."""
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    # Load environment variables
    env = os.environ.copy()
    for env_name in (".env", ".env.example"):
        env_path = BACKEND_DIR / env_name
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    env[k] = v

    # Use venv Python executable if available, otherwise fall back to sys.executable
    python_exe = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    proc = subprocess.Popen(
        [str(python_exe), "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT),
        creationflags=creationflags,
        env=env,
    )
    return int(proc.pid)


def write_pids(frontend_pid: int, backend_pid: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "frontend_pid": frontend_pid,
        "backend_pid": backend_pid,
        "started_at_unix": int(time.time())
    }
    PIDS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Sana All May Label (preflight + backend + frontend).")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install missing frontend dependencies.",
    )
    parser.add_argument("--pull-models", action="store_true", help="Auto-pull missing Ollama models.")
    args = parser.parse_args()

    if not FRONTEND_DIR.exists():
        raise RuntimeError("Missing ./frontend directory.")
    if not BACKEND_DIR.exists():
        raise RuntimeError("Missing ./backend directory.")

    # Kill any existing processes on ports 8000-8090 first
    print("[startup] Cleaning up any existing processes...")
    for port in range(8000, 8091):
        try:
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}"' if os.name == "nt" else f"lsof -i :{port}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.stdout:
                try:
                    parts = result.stdout.split()
                    pid = int(parts[-1]) if parts else 0
                    if pid > 0:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt" else ["kill", "-9", str(pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                except Exception:
                    pass
        except Exception:
            pass
    
    time.sleep(1)  # Wait for ports to be released

    # Load backend/.env so model names + keys are visible to this process.
    _load_backend_env_into_process()

    node_missing = preflight_node_tools()
    if node_missing:
        raise RuntimeError(f"Missing required tools: {', '.join(node_missing)}. Install Node.js (includes npm) and retry.")

    py_missing = preflight_python_imports()
    if py_missing:
        msg = (
            "Missing Python packages for backend phases: "
            + ", ".join(py_missing)
            + "\nInstall with: pip install -r backend/requirements.txt"
        )
        raise RuntimeError(msg)

    if preflight_spider_key() is None:
        raise RuntimeError(
            "Missing SPIDER_API_KEY. Set it via environment variable, or create backend/.env "
            "from backend/.env.example and set SPIDER_API_KEY, then retry."
        )

    installed_models = ollama_list_models()
    required_models = sorted(set(m for m in REQUIRED_OLLAMA_MODELS if m))
    missing_models = [m for m in required_models if not _model_is_satisfied(m, installed_models)]
    if missing_models:
        if not args.pull_models:
            raise RuntimeError(
                "Missing required Ollama models: "
                + ", ".join(missing_models)
                + "\nPull them with: ollama pull <model>\nOr rerun: python start.py --pull-models"
            )
        for m in missing_models:
            ollama_pull(m)

    ensure_frontend_deps(auto_install=not args.no_install)

    # Start backend first (it's a dependency for frontend)
    print("[startup] Starting backend API on localhost:8000...")
    backend_pid = start_backend()
    time.sleep(2)  # Give backend time to start
    print(f"[ok] Backend started (pid={backend_pid}) on http://localhost:8000")

    # Start frontend
    print("[startup] Starting frontend on localhost:8080...")
    frontend_pid = start_frontend()
    time.sleep(2)  # Give frontend time to start
    print(f"[ok] Frontend started (pid={frontend_pid}) on http://localhost:8080")
    
    write_pids(frontend_pid, backend_pid)
    print("[ok] All services running. To stop: python stop.py")
    print(f"[ok] Dashboard: http://localhost:8080/dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

