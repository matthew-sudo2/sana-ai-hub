from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIDS_PATH = ROOT / ".sana" / "pids.json"


def _taskkill(pid: int) -> None:
    """Force kill a process and its children."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        pass


def _kill_all_ports_single_cmd() -> None:
    """Kill all processes on main ports using a single netstat command."""
    try:
        # Run netstat once and find all processes on our ports
        result = subprocess.run(
            "netstat -ano",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        
        killed_pids = set()
        for line in result.stdout.splitlines():
            # Look for lines with ports 8000-8090
            if any(f":{port} " in line for port in range(8000, 8091)):
                parts = line.split()
                if len(parts) > 0:
                    try:
                        pid = int(parts[-1])
                        if pid > 0 and pid not in killed_pids:
                            _taskkill(pid)
                            killed_pids.add(pid)
                            port_str = line.split()[1] if len(line.split()) > 1 else "unknown"
                            print(f"[ok] Killed process (pid={pid}) on {port_str}.")
                    except (ValueError, IndexError):
                        pass
    except subprocess.TimeoutExpired:
        print("[warn] Netstat timeout - some ports may still be in use")
    except Exception as e:
        print(f"[warn] Could not enumerate ports: {e}")


def main() -> int:
    print("[cleanup] Stopping all services...")
    
    # Kill from PID file first (most reliable)
    if PIDS_PATH.exists():
        try:
            payload = json.loads(PIDS_PATH.read_text(encoding="utf-8"))
            frontend_pid = payload.get("frontend_pid")
            backend_pid = payload.get("backend_pid")

            if isinstance(frontend_pid, int) and frontend_pid > 0:
                _taskkill(frontend_pid)
                print(f"[ok] Stopped frontend (pid={frontend_pid}).")

            if isinstance(backend_pid, int) and backend_pid > 0:
                _taskkill(backend_pid)
                print(f"[ok] Stopped backend (pid={backend_pid}).")

            try:
                PIDS_PATH.unlink()
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] Could not read PID file: {e}")
    
    # Try single netstat call to catch any remaining processes
    print("[cleanup] Checking for orphaned processes on ports 8000-8090...")
    time.sleep(1)  # Give processes time to die
    _kill_all_ports_single_cmd()

    print("[ok] All services stopped. Ports 8000-8090 should be free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


