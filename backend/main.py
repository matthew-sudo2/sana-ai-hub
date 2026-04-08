from __future__ import annotations

"""CLI orchestrator for the 5-agent flow.

Flow order:
1) Scout (acquisition)
2) Labeler (preparation)
3) Analyst (analysis)
4) Artist (visualization)
5) Validator (sanity + confidence)
"""

import argparse
import json

try:
    from .graph import create_run, execute_run, get_run
except ImportError:
    from backend.graph import create_run, execute_run, get_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full 5-agent pipeline.")
    parser.add_argument(
        "source",
        help="Input source for Scout: URL, CSV/XLSX/JSON path, or raw text",
    )
    args = parser.parse_args()

    run_id = create_run(args.source)
    execute_run(run_id)

    state = get_run(run_id)
    if state is None:
        raise RuntimeError("Pipeline state not found after execution")

    # Handle both dict and object returns from get_run()
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    output = {
        "run_id": _get(state, "run_id"),
        "phase": _get(state, "phase"),
        "status": _get(state, "status"),
        "error": _get(state, "error"),
        "run_dir": _get(state, "run_dir"),
        "scout_output_dir": _get(state, "scout_output_dir"),
        "labeler_output_dir": _get(state, "labeler_output_dir"),
        "analyst_output_dir": _get(state, "analyst_output_dir"),
        "artist_output_dir": _get(state, "artist_output_dir"),
        "validator_output_dir": _get(state, "validator_output_dir"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if _get(state, "status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

