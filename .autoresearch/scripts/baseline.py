#!/usr/bin/env python3
"""Run baseline eval and initialize .ar_state.

Python replacement for baseline.sh — avoids bash-on-Windows path mangling.

Usage:
    python .autoresearch/scripts/baseline.py <task_dir> [--device-id N] [--worker-url URL]
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from phase_machine import parse_last_json_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--worker-url", default=None)
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    extra = []
    if args.device_id is not None:
        extra += ["--device-id", str(args.device_id)]
    if args.worker_url:
        extra += ["--worker-url", args.worker_url]

    print("[baseline] Running baseline eval...", flush=True)
    ev = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "eval_wrapper.py"), task_dir] + extra,
        capture_output=True, text=True,
    )
    if ev.stdout:
        print(ev.stdout, end="", flush=True)
    if ev.stderr:
        print(ev.stderr, end="", file=sys.stderr, flush=True)
    if ev.returncode != 0:
        print(f"[baseline] eval_wrapper failed (rc={ev.returncode})", file=sys.stderr)
        sys.exit(ev.returncode)

    eval_data = parse_last_json_line(ev.stdout)
    if eval_data is None:
        print("[baseline] ERROR: no JSON output from eval_wrapper", file=sys.stderr)
        sys.exit(1)

    rc = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "_baseline_init.py"),
         task_dir, json.dumps(eval_data)],
    ).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
