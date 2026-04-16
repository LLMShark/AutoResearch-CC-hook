#!/usr/bin/env python3
"""
Eval wrapper for Claude Code autoresearch.

Zero AKG dependency — uses local task_config.py for YAML parsing and eval execution.

Usage:
    python .autoresearch/scripts/eval_wrapper.py <task_dir> [--device-id N] [--worker-url URL,...]

Output (last line of stdout):
    {"correctness": true, "metrics": {"latency_us": 145.3}, "error": null}
"""

import argparse
import json
import os
import sys

# Import from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config, run_eval, format_result_summary


def main():
    parser = argparse.ArgumentParser(description="Run eval and output JSON")
    parser.add_argument("task_dir", help="Path to the task directory")
    parser.add_argument("--device-id", type=int, default=None, help="Device ID (local eval)")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated. Overrides task.yaml worker.urls.")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)

    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({
            "correctness": False,
            "metrics": {},
            "error": f"task.yaml not found in {task_dir}",
        }))
        sys.exit(0)

    # CLI --worker-url overrides task.yaml
    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    mode = "remote" if (worker_urls or config.worker_urls) else "local"
    print(f"[eval] Running {mode} eval for {config.name}...", file=sys.stderr)

    result = run_eval(task_dir, config, device_id=args.device_id, worker_urls=worker_urls)

    print(f"[eval] {format_result_summary(result)}", file=sys.stderr)

    output = {
        "correctness": result.correctness,
        "metrics": result.metrics,
        "error": result.error,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
