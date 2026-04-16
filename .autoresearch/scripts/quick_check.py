#!/usr/bin/env python3
"""
Quick syntax check for editable files.

Zero AKG dependency — uses local task_config.py.

Usage:
    python .autoresearch/scripts/quick_check.py <task_dir>

Output:
    stdout: "OK" or JSON error details
    exit code: 0 = pass, 1 = fail
"""

import argparse
import json
import os
import py_compile
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config


def main():
    parser = argparse.ArgumentParser(description="Quick syntax check")
    parser.add_argument("task_dir", help="Path to task directory")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({"ok": False, "error": "task.yaml not found"}))
        sys.exit(1)

    errors = []

    for fname in config.editable_files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            errors.append(f"{fname}: file not found")
            continue
        try:
            py_compile.compile(fpath, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{fname}: {e}")

    if not errors and config.smoke_test_script:
        smoke_path = os.path.join(task_dir, config.smoke_test_script)
        if os.path.exists(smoke_path):
            try:
                result = subprocess.run(
                    [sys.executable, smoke_path],
                    capture_output=True, text=True,
                    timeout=config.smoke_test_timeout,
                    cwd=task_dir,
                )
                if result.returncode != 0:
                    stderr_tail = result.stderr[-500:] if result.stderr else ""
                    errors.append(f"smoke test failed (exit {result.returncode}): {stderr_tail}")
            except subprocess.TimeoutExpired:
                errors.append(f"smoke test timed out after {config.smoke_test_timeout}s")
            except Exception as e:
                errors.append(f"smoke test error: {e}")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}))
        sys.exit(1)
    else:
        print("OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
