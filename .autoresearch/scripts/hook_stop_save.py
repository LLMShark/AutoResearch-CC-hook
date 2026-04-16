#!/usr/bin/env python3
"""
Stop hook: When Claude Code stops (context limit, user interrupt, etc.),
stamp the reason into progress.json and print a final status summary.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status
from phase_machine import get_task_dir, load_progress, update_progress


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    if progress is None:
        sys.exit(0)

    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    emit_status(f"\n[AR] Session stopped: {stop_reason}")
    emit_status(f"[AR] Progress: {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume with: /autoresearch {task_dir}")


if __name__ == "__main__":
    main()
