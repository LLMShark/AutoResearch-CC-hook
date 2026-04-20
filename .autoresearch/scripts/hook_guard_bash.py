#!/usr/bin/env python3
"""
PreToolUse hook for Bash — phase-aware command gating.

Phase table:
  INIT:      export AR_TASK_DIR=* only
  BASELINE:  baseline.py only
  EDIT:      code edits + pipeline.py (individual step scripts blocked)
  PLAN /
  DIAGNOSE /
  REPLAN:    create_plan.py only (+read-only queries)
  All:       dashboard.py, git log/diff/status/show/branch, ls, cat, head, tail

Also enforces a script whitelist — blocks hallucinated script names.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input
from phase_machine import (
    read_phase, get_guidance, get_task_dir, touch_heartbeat,
    INIT, BASELINE, EDIT, PLAN, DIAGNOSE, REPLAN,
)

# Scripts that exist
_BLESSED_SCRIPTS = {
    "quick_check.py", "eval_wrapper.py", "keep_or_discard.py",
    "scaffold.py", "baseline.py", "_baseline_init.py", "dashboard.py",
    "create_plan.py", "settle.py", "pipeline.py", "resume.py",
    "reference_capture.py", "code_checker.py",
}

# Commonly hallucinated names → suggestions
_BANNED_SCRIPTS = {
    "eval.py": "eval_wrapper.py",
    "run_eval.py": "eval_wrapper.py",
    "verify.py": "eval_wrapper.py",
    "check.py": "quick_check.py",
    "run.py": "eval_wrapper.py",
    "test.py": "quick_check.py",
    "profile.py": "eval_wrapper.py",
}

# Read-only commands always allowed
_READONLY_PATTERNS = [
    r"^(ls|cat|head|tail|wc|find|grep|git\s+(log|diff|status|show|branch))",
    r"dashboard\.py",
    r"^echo\s",
    r"^pwd$",
]


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def _is_readonly(command: str) -> bool:
    for pattern in _READONLY_PATTERNS:
        if re.search(pattern, command.strip()):
            return True
    return False


def main():
    hook_input = read_hook_input()
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name != "Bash":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    command = tool_input.get("command", "")
    phase = read_phase(task_dir)
    guidance = get_guidance(task_dir)

    # Always allow read-only commands
    if _is_readonly(command):
        sys.exit(0)

    # --- Script whitelist ---
    m = re.search(r'python\s+["\']?([^\s"\']+\.py)', command)
    if m:
        script_path = m.group(1).replace("\\", "/")
        script_name = os.path.basename(script_path)

        if ".autoresearch/scripts/" in script_path and script_name not in _BLESSED_SCRIPTS:
            _block(f"[AR] Unknown script: '{script_name}'. "
                   f"Valid scripts: {sorted(_BLESSED_SCRIPTS)}")

        if script_name in _BANNED_SCRIPTS:
            suggestion = _BANNED_SCRIPTS[script_name]
            _block(f"[AR] '{script_name}' does not exist. "
                   f"Use: python .autoresearch/scripts/{suggestion}")

    # --- Phase gating ---
    is_export = command.strip().startswith("export AR_TASK_DIR=")
    is_baseline = "baseline.py" in command
    is_quick_check = "quick_check.py" in command
    is_eval = "eval_wrapper.py" in command
    is_keep_discard = "keep_or_discard.py" in command
    is_git_commit = "git commit" in command

    # Block manual git commit always
    if is_git_commit:
        _block("[AR] Manual 'git commit' forbidden. "
               "Use python .autoresearch/scripts/keep_or_discard.py")

    if phase == INIT:
        if is_export:
            sys.exit(0)
        _block(f"[AR] Phase INIT — only 'export AR_TASK_DIR=...' allowed. {guidance}")

    elif phase == BASELINE:
        if is_baseline:
            sys.exit(0)
        _block(f"[AR] Phase BASELINE — only baseline.py allowed. {guidance}")

    elif phase == EDIT:
        # EDIT: allow multiple edits + pipeline.py to finalize the round
        if "pipeline.py" in command:
            sys.exit(0)
        # Block individual step scripts (pipeline.py wraps them)
        if is_quick_check or is_eval or is_keep_discard:
            _block(f"[AR] Phase EDIT — use pipeline.py instead of individual scripts. {guidance}")
        # Block create_plan in EDIT phase — Claude must finish current item via pipeline first.
        # Replanning is only allowed in DIAGNOSE/REPLAN phase (after items settled or failures trigger).
        if "create_plan.py" in command:
            _block(f"[AR] Cannot create new plan in EDIT phase. Finish current item first "
                   f"by running pipeline.py. Replanning happens automatically when all items "
                   f"settle (REPLAN) or after 3 consecutive failures (DIAGNOSE).")
        sys.exit(0)

    else:
        # PLAN / DIAGNOSE / REPLAN / FINISH / GENERATE_* — allow create_plan.py
        # in plan-writing phases, block the pipeline inner scripts everywhere
        # (they're subprocess-only, never user-facing).
        if "create_plan.py" in command and phase in (PLAN, DIAGNOSE, REPLAN):
            sys.exit(0)
        if is_eval or is_keep_discard or is_quick_check:
            _block(f"[AR] Phase {phase} — eval/check scripts are subprocess-only. {guidance}")
        sys.exit(0)


if __name__ == "__main__":
    main()
