#!/usr/bin/env python3
"""
PreToolUse hook for Edit/Write — phase-aware edit gating.

Phase table:
  GENERATE_REF:    only reference.py
  GENERATE_KERNEL: only editable_files (typically kernel.py)
  EDIT:            only editable_files
  Others:          BLOCK ALL edits

plan.md is machine-generated (create_plan.py / settle.py / pipeline.py) and
is NEVER a legal target for the Edit/Write tool — the block fires in every
phase to enforce that invariant.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input
from phase_machine import (
    read_phase, get_guidance, get_task_dir, touch_heartbeat,
    edit_marker_path, CODE_EDIT_PHASES, REF_WRITE_PHASES,
)


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def main():
    hook_input = read_hook_input()
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name not in ("Edit", "Write"):
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = tool_input.get("file_path", "")
    if not file_path:
        sys.exit(0)

    phase = read_phase(task_dir)

    # Normalize paths
    file_path_norm = os.path.normpath(os.path.abspath(file_path)).replace("\\", "/")
    task_dir_norm = os.path.normpath(os.path.abspath(task_dir)).replace("\\", "/")

    # Files outside task_dir: allow (not our concern)
    if not file_path_norm.startswith(task_dir_norm):
        sys.exit(0)

    rel = os.path.relpath(file_path, task_dir).replace("\\", "/")

    # Classify the file
    is_plan = rel == ".ar_state/plan.md"
    is_ref = rel == "reference.py"

    is_editable = False
    try:
        from task_config import load_task_config
        config = load_task_config(task_dir)
        if config:
            is_editable = rel in set(config.editable_files)
    except Exception:
        pass

    # --- Phase gating ---
    guidance = get_guidance(task_dir)

    if is_plan:
        _block(
            "[AR] plan.md is machine-generated — never hand-edit it. "
            "Use `python .autoresearch/scripts/create_plan.py \"<task_dir>\" '<items_json>'` "
            "to propose a new plan. settle.py/pipeline.py handle per-round updates."
        )

    elif is_ref:
        if phase in REF_WRITE_PHASES:
            sys.exit(0)
        else:
            _block(f"[AR] Cannot write reference.py in phase {phase}. {guidance}")

    elif is_editable:
        if phase not in CODE_EDIT_PHASES:
            _block(f"[AR] Cannot edit code in phase {phase}. {guidance}")

        # In EDIT phase: if there's already uncommitted code (from a previous
        # edit session), block further edits until pipeline runs.
        # Exception: GENERATE_KERNEL phase allows multiple edits freely.
        if phase == "EDIT":
            import subprocess
            try:
                repo_root = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=task_dir, capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                for ef in (config.editable_files if config else []):
                    rel_in_repo = os.path.relpath(os.path.join(task_dir, ef), repo_root)
                    # Check if this specific file has uncommitted changes AND
                    # the file being edited is a DIFFERENT file
                    diff = subprocess.run(
                        ["git", "diff", "--name-only", "--", rel_in_repo],
                        cwd=repo_root, capture_output=True, text=True, timeout=5,
                    )
                    if diff.stdout.strip():
                        # There are uncommitted changes
                        # But we still want to allow re-editing the SAME file
                        # (Claude often edits one file multiple times)
                        # So we check: is this a fresh round, OR continuing same round?
                        # Heuristic: uncommitted diff from previous round means
                        # pipeline wasn't run. Block to force pipeline.
                        # We track this via a marker file: .ar_state/.edit_started
                        marker = edit_marker_path(task_dir)
                        if not os.path.exists(marker):
                            # Uncommitted diff exists but no edit marker = previous
                            # round's leftover. Force pipeline.
                            _block(
                                f"[AR] Uncommitted changes from previous round detected. "
                                f"Run pipeline.py to finalize before editing: "
                                f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
                            )
                        break
                # Create marker so subsequent edits in same round are allowed
                marker = edit_marker_path(task_dir)
                os.makedirs(os.path.dirname(marker), exist_ok=True)
                with open(marker, "w") as f:
                    f.write("1")
            except Exception:
                pass  # don't block on git errors

        sys.exit(0)

    else:
        # Other files in task_dir
        if rel.startswith(".ar_state/"):
            sys.exit(0)  # Allow hook-internal files
        _block(f"[AR] Cannot write '{rel}' — not an editable file or plan.md. {guidance}")


if __name__ == "__main__":
    main()
