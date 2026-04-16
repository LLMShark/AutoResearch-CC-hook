#!/usr/bin/env python3
"""
PostToolUse hook for Edit/Write — advances phase after code edits.

- reference.py in GENERATE_REF → GENERATE_KERNEL or BASELINE (depending on
  whether kernel.py is still a placeholder)
- editable file in GENERATE_KERNEL → BASELINE
- editable file in EDIT → no phase change; Claude runs pipeline.py when done

plan.md is never a legal target for Edit/Write — hook_guard_edit blocks it
at every phase and directs Claude to create_plan.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status
from phase_machine import (
    read_phase, write_phase, get_guidance, _load_config_safe,
    get_task_dir, touch_heartbeat,
    EDIT, BASELINE, GENERATE_REF, GENERATE_KERNEL,
)


def _same_path(a: str, b: str) -> bool:
    norm = lambda p: os.path.normpath(os.path.abspath(p)).replace("\\", "/")
    return norm(a) == norm(b)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in ("Edit", "Write"):
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = hook_input.get("tool_input", {}).get("file_path", "")
    if not file_path:
        sys.exit(0)

    phase = read_phase(task_dir)
    is_ref = _same_path(file_path, os.path.join(task_dir, "reference.py"))

    config = _load_config_safe(task_dir)
    is_editable = False
    if config:
        try:
            rel = os.path.relpath(file_path, task_dir).replace("\\", "/")
            is_editable = rel in set(config.editable_files)
        except ValueError:
            is_editable = False

    if is_ref and phase == GENERATE_REF:
        kernel = os.path.join(task_dir, "kernel.py")
        placeholder = True
        if os.path.exists(kernel):
            with open(kernel, "r") as f:
                content = f.read()
            placeholder = "TODO" in content or len(content) < 50
        next_phase = GENERATE_KERNEL if placeholder else BASELINE
        write_phase(task_dir, next_phase)
        emit_status(f"[AR] Reference written. Phase -> {next_phase}. {get_guidance(task_dir)}")

    elif is_editable and phase == GENERATE_KERNEL:
        write_phase(task_dir, BASELINE)
        emit_status(f"[AR] Kernel generated. Phase -> BASELINE. {get_guidance(task_dir)}")

    elif is_editable and phase == EDIT:
        emit_status(
            f"[AR] Code edited. Continue editing OR run: "
            f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
