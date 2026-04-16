#!/usr/bin/env python3
"""
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

The only commands that advance phase from this hook are those Claude runs
directly via the Bash tool:
  - `export AR_TASK_DIR=...` → activate task, compute starting phase
  - `baseline.py`             → PLAN (on success)
  - `pipeline.py`              → whatever phase pipeline.py itself wrote
  - `create_plan.py`           → EDIT (on plan validation pass)

The inner pipeline steps (quick_check / eval_wrapper / keep_or_discard /
settle) are subprocess children of pipeline.py and never re-enter this hook,
so they don't need their own phase constants or branches here.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, emit_todowrite_context
from phase_machine import (
    read_phase, write_phase, get_guidance, compute_resume_phase,
    get_task_dir, set_task_dir, get_active_item, touch_heartbeat,
    load_progress, update_progress,
    progress_path, history_path, plan_path, edit_marker_path, state_path,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, GENERATE_REF, GENERATE_KERNEL,
)


def _activation_target(command: str) -> str | None:
    if "AR_TASK_DIR=" not in command:
        return None
    m = re.search(r'AR_TASK_DIR=["\']?([^"\';\s&]+)', command)
    return m.group(1) if m else None


def _clean_stale_edit_marker(task_dir: str):
    """Remove .edit_started if git is clean (nothing to resume)."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    try:
        import subprocess as _sp
        diff = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=task_dir, capture_output=True, text=True, timeout=5,
        )
        if not diff.stdout.strip():
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
    except Exception:
        pass


def _handle_activation(new_task_dir: str):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    set_task_dir(new_task_dir)
    _clean_stale_edit_marker(new_task_dir)

    has_phase = os.path.exists(state_path(new_task_dir, ".phase"))
    has_progress = os.path.exists(progress_path(new_task_dir))

    if has_phase:
        phase = read_phase(new_task_dir)
        emit_status(f"[AR] Resuming. Phase: {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    elif has_progress:
        phase = compute_resume_phase(new_task_dir)
        write_phase(new_task_dir, phase)
        emit_status(f"[AR] Resuming from progress. Phase -> {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    else:
        _fresh_start(new_task_dir)


def _fresh_start(task_dir: str):
    """Pick initial phase for a fresh task based on which files are present."""
    def _real(path: str, needle: str = "") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            content = f.read()
        if "TODO" in content:
            return False
        return (len(content) > 50) and (needle in content if needle else True)

    ref_ok = _real(os.path.join(task_dir, "reference.py"), "class Model")
    kernel_ok = _real(os.path.join(task_dir, "kernel.py"))

    if not ref_ok:
        write_phase(task_dir, GENERATE_REF)
        emit_status(f"[AR] Fresh start (no reference). Phase -> GENERATE_REF. {get_guidance(task_dir)}")
    elif not kernel_ok:
        write_phase(task_dir, GENERATE_KERNEL)
        emit_status(f"[AR] Fresh start (no kernel). Phase -> GENERATE_KERNEL. {get_guidance(task_dir)}")
    else:
        write_phase(task_dir, BASELINE)
        emit_status(f"[AR] Fresh start. Phase -> BASELINE. {get_guidance(task_dir)}")


def _progress_update_for_plan(task_dir: str, phase: str):
    """Set status=active after a valid new plan. `plan_version` is owned and
    bumped by create_plan.py — this hook must NOT re-bump it (caused double
    increments that jumped plan_version by 2 each REPLAN)."""
    fields = {"status": "active"}
    if phase == DIAGNOSE:
        fields["consecutive_failures"] = 0
    update_progress(task_dir, **fields)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)

    if "baseline.py" in command and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            emit_status("[AR] Baseline failed (no progress.json). Retry.")
        elif progress.get("seed_metric") is None:
            emit_status(
                "[AR] Baseline profiled NO timing for the seed kernel. "
                "Fix kernel.py (see worker log for compile/runtime error) and "
                f"re-run: python .autoresearch/scripts/baseline.py \"{task_dir}\""
            )
        else:
            write_phase(task_dir, PLAN)
            emit_status(f"[AR] Baseline complete. Phase -> PLAN. {get_guidance(task_dir)}")

    elif "pipeline.py" in command:
        # pipeline.py writes .phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif "create_plan.py" in command and phase in (PLAN, DIAGNOSE, REPLAN):
        from phase_machine import validate_plan
        ok, err = validate_plan(task_dir)
        if ok:
            _progress_update_for_plan(task_dir, phase)
            write_phase(task_dir, EDIT)
            emit_status(f"[AR] Plan validated. Phase -> EDIT. {get_guidance(task_dir)}")
            emit_todowrite_context(task_dir, "[AR] Plan validated. Phase -> EDIT.")
        else:
            emit_status(f"[AR] Plan not valid yet: {err}")

    sys.exit(0)


def _print_resume_context(task_dir: str):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            emit_status(f"[AR] Active item: {active['id']}: {active['description'][:50]}")
        emit_status("[AR] Read .ar_state/plan.md and .ar_state/history.jsonl for full context.")


if __name__ == "__main__":
    main()
