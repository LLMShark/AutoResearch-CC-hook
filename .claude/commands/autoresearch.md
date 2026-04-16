# AutoResearch — Init / Resume / Run Optimization Loop

Single entry point for the whole loop: initialize a new task, resume an existing
one, or kick off the optimization. The hook machine takes it from there.

## Arguments

`$ARGUMENTS` — one of:

- **`--resume`** or **`--resume <task_dir>`** — continue the most recent task
  (or the specified one).
- **Task dir** — resume that specific task: `ar_tasks/my_task_123456_abc`.
- **Init flags** — new task from an existing reference file:
  `--ref <file> --op-name <name> --backend ascend|cuda|cpu [--arch <arch>]
  [--kernel <file>] [--worker-url <host:port>] [--max-rounds <N>]
  [--dsl <name>] [--framework <name>]`
- **Desc mode** — new task from a natural-language description:
  `--desc "fused ReLU + LayerNorm, (32,1024), fp16" --backend cuda`

Required init flags: `--ref` (or `--desc`) and `--op-name`. `--output-dir`
defaults to `ar_tasks`.

## Step 1: Decide path

1. `$ARGUMENTS` contains `--resume` → resume most recent (or given) task:
   ```bash
   python .autoresearch/scripts/resume.py [optional_task_dir]
   ```
   The last line of stdout is the task_dir. Non-zero exit ⇒ stop and report
   (likely an incompatible on-disk version).

2. `$ARGUMENTS` is an existing directory → resume it:
   ```bash
   python .autoresearch/scripts/resume.py "$ARGUMENTS"
   ```

3. `$ARGUMENTS` starts with `--` (and is not `--resume`) → scaffold a new task:
   ```bash
   python .autoresearch/scripts/scaffold.py $ARGUMENTS --output-dir ar_tasks --run-baseline
   ```
   `--run-baseline` runs the baseline eval immediately AND writes
   `.ar_state/.phase = PLAN` on success, so when **both `--ref` and `--kernel`
   are provided** there are no user-visible init/baseline steps: the next
   activation drops you straight into PLAN. (`--desc` mode and `--ref` without
   `--kernel` will instead start in GENERATE_REF / GENERATE_KERNEL.) Read the
   `task_dir` from the JSON output.

4. No arguments → ask the user: reference path, op name, backend, worker URL,
   max rounds. Then use path 3.

## Step 2: Activate

```bash
export AR_TASK_DIR="<task_dir from step 1>"
```

The activation hook prints `[AR Phase: ...]` guidance. Follow it.

## Step 3: Loop

Follow the phase guidance. Never stop between phases.

- **GENERATE_REF / GENERATE_KERNEL** — Write `reference.py` / `kernel.py` with
  the Edit tool (only needed for `--desc` mode or when you skipped `--kernel`).
- **BASELINE** — `python .autoresearch/scripts/baseline.py "$AR_TASK_DIR"`
  (append `--worker-url` if configured). If scaffold already ran baseline,
  this phase is skipped automatically.
- **PLAN / DIAGNOSE / REPLAN** —
  `python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR" '<JSON>'`.
  When the hook's `additionalContext` gives you a TodoWrite payload, call
  TodoWrite with it verbatim.
- **EDIT** — Edit `kernel.py` (multiple Edit calls OK). When done:
  `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`.
- **FINISH** — Write `.ar_state/ranking.md`, summarize, stop.

## Rules

- Keep going between phases.
- Hooks block wrong actions and tell you what to do next — read their messages.
- Never hand-edit `plan.md` or `.ar_state/.phase`; always go through the scripts.
