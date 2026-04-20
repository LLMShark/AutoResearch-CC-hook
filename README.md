# Claude AutoResearch

Claude Code 驱动的算子迭代优化框架。纯 Python + PyYAML。

Claude 做需要 LLM 判断的事：读代码、写 plan、改 kernel、诊断失败。
Hooks 做确定性的事：阶段转移、plan 校验、eval 调度、KEEP/DISCARD、回滚。

## 快速开始

```bash
cd claude-autoresearch
claude
```

一行命令启动。`scaffold` + 第一次 `baseline` 会原子完成，然后直接进入 PLAN：

```
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --backend ascend --arch ascend910b2 \
  --worker-url 127.0.0.1:9002 --max-rounds 200
```

候选 ref / kernel 源文件统一放在 [workspace/](workspace/) 下，文件名是
`<op_name>_ref.py` / `<op_name>_kernel.py`。例如
[workspace/sinkhorn_ref.py](workspace/sinkhorn_ref.py)、
[workspace/sinkhorn_kernel.py](workspace/sinkhorn_kernel.py)。新增算子按这个
约定丢进 `workspace/`，`/autoresearch` 用 `--ref workspace/<op>_ref.py
--kernel workspace/<op>_kernel.py` 直接传。

长跑用 `/loop` 自驱模式。失败会自恢复，上下文满了也会延续：

```
/loop /autoresearch --resume
```

另开终端看 dashboard：

```bash
python .autoresearch/scripts/dashboard.py
```

省略路径时默认挑最近活跃的 task。

## 四种启动模式

| 模式 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 有 PyTorch ref 和种子 kernel | 直接 PLAN |
| `--ref X.py` | 只有 ref，kernel 现生 | GENERATE_KERNEL |
| `--desc "..."` | 纯自然语言描述 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 自然语言 + 种子 kernel | GENERATE_REF |

Resume：`/autoresearch --resume [task_dir]`。省略路径时自动挑最近活跃的。

## 单一入口：`/autoresearch`

`/autoresearch` 是项目唯一的 slash command：

- 参数以 `--` 开头：新建任务（scaffold + 第一次 baseline 原子完成）
- 参数是已存在的目录：resume 该目录
- `--resume`：resume 最近活跃任务
- 没参数：交互式询问

连续 3 次 FAIL 后 hook 自动切到 `DIAGNOSE` 阶段。phase_machine 的 guidance
会指引 Claude spawn subagent 做 root-cause 分析。进度看板用 `dashboard.py`，
另一个终端跑。

## 两阶段精度检查

scaffold 时本地 CPU 跑一次 PyTorch `Model`，把输入/输出 dump 成
`.ar_state/reference.pt`。此后每轮 verify：

1. Worker 端解包拿到 `reference.pt`
2. `torch.load` 取出 ref inputs/outputs
3. 只跑 `ModelNew` 的 forward，跟 stored ref 对比
4. 输出 `max_abs / max_rel / bad_elems(%)` 诊断信息

缺 `.pt` 时自动降级为 inline 对比模式。容差在 `task.yaml` 配：

```yaml
metric:
  primary: latency_us
  correctness_atol: 1.0e-2
  correctness_rtol: 1.0e-2
```

好处：每轮省一次 PyTorch forward（200 轮 × 大算子可观）。verify 失败时 ref
时延仍由 `/api/v1/profile` 测出，跟 verify 解耦，dashboard 顶栏始终有 PyTorch
baseline。

## 远程 Worker

Ascend NPU / CUDA 等远端硬件通过 SSH tunnel 接入。**评测算子的部分复用了
akg agent 的 worker 功能**——框架负责打包 + 调用 `/api/v1/verify` 和
`/api/v1/profile`，实际在 NPU / GPU 上跑 verify/profile 的执行器来自 akg
agent。

### 启动远端 worker

```bash
ssh npu 'bash -lc "source /path/to/conda/etc/profile.d/conda.sh && conda activate <env> && \
  cd /path/to/akg_agents && \
  nohup bash scripts/server_related/start_worker_service.sh ascend ascend910b2 4 9002 \
  > /tmp/worker_9002.log 2>&1 < /dev/null &"'
```

参数：`backend arch device_id port`。

### 建本地 tunnel

```bash
ssh -f -N -L 127.0.0.1:9002:127.0.0.1:9002 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu

curl http://127.0.0.1:9002/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b2","devices":[4]}
```

任务启动加 `--worker-url 127.0.0.1:9002`，eval 自动走这台 worker。多台用
逗号分隔，框架自动选可达的。

## 实时监控

```bash
# 自动选当前任务，5 秒刷新
python .autoresearch/scripts/dashboard.py

# 指定任务和刷新间隔
python .autoresearch/scripts/dashboard.py ar_tasks/my_task --watch 2

# 键盘：↑/↓/PgUp/PgDn/Home/End 滚 history，q/Esc 退出
```

顶栏：task 名、阶段、plan 版本、budget、Baseline（PyTorch ref 时延）、Seed
（种子 kernel 时延）、Best、改进比。下栏：history table（每条带 `pN:` 前缀）
和 current plan。

## 流程：对齐 akg autoresearch

整体节奏跟 `akg_agents/python/akg_agents/op/autoresearch/` 一致。单次循环走
**PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle**。连续失败走
DIAGNOSE，plan 全部 settle 走 REPLAN，预算用完落 FINISH。执行体上有差别：
akg 是 agent 框架内嵌的 `update_plan` / `edit` / `acknowledge_skill` 工具
协议；这里换成 Claude Code 的 Edit / Bash 工具 + 一组 Python 脚本 + 一组
Hook 守卫。

```
INIT
  ├─ (--desc?)            GENERATE_REF ─→ GENERATE_KERNEL
  ├─ (--ref only?)                       GENERATE_KERNEL
  └─ (--ref + --kernel?)                            ─────→ BASELINE
                                                            │
                                          ┌─ scaffold --run-baseline 原子完成
                                          ▼
   ┌────────────────────────  PLAN  ◀────────────────────────┐
   │   create_plan.py 校验 (≥3 项 / 多样性 / rationale 长度) │
   ▼                                                          │
  EDIT  ──→  pipeline.py:                                     │
            quick_check → eval_wrapper → keep_or_discard      │
            → settle ──→ history.jsonl + plan.md + .phase     │
            │                                                 │
            ├─ KEEP    : git commit (kernel.py)，best 更新   │
            ├─ DISCARD : no_improvement++，回滚              │
            └─ FAIL    : consecutive_failures++              │
            │                                                 │
            ├─ consecutive_failures ≥ 3 ─→ DIAGNOSE ─────────┤
            ├─ plan 全部 settle          ─→ REPLAN ──────────┘
            └─ 预算用完                  ─→ FINISH
```

每个 `pN` 必有 KEEP/DISCARD/FAIL 终态。换代时旧版本里仍 pending 的项被
`create_plan.py` 自动写为 `DISCARD (superseded by replan vN)`，没有"消失"
的项。

各阶段的产物 / 输入：

| 阶段 | Claude 干的事 | 产物 |
|------|---------------|------|
| GENERATE_REF | Edit `reference.py` | reference.py |
| GENERATE_KERNEL | Edit `kernel.py` | kernel.py（种子） |
| BASELINE | `baseline.py` | seed_metric → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py '<JSON>'` | plan.md（含 (ACTIVE) 标记）+ 全局 pN |
| EDIT | Edit `kernel.py`，然后 `pipeline.py` | history.jsonl 一行 + 可能的 git commit + 下个 .phase |
| FINISH | Write `ranking.md` | ranking.md |

## Hooks 与状态机：约束是怎么产生的

总览：
[phase_machine.py](.autoresearch/scripts/phase_machine.py) 是一张**规则
表**。`<task_dir>/.ar_state/.phase` 是一根**指针**，指到表里哪一行。Hook
脚本是**执行官**。每次 Claude 要动工具，Hook 按指针查表，说挡就挡。

### 1. `phase_machine.py` 本身不守门，它只是查表库

导出三样东西。

**(a) 九个 phase 字符串常量**（[:30-41](.autoresearch/scripts/phase_machine.py#L30-L41)）：
`INIT / GENERATE_REF / GENERATE_KERNEL / BASELINE / PLAN / EDIT / DIAGNOSE /
REPLAN / FINISH`。

**(b) 两张普通 Python dict**（[:153-176](.autoresearch/scripts/phase_machine.py#L153-L176)）：

```python
_BASH_RULES = {
    INIT:            _BashPolicy("strict",     required={"export AR_TASK_DIR="}),
    BASELINE:        _BashPolicy("strict",     required={"baseline.py"}),
    GENERATE_REF:    _BashPolicy("strict",     required=set()),
    GENERATE_KERNEL: _BashPolicy("strict",     required=set()),
    PLAN:            _BashPolicy("permissive", banned=set()),
    DIAGNOSE:        _BashPolicy("permissive", banned=set()),
    REPLAN:          _BashPolicy("permissive", banned=set()),
    EDIT:            _BashPolicy("permissive", banned={"create_plan.py"}),
    FINISH:          _BashPolicy("permissive", banned=set()),
}

_EDIT_RULES = {
    GENERATE_REF:    {"ref"},        # 只允许写 reference.py
    GENERATE_KERNEL: {"editable"},   # 只允许写 task.yaml.editable_files 里的
    EDIT:            {"editable"},
    # 其他 phase：没有任何用户文件可写
}
```

`strict` 是白名单子串匹配，`permissive` 是黑名单子串匹配。模式选哪种取决于
这个 phase 该不该让 Claude 跑 ad-hoc 命令。PLAN/EDIT/DIAGNOSE/REPLAN 要查
git log、读文件，所以走 permissive；BASELINE/INIT 只有一件事可做，所以走
strict。

**(c) 两个纯函数 `check_bash` / `check_edit`**（[:212-273](.autoresearch/scripts/phase_machine.py#L212-L273)）。
传入 phase 名 + 命令/文件名，查上面两张表，返回 `(allowed, reason)`。
**不读状态、不写状态**。就是查表。

跨 phase 的全局黑名单也写在这里：`quick_check.py` / `eval_wrapper.py` /
`keep_or_discard.py` / `settle.py` 在任何 phase 都禁（只能由 `pipeline.py`
子进程跑）；`git commit` 只允许 `keep_or_discard.py` 在 KEEP 时调用。读类
命令（`ls / cat / grep / git log|diff|status / dashboard.py / echo / pwd`）
跨 phase 放行。

### 2. "当前 phase 在哪"——磁盘上一行文本

phase 状态就是 `<task_dir>/.ar_state/.phase` 这个文件，一行文本，内容是
`PLAN` / `EDIT` 之类。

有权写它的只有这几个：

- `scaffold.py --run-baseline` 结束时写 `PLAN`
- `create_plan.py` 校验通过时写 `EDIT`
- `pipeline.py` 收尾时通过 `compute_next_phase()` 算出下一 phase 并写入
- `hook_post_edit.py` / `hook_post_bash.py` 在 Edit / 脚本成功后按情况写

Claude 自己不碰这个文件。`.ar_state/*` 在 hook_guard_edit 里是放行的，但
Claude 没有动机去改（guidance 只会叫它跑脚本）。真正不可绕过的是**下一次
Hook 进程起来时查表的那一刻**。

### 3. 真正的守卫：Hook 进程 + Claude Code PreToolUse 协议

约束由 **Hook 进程**产生。看
[hook_guard_bash.py:59-76](.autoresearch/scripts/hook_guard_bash.py#L59-L76)
就够：

```python
def main():
    hook_input = read_hook_input()              # 从 stdin 读 Claude Code 传来的 JSON
    if hook_input.get("tool_name") != "Bash":
        sys.exit(0)

    task_dir = get_task_dir()                   # 从 .autoresearch/.active_task 读
    command  = hook_input["tool_input"]["command"]
    phase    = read_phase(task_dir)             # 读 <task_dir>/.ar_state/.phase

    ok, reason = check_bash(phase, command)     # ← 唯一调用 phase_machine 的地方
    if not ok:
        print(json.dumps({"decision": "block", "reason": f"[AR] {reason}. …"}))
        sys.exit(2)                             # 退出码 2 = Claude Code 拒绝执行该工具
    sys.exit(0)
```

关键在最后两行。Hook 按 Claude Code 的 PreToolUse 协议，往 stdout 写
`{"decision":"block","reason":"..."}` 再 `sys.exit(2)`。Claude Code 看到
就**直接不执行这次工具调用**，把 reason 作为工具错误反馈给 LLM。
`hook_guard_edit.py` 结构一样，调的是 `check_edit`。

### 4. 端到端：一次约束是怎么产生的

场景：`.phase` 是 `BASELINE`，Claude 想偷跑 `create_plan.py`。

```
1. Claude 发工具调用: Bash(command="python .autoresearch/scripts/create_plan.py …")
2. Claude Code 看到 .claude/settings.json 里 PreToolUse/Bash 匹配 → hook_guard_bash.py
3. Claude Code 把 {tool_name:"Bash", tool_input:{command:"…"}} 通过 stdin 发给 hook
4. hook_guard_bash.py 独立 Python 进程运行:
     task_dir = 读 .autoresearch/.active_task       → "/…/ar_tasks/xxx"
     phase    = 读 <task_dir>/.ar_state/.phase      → "BASELINE"
     ok, why  = check_bash("BASELINE", command)
          查 _BASH_RULES["BASELINE"] → ("strict", required={"baseline.py"})
          strict 模式：command 不含 "baseline.py" → (False, "phase BASELINE: …")
5. hook 打印 {"decision":"block","reason":"[AR] phase BASELINE: allowed commands = ['baseline.py']. [AR Phase: BASELINE] …"}
6. sys.exit(2)
7. Claude Code 不执行这次 Bash；LLM 收到 block reason，按 guidance 改跑 baseline.py
```

整条链路：

```
  Claude 发工具调用
        │
        ▼
  Claude Code PreToolUse ──配置在── .claude/settings.json
        │
        ▼
  hook_guard_bash.py  (独立 Python 进程，每次工具调用都 fork 一次)
        │
        ├── 读 .ar_state/.phase                     ← "当前在哪"
        ├── check_bash(phase, cmd) 查 _BASH_RULES   ← "这个 phase 允许啥"
        └── {"decision":"block"} + exit 2           ← "不让 Claude 跑"
```

### 5. Hook 接线（`.claude/settings.json`）

| 触发事件 | 匹配工具 | Hook 脚本 | 职责 |
|----------|----------|-----------|------|
| PreToolUse  | Edit / Write | `hook_guard_edit.py` | 调 `check_edit`，按 phase 拦住非法写入 |
| PreToolUse  | Bash         | `hook_guard_bash.py` | 调 `check_bash`，按 phase 拦住非法命令；顺带检测幻觉脚本名 |
| PostToolUse | Edit / Write | `hook_post_edit.py`  | Edit 完成后写 `.phase` 推进下一阶段 |
| PostToolUse | Bash         | `hook_post_bash.py`  | 脚本退出后切 phase；处理 `export AR_TASK_DIR=` 激活 |
| Stop        | —            | `hook_stop_save.py`  | 把 stop reason / 时间写进 progress.json，便于 resume |

`hook_guard_edit.py` 在 phase 表之外还有几条硬规：

- `plan.md` 永远禁写（它是 `create_plan.py` / `settle.py` / `pipeline.py`
  的输出，手改会破坏审计链）
- `.ar_state/*` 永远放行（hooks 和脚本要写状态文件）
- EDIT 阶段额外有 git gate：上一轮 kernel.py 还没 `pipeline.py` 走完就再次
  Edit，会被拦下，提示"先 pipeline.py 收尾"。这条用来防止一轮里堆叠多个未
  结算改动。

### 6. 为什么这就是一个 Mealy 状态机

- **状态**：`.phase` 里那一行文字，九个之一。
- **输入**：Claude 的工具调用（Bash/Edit）+ 脚本退出码 + `progress.json` 里
  的计数器（`consecutive_failures` / `eval_rounds` / 剩余 budget 等）。
- **输出（允许/拒绝）**：`check_bash` / `check_edit` 的返回值 → Hook 的
  block 或 pass。
- **状态转移**：由 `hook_post_bash.py` / `hook_post_edit.py` /
  `pipeline.py.compute_next_phase()` 在 PostToolUse 或子流程结束时写
  `.phase`。转移机械化：`consecutive_failures ≥ 3` 必去 DIAGNOSE，plan 全部
  settle 必去 REPLAN，预算用完必去 FINISH。没有"LLM 觉得可以继续"的逃生口。

Claude 不能越位，靠这四点：

1. **查表逻辑集中在 phase_machine，一处生效两处**。两个 PreToolUse Hook
   都 import 同一份 `_BASH_RULES` / `_EDIT_RULES`。改一处规则，两个 Hook
   行为同步变，不会漏打补丁。
2. **全局黑名单凌驾所有 phase**。`quick_check.py` / `eval_wrapper.py` /
   `keep_or_discard.py` / `settle.py` / `git commit` 在任何 phase 都禁。
   Claude 不能手动跑 pipeline 子步骤，也不能跳过 KEEP/DISCARD 直接提交。
3. **`pipeline.py` 是一轮的不可分割闭包**。LLM 在 EDIT 阶段最后一件事就是
   `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`。内部串行跑
   quick_check → eval_wrapper → keep_or_discard → settle →
   `compute_next_phase()` 写 `.phase`。pipeline 没跑完时，hook_guard_edit
   的 git gate 会让 Claude 走不出 EDIT。`keep_or_discard` 的三态：KEEP →
   `git commit` + 重置失败计数 + 更新 best；DISCARD → 工作区回滚到上一个
   KEEP commit；FAIL → 失败计数 +1 + 回滚。
4. **`create_plan.py` 校验失败就卡住**。规则：全局单调 pid（`progress.json
   .next_pid` 顺序分配，pN 永不复用、永不跳号）；≥ 3 项；最多 1 项纯参数
   调优；rationale 30–400 字符。不过关直接非零退出码，hook_post_bash 不
   推进 phase，LLM 只能按 stderr 重写 JSON。换代时旧版 pending 项被一次性
   settle 成 `DISCARD (superseded by replan vN)`。每个 `pN` 必有
   KEEP/DISCARD/FAIL 终态。

### 7. `[AR Phase: …]` Guidance + Resume

每次 phase 切换，Hook 调 `phase_machine.get_guidance(task_dir)` 拼一段
phase-specific 提示（含 editable_files、当前 active item、最近三条
history、剩余 budget 等），通过 `additionalContext` 回注给 LLM。Claude 拿到
就知道这一轮该干啥，不用回忆流程。

`/autoresearch --resume` 由 `resume.py` 找最新 task → `export AR_TASK_DIR=…`
→ PostToolUse 命中 `_handle_activation()`：`.phase` 在就暖启动；只剩
`progress.json` 就走 `compute_resume_phase()`，按 seed_metric / plan 状态
重新路由；reference.py / kernel.py 文件存在性决定 GENERATE_REF /
GENERATE_KERNEL / BASELINE 的入口。DIAGNOSE 的 guidance 里写死要 spawn
subagent 做 Root cause / Fix direction / What to avoid，逼下一轮 plan 换
思路，避免微调超参。

## Skills 库

`skills/` 按 DSL/backend 组织，88 份优化知识文档：

```
skills/triton-ascend/   — Triton on Ascend NPU (guides + cases)
skills/triton-cuda/     — Triton on CUDA GPU
skills/cuda-c/          — CUDA C
skills/cpp/             — CPU C++
skills/tilelang-cuda/   — TileLang DSL
skills/pypto/           — PyTorch operator patterns
```

PLAN 阶段 Claude 用 `Glob("skills/<dsl>/**/*.md")` 捞相关 skill，Read 对应
SKILL.md（带 YAML frontmatter），把 id 挂进 plan item 的 rationale。

## 配置与状态

| 路径 | 用途 | Git |
|------|------|-----|
| `workspace/<op>_ref.py` / `workspace/<op>_kernel.py` | 候选 ref/kernel 源文件，`/autoresearch --ref/--kernel` 的输入 | ✔ |
| `task.yaml` | 任务配置（每个 task 目录） | 随 task 分发到 worker |
| `.ar_state/progress.json` | 当前任务运行时状态 | — |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态） | — |
| `.ar_state/history.jsonl` | 每轮的 decision/metrics/commit | — |
| `.ar_state/reference.pt` | 缓存的 PyTorch ref 输出 | — |
| `.ar_state/.phase` | 当前阶段 | — |
| `.claude/settings.json` | hooks + permissions | ✔ |
| `.claude/settings.local.json` | API key、model 覆盖 | ✗ |
| `.claude/scheduled_tasks.lock` | runtime session lock | ✗ |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- 远端 NPU / CUDA 机器（可选），通过 SSH tunnel 暴露 worker HTTP 端口
