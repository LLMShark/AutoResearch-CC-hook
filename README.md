# Claude AutoResearch

Claude Code 驱动的算子迭代优化框架，纯 Python + PyYAML 实现。

> Claude 负责需要 LLM 判断的事（读代码、写 plan、改 kernel、诊断失败）；
> Hooks 负责确定性的事（阶段转移、plan 校验、eval 调度、KEEP/DISCARD、回滚）。

## 快速开始

```bash
cd claude-autoresearch
claude
```

一行命令启动（init + baseline + 进入 PLAN 自动衔接）：

```
/autoresearch --ref sinkhorn_ref.py --kernel sinkhorn_kernel.py \
  --op-name sinkhorn --backend ascend --arch ascend910b2 \
  --worker-url 127.0.0.1:9002 --max-rounds 200
```

长跑用 `/loop` 自驱模式（失败自恢复、上下文满也会延续）：

```
/loop /autoresearch --resume
```

另开终端实时看 dashboard：

```bash
python .autoresearch/scripts/dashboard.py
```

不传路径默认挑最近活跃的 task。

## 四种启动模式

| 模式 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 有 PyTorch ref 和种子 kernel | 直接 PLAN |
| `--ref X.py` | 只有 ref，kernel 现生 | GENERATE_KERNEL |
| `--desc "..."` | 纯自然语言描述 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 自然语言 + 种子 kernel | GENERATE_REF |

Resume：`/autoresearch --resume [task_dir]`（省略路径则自动挑最近活跃的）。

## 单一入口：`/autoresearch`

`/autoresearch` 是唯一的 slash command：

- 参数以 `--` 开头 → 新建任务（scaffold + 第一次 baseline 原子完成）
- 参数是已存在的目录 → resume 该目录
- `--resume` → resume 最近活跃任务
- 没参数 → 交互式询问

失败诊断走 `DIAGNOSE` 阶段：连续 3 次 FAIL 后 hook 自动切过去，phase_machine 的 guidance 指引 Claude spawn subagent 做 root-cause 分析。进度看板在 `dashboard.py`，另一个终端跑。

## 两阶段精度检查

scaffold 时本地 CPU 跑一次 PyTorch `Model` 把输入/输出 dump 成 `.ar_state/reference.pt`；此后每轮 verify：

1. Worker 端解包拿到 `reference.pt`
2. `torch.load` 取出 ref inputs/outputs
3. 只跑 `ModelNew` 的 forward，跟 stored ref 对比
4. 输出 `max_abs / max_rel / bad_elems(%)` 诊断信息

缺 `.pt` 时自动降级为原来的 inline 对比模式。容差在 `task.yaml` 配：

```yaml
metric:
  primary: latency_us
  correctness_atol: 1.0e-2
  correctness_rtol: 1.0e-2
```

好处：每轮省一次 PyTorch forward（200 轮 × 大算子可观）；verify 失败时 ref 时延仍被 `/api/v1/profile` 测出（两者彻底解耦，dashboard 顶栏始终有 PyTorch baseline）。

## 远程 Worker

Ascend NPU / CUDA 等远端硬件通过 SSH tunnel 接入。**评测算子的部分复用了 akg agent 的 worker 功能**——框架本身只负责打包 + 调用 `/api/v1/verify` 和 `/api/v1/profile`，实际在 NPU / GPU 上跑 verify/profile 的执行器来自 akg agent。

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

任务启动时加 `--worker-url 127.0.0.1:9002`，eval 自动走这台 worker。多台用逗号分隔，框架自动选可达的。

## 实时监控

```bash
# 自动选当前任务，5 秒刷新
python .autoresearch/scripts/dashboard.py

# 指定任务和刷新间隔
python .autoresearch/scripts/dashboard.py ar_tasks/my_task --watch 2

# 键盘：↑/↓/PgUp/PgDn/Home/End 滚 history，q/Esc 退出
```

显示顶栏：task 名、阶段、plan 版本、budget、Baseline（PyTorch ref 时延）、Seed（种子 kernel 时延）、Best、改进比；下栏 history table（每条带 `pN:` 前缀）和 current plan。

## 架构

```
Claude Code (LLM)                        Hooks (确定性)
  │                                         │
  ├─ GENERATE_REF: Edit(reference.py)      ├─ hook_guard_edit: 阶段门控可写文件
  ├─ GENERATE_KERNEL: Edit(kernel.py)      ├─ hook_post_edit: 自动推进阶段
  ├─ PLAN: create_plan.py (JSON)           ├─ hook_guard_bash: 阶段门控可执行脚本
  ├─ EDIT: Edit(kernel.py)                 ├─ hook_post_bash: 阶段转换 + resume
  │    └→ pipeline.py                      ├─ pipeline.py: quick_check → eval → K/D → settle
  ├─ DIAGNOSE: Agent subagent              ├─ create_plan.py: 全局 pid 分配 + 多样性 + 旧 pending 自动 supersede
  └─ FINISH: Write ranking.md              └─ settle.py: 机械更新 plan.md
```

## 阶段状态机

```
INIT
  → (--desc?) GENERATE_REF → GENERATE_KERNEL
  → (--ref only?) GENERATE_KERNEL
  → BASELINE (scaffold --run-baseline 自动跑完并写 .phase=PLAN)
  → PLAN
  → EDIT → pipeline → EDIT → pipeline → ...
  → DIAGNOSE (连续 3 次失败)  → 新 plan → EDIT
  → REPLAN (plan 全部 settle) → 新 plan → EDIT
  → FINISH (预算用完 / 手动)
```

Plan 版本换代时，旧版本里仍 pending 的项会被自动记为 `DISCARD (superseded by replan vN)`，既不被丢弃也不阻塞推进——保证每个 `pN` 都有 KEEP/DISCARD/FAIL 三态之一。

## 多样性强制

`create_plan.py` 拒绝以下 plan：

- ≥ N-1 项都是参数调优（`block_size`、`num_warps` 等词）——必须有结构性改动（fusion、内存排布、算法）
- 重复命中历史失败关键词（stderr 警告，不拒绝）
- rationale < 30 字符或 > 400 字符

`DIAGNOSE` 强制 spawn subagent（Agent 工具）做 Root cause / Fix direction / What to avoid 分析，确保替换方向真的换思路。

## Skills 库

`skills/` 按 DSL/backend 组织的 88 份优化知识文档：

```
skills/triton-ascend/   — Triton on Ascend NPU (guides + cases)
skills/triton-cuda/     — Triton on CUDA GPU
skills/cuda-c/          — CUDA C
skills/cpp/             — CPU C++
skills/tilelang-cuda/   — TileLang DSL
skills/pypto/           — PyTorch operator patterns
```

PLAN 阶段里 Claude 会 `Glob("skills/<dsl>/**/*.md")` 捞相关 skill，Read 其 SKILL.md（带 YAML frontmatter），然后把 id 挂进 plan item 的 rationale 里。

## 配置与状态

| 路径 | 用途 | Git |
|------|------|-----|
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
- （可选）远端 NPU / CUDA 机器，通过 SSH tunnel 暴露 worker HTTP 端口
