## 2. 核心融合方案 B: Expert gate_up_proj + SwiGLU (MatMul + Activation 深度融合)

### 2.1 原始代码

**原始代码路径**: `vllm/vllm/model_executor/layers/fused_moe/`
**逻辑位置**: vllm-ascend `AscendUnquantizedFusedMoEMethod.apply` (`fused_moe.py:97-208`)

在 vllm-ascend 的 MoE expert 计算中，每个 expert 的 MLP 执行流程:

```python
# --- [原始代码] 每个 Expert 的 MLP 计算 (在 fused_experts 内部) ---
# w13_weight: [256, 3072, 3072] (gate+up merged, intermediate_size*2=3072)
#             TP=8: [256, 384, 3072]
# w2_weight:  [256, 3072, 1536] (down proj)
#             TP=8: [256, 3072, 192]

# Step 1: gate_up projection (MatMul)
gate_up = torch.nn.functional.linear(hidden_states, w13_weight)
# gate_up: [N_expert, 3072]  (TP=8: [N_expert, 384])

# Step 2: SwiGLU activation (SiLU + Mul)
# 当前使用 npu_swiglu 已融合 SiLU + Mul
activated = torch_npu.npu_swiglu(gate_up)
# activated: [N_expert, 1536]  (TP=8: [N_expert, 192])

# Step 3: down projection (MatMul)
output = torch.nn.functional.linear(activated, w2_weight)
```

### 2.2 替换后的代码

**替换建议**: 将 gate_up MatMul 与 SwiGLU 激活函数深度融合为一个 kernel，消除 `gate_up` 中间张量的 HBM 写回。

```python
# --- [融合代码] Expert MLP: MatMul + SwiGLU 深度融合 ---
# hidden_states: [N_expert, 3072]
# w13_weight: [256, 3072, 3072] (TP=8: [256, 384, 3072])
activated = torch.ops.npu.npu_fused_matmul_swiglu(
    input=hidden_states,
    weight=w13_weight,
    bias=None,
)
# activated: [N_expert, 1536] (TP=8: [N_expert, 192]) — 中间 gate_up 张量不落地

# down projection (不变)
output = torch.nn.functional.linear(activated, w2_weight)
```

### 2.3 融合可行性与收益分析

| 维度 | 分析 |
|------|------|
| **消除中间张量** | 消除 `gate_up: [N_expert, 3072]` (TP=8: `[N_expert, 384]`) 中间张量。256 个 expert 每个 expert 均产生此中间张量，top-8 时 8 个 expert 并行计算 |
| **延迟收益** | 减少 1 次 HBM 写 + 1 次 HBM 读 + 1 次 kernel launch / expert。top-8 意味着每 token 有 8 个 expert 各自受益 |
| **实现复杂度** | 中。需实现 NPU 原生 MatMul + SwiGLU 融合算子。参考 `torch_npu.npu_swiglu` 和 `torch_npu.npu_bmmV2` 的组合 |
| **TP 兼容性** | expert 权重已按 TP 分片 (column parallel for w13, row parallel for w2)，融合算子需保持相同的数据流 |
| **风险** | 中。MatMul + SwiGLU 融合需要 NPU CANN 底层支持或 Triton 实现。若 CANN 已提供 `npu_fused_matmul_swiglu` 则风险较低。此外 fp8 权重在 load-time 已反量化为 bf16，不影响运行时融合 |