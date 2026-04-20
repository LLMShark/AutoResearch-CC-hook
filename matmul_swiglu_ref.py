"""PyTorch reference for the fused MatMul + SwiGLU workload.

Origin: vLLM-Ascend MoE expert gate_up_proj + SwiGLU activation. Each expert's
MLP computes

    gate_up  = hidden_states @ w13_weight.T        # [N, 2H]
    gate, up = gate_up.chunk(2, dim=-1)            # [N, H] each
    return silu(gate) * up                         # [N, H]

Shapes follow vLLM-Ascend defaults (top-8 routing, TP=1, one expert's slice):
    hidden_states: [N=128, in_dim=3072]
    w13_weight:    [2H=3072, in_dim=3072]  (F.linear convention: [out, in])
    activated:     [N=128, H=1536]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_dim: int = 3072, intermediate_size: int = 1536):
        super().__init__()
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size

    def forward(self, hidden_states: torch.Tensor, w13_weight: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden_states, w13_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


def get_inputs():
    torch.manual_seed(2026)
    N, in_dim, H = 128, 3072, 1536
    # fp16 is vLLM-Ascend's inference dtype. Small scale keeps gate values in
    # a range where exp() inside SiLU does not overflow fp16.
    hidden_states = torch.randn(N, in_dim, dtype=torch.float16) * 0.1
    w13_weight = torch.randn(2 * H, in_dim, dtype=torch.float16) * 0.02
    return [hidden_states, w13_weight]


def get_init_inputs():
    return [3072, 1536]
