"""Seed kernel for MatMul + SwiGLU fusion (vLLM-Ascend MoE gate_up_proj).

Intentionally UNFUSED seed:
    1. Triton-Ascend matmul produces full gate_up:[N, 2H]  ← HBM write
    2. PyTorch chunk + F.silu + mul produces activated:[N, H]

The optimization target (per the op spec) is a single Triton kernel whose
epilogue applies `silu(gate) * up` and writes only the [N, H] activated
tensor, eliminating the [N, 2H] gate_up round-trip. The grid / block sizes
here are conservative defaults; autoresearch is expected to fuse the SwiGLU
into this kernel AND tune tiling / parallelism.

NOTE: The worker's verify pipeline imports `ModelNew` by name — do not rename.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    torch_npu = None


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M, N] = A[M, K] @ B.T  where B is [N, K] (F.linear weight layout).

    Grid = (NUM_CORES,); each core sweeps its slice of the (M_tiles × N_tiles)
    output grid via stride-NUM_CORES stepping (canonical Ascend pattern —
    better than raw 2D grid because block-count rarely matches AIcore count).
    Accumulator is fp32 for numerical stability, cast to fp16 on store.
    """
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    pid = tl.program_id(0)
    for block_idx in range(pid, num_blocks, NUM_CORES):
        bm = block_idx // num_blocks_n
        bn = block_idx % num_blocks_n

        offs_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # A tile: [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # B tile: [BLOCK_K, BLOCK_N] — built via stride ordering from the
            # stored [N, K] weight so tl.dot can consume it directly.
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, in_dim: int = 3072, intermediate_size: int = 1536):
        super().__init__()
        self.in_dim = in_dim
        self.H = intermediate_size

    def forward(self, hidden_states: torch.Tensor, w13_weight: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        w13_weight = w13_weight.contiguous()
        M, K = hidden_states.shape
        N2 = w13_weight.shape[0]  # 2H

        gate_up = torch.empty(M, N2, dtype=hidden_states.dtype, device=hidden_states.device)

        # Conservative defaults — autoresearch will tune.
        NUM_CORES = 20
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        matmul_kernel[(NUM_CORES,)](
            hidden_states, w13_weight, gate_up,
            M, N2, K,
            hidden_states.stride(0), hidden_states.stride(1),
            w13_weight.stride(0), w13_weight.stride(1),
            gate_up.stride(0), gate_up.stride(1),
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # SwiGLU as a separate PyTorch step — THIS is the fusion target.
        # autoresearch should rewrite the kernel so the epilogue computes
        # `silu(gate) * up` in-tile and writes only the [M, H] activated
        # output, eliminating the [M, 2H] gate_up intermediate.
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up
