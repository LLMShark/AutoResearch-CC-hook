import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    torch_npu = None


@triton.jit
def sinkhorn_kernel(
    X_ptr, Y_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Sinkhorn-Knopp for one (H, W) matrix per program.

    The full H*W block stays in registers across all iterations — no DRAM
    traffic between iters, which is the whole point of fusing this into a
    single kernel instead of calling 20 pairs of reductions.
    """
    pid = tl.program_id(0)
    row = tl.arange(0, H)[:, None]
    col = tl.arange(0, W)[None, :]
    offs = pid * H * W + row * W + col

    x = tl.load(X_ptr + offs)

    # Row-wise softmax stabilization: exp(x - max_row(x))
    x_max = tl.max(x, axis=1)
    x = tl.exp(x - x_max[:, None])

    # Sinkhorn iterations: alternate row / column normalization
    for _ in tl.static_range(ITERS):
        row_sum = tl.sum(x, axis=1)
        x = x / (row_sum[:, None] + EPS)
        col_sum = tl.sum(x, axis=0)
        x = x / (col_sum[None, :] + EPS)

    tl.store(Y_ptr + offs, x)


class ModelNew(nn.Module):
    """Triton-Ascend seed kernel for SinkhornKnopp.construct().

    Same interface and numerical behavior as sinkhorn_ref.Model. One Triton
    program per (H, W) matrix; the whole block stays in registers across all
    20 iterations. Serves as a correct starting point for autoresearch, which
    will explore batching, fusion, and grid tiling.

    NOTE: Worker's verify pipeline imports `ModelNew` by name — do not rename.
    """

    def __init__(self, iters=20, eps=1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, logits):
        logits = logits.contiguous().float()
        B, C, H, W = logits.shape
        N = B * C
        x_flat = logits.view(N, H, W)
        y = torch.empty_like(x_flat)
        sinkhorn_kernel[(N,)](
            x_flat, y,
            H=H, W=W,
            ITERS=self.iters, EPS=self.eps,
        )
        return y.view(B, C, H, W)
