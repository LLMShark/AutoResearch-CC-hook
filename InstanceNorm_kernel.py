import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    torch_npu = None


@triton.jit
def instance_norm_kernel(
    X_ptr, Y_ptr,
    num_instances: tl.constexpr,
    spatial_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    Instance Normalization Triton kernel.

    Each instance = one (batch, channel) pair, normalized over spatial dims (H*W).
    Uses two-pass approach: compute mean/var, then normalize.
    """
    core_id = tl.program_id(0)

    for inst_idx in range(core_id, num_instances, CORE_NUM):
        base_offset = inst_idx * spatial_size

        # Pass 1: compute mean and variance
        mean_acc = 0.0
        var_acc = 0.0
        for i in range(0, spatial_size, BLOCK_SIZE):
            offsets = base_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = (i + tl.arange(0, BLOCK_SIZE)) < spatial_size
            x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            mean_acc += tl.sum(x, axis=0)
            var_acc += tl.sum(x * x, axis=0)

        mean_val = mean_acc / spatial_size
        var_val = var_acc / spatial_size - mean_val * mean_val
        inv_std = 1.0 / tl.sqrt(var_val + eps)

        # Pass 2: normalize
        for i in range(0, spatial_size, BLOCK_SIZE):
            offsets = base_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = (i + tl.arange(0, BLOCK_SIZE)) < spatial_size
            x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            y = (x - mean_val) * inv_std
            tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-based Instance Normalization for Ascend NPU.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5

        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W)
        x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        N, C, H, W = x.shape
        spatial_size = H * W
        num_instances = N * C  # each (batch, channel) is one instance

        y = torch.empty_like(x)

        BLOCK_SIZE = min(1024, spatial_size)
        # Round up to power of 2 for Triton
        BLOCK_SIZE = 1 << (BLOCK_SIZE - 1).bit_length()

        grid = (self.VEC_CORE_NUM,)
        instance_norm_kernel[grid](
            x, y,
            num_instances=num_instances,
            spatial_size=spatial_size,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            CORE_NUM=self.VEC_CORE_NUM,
        )

        return y
