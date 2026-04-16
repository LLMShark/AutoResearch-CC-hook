import torch
import torch.nn as nn


class Model(nn.Module):
    """PyTorch reference for SinkhornKnopp.construct()."""

    def __init__(self, iters=20, eps=1e-6):
        super(Model, self).__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, logits):
        logits = logits.float()
        logits_max = logits.amax(dim=-1, keepdim=True)
        matrix = torch.exp(logits - logits_max)
        for _ in range(self.iters):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + self.eps)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + self.eps)
        return matrix


def get_inputs():
    torch.manual_seed(2026)
    logits = torch.randn(
        16384,
        2,
        4,
        4,
        dtype=torch.float32,
    )
    return [logits]


def get_init_inputs():
    return [20, 1e-6]
