import torch
import torch.nn as nn
import numpy as np


class ScaleNorm(nn.Module):
    """Simple L2-normalization with a single (trainable) scale parameter; see https://arxiv.org/abs/1910.05895."""

    def __init__(self, dim, fixed=True, eps=1e-5):
        super().__init__()
        self.g = np.sqrt(dim) if fixed else nn.Parameter(np.sqrt(dim)*torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return x
