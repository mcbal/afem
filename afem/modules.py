import torch
import torch.nn as nn
import numpy as np

from .utils import exists


class ScaleNorm(nn.Module):
    """Simple L2-normalization with a single (trainable) scale parameter; see https://arxiv.org/abs/1910.05895."""

    def __init__(self, scale=1.0, fixed=True, eps=1e-5):
        super().__init__()
        self.scale = scale if fixed else nn.Parameter(scale*torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.scale
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=1, dropout=0.1, dense=nn.Linear):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
