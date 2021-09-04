import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0., dense=nn.Linear):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
