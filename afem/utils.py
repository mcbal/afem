from inspect import isfunction

import torch
from torch import autograd


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def remove_kwargs(kwargs, prefix):
    """Remove kwargs starting with `prefix` from dict."""
    return {k: v for k, v in kwargs.items() if not k.startswith(prefix)}


def filter_kwargs(kwargs, prefix):
    """Keep only kwargs starting with `prefix` and strip `prefix` from keys of kept items."""
    return {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}


def batch_eye(bsz: int, dim: int, device, dtype):
    """Return batch of identity matrices."""
    return torch.eye(dim, device=device, dtype=dtype)[None, :, :].repeat(bsz, 1, 1)


def batch_eye_like(X: torch.Tensor):
    """Return batch of identity matrices like given batch of matrices `X`."""
    return torch.eye(*X.shape[1:], out=torch.empty_like(X))[None, :, :].repeat(X.size(0), 1, 1)


@torch.enable_grad()
def batch_jacobian(f, x, create_graph=False, swapaxes=True):
    """Compute batched Jacobian for vector outputs of `f`.

    https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571
    """
    def _f_sum(x):
        return f(x).sum(dim=0) if callable(f) else f.sum(dim=1)
    jac_f = autograd.functional.jacobian(_f_sum, x, create_graph=create_graph)
    return jac_f.swapaxes(1, 0) if swapaxes else jac_f
