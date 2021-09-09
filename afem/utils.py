import random
from inspect import isfunction

import numpy as np
import torch
from torch import autograd


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_kwargs(kwargs, prefix):
    """Remove kwargs starting with `prefix` from dict."""
    return {k: v for k, v in kwargs.items() if not k.startswith(prefix)}


def filter_kwargs(kwargs, prefix):
    """Keep only kwargs starting with `prefix` and strip `prefix` from keys of kept items."""
    return {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def batched_eye(bsz: int, dim: int, device, dtype):
    return torch.eye(dim, device=device, dtype=dtype)[None, :, :].repeat(
        bsz, 1, 1
    )


def batched_eye_like(X: torch.Tensor):
    return torch.eye(*X.shape[1:], out=torch.empty_like(X))[None, :, :].repeat(
        X.shape[0], 1, 1
    )


def batched_jacobian_for_scalar_fun(f, x, create_graph=False):
    """
    https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6
    https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/4
    """
    def _f_sum(x):
        return f(x).sum(dim=0)
    return autograd.functional.jacobian(_f_sum, x, create_graph=create_graph)


def batched_jacobian_for_vector_fun(f, x, create_graph=False):
    """
    https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6
    https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/4
    """
    def _f_sum(x):
        return f(x).sum(dim=0)
    return autograd.functional.jacobian(_f_sum, x, create_graph=create_graph)  # .permute(1, 0, 2)
