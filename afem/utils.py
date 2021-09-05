import random
from inspect import isfunction

import numpy as np
import torch


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
