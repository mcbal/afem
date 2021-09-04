import random

import numpy as np
import torch


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_kwargs(kwargs, prefix):
    return {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}


##############################################################################
# TENSORS
##############################################################################


def batched_eye(bsz: int, dim: int, device, dtype):
    return torch.eye(dim, device=device, dtype=dtype)[None, :, :].repeat(
        bsz, 1, 1
    )


def batched_eye_like(X: torch.Tensor):
    return torch.eye(*X.shape[1:], out=torch.empty_like(X))[None, :, :].repeat(
        X.shape[0], 1, 1
    )
