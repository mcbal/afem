# Root-finding algorithms.

import torch
from torch.autograd.functional import jacobian


def batch_jacobian(f, x):
    """
    https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6
    """
    def f_sum(x): return torch.sum(f(x), dim=0)
    ret = jacobian(f_sum, x).permute(1, 0, 2)
    # print(ret)
    return ret


def newton(f, z_init, max_iter=40, tol=1e-5):
    def g(z):
        # print(f'inside g: {z.shape}, {batch_jacobian(f, z).shape}, {f(z).shape}')
        print(batch_jacobian(f, z))
        return z - torch.linalg.solve(batch_jacobian(f, z), f(z))
    z_prev, z, num_steps = z_init, g(z_init), 0
    while torch.linalg.norm(z_prev - z) > tol or num_steps < max_iter:
        z_prev, z = z, g(z)
        print(f'iteration {num_steps}: {z}, {f(z)}')
        num_steps += 1
    return z
