# Basic Newton method for finding roots in k variables.
#
# Requires adding `create_graph=True` in backward hook of forward pass of
# `rootfind.RootFind` to get Jacobian of gradient function. Also possible
# to pass along an analytical expression for the Jacobian via `jac_f`.

import torch

from .utils import batch_jacobian


def newton(f, z_init, jac_f=None, max_iter=40, tol=1e-4):
    def jacobian(f, z):
        return jac_f(z) if jac_f is not None else batch_jacobian(f, z)

    def g(z):
        return z - torch.linalg.solve(jacobian(f, z), f(z))

    z_prev, z, n_steps, trace = z_init, g(z_init), 0, []

    while torch.linalg.norm(z_prev - z) > tol and n_steps < max_iter:
        z_prev, z = z, g(z)
        n_steps += 1
        trace.append(torch.linalg.norm(f(z)).detach())

    return {
        'result': z,
        'n_steps': n_steps,
        'trace': trace,
        'max_iter': max_iter,
        'tol': tol,
    }
