import torch
import torch.nn as nn
import torch.autograd as autograd

from .solvers import newton
from .utils import filter_kwargs, remove_kwargs


class RootFind(nn.Module):
    _default_kwargs = {
        'solver_fwd_max_iter': 30,
        'solver_fwd_tol': 1e-4,
        'solver_bwd_max_iter': 30,
        'solver_bwd_tol': 1e-4,
    }

    def __init__(self, fun, solver=newton, **kwargs):
        super().__init__()
        self.fun = fun
        self.solver = solver
        self.kwargs = self._default_kwargs
        self.kwargs.update(**kwargs)

    def _root_find(self, z0, x, *args, **kwargs):

        # Compute forward pass: find root of function outside autograd tape.
        with torch.no_grad():
            z_root = self.solver(
                lambda z: self.fun(z, x, *args, **remove_kwargs(kwargs, 'solver_')),
                z0,
                **filter_kwargs(kwargs, 'solver_fwd_'),
            )['result']

        if self.training:
            # Re-engage autograd tape (no-op in terms of value of z).
            z_root = z_root + self.fun(z_root.requires_grad_(), x, *args, **remove_kwargs(kwargs, 'solver_'))

            # Set up backward hook for root-solving in backward pass.
            z_bwd = z_root.clone().detach().requires_grad_()
            fun_bwd = self.fun(z_bwd, x, *args, **remove_kwargs(kwargs, 'solver_'))

            def backward_hook(grad):
                return self.solver(
                    lambda y: autograd.grad(fun_bwd, z_bwd, y, retain_graph=True, create_graph=True)[0] + grad,
                    torch.zeros_like(grad), **filter_kwargs(kwargs, 'solver_bwd_')
                )['result']

            z_root.register_hook(backward_hook)

        return z_root

    def forward(self, z0, x, *args, **kwargs):
        return self._root_find(z0, x, *args, **{**self.kwargs, **kwargs})
