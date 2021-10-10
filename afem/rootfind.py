import torch
import torch.nn as nn
import torch.autograd as autograd

from .solvers import newton
from .utils import filter_kwargs, remove_kwargs


class RootFind(nn.Module):
    """Differentiable root-solving using implicit differentiation. 

    See https://implicit-layers-tutorial.org/introduction/ and https://github.com/locuslab/deq.
    """
    _default_kwargs = {
        'solver_fwd_max_iter': 30,
        'solver_fwd_tol': 1e-4,
        'solver_bwd_max_iter': 30,
        'solver_bwd_tol': 1e-4,
    }

    def __init__(self, f, solver=newton, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = self._default_kwargs
        self.kwargs.update(**kwargs)

    def _root_find(self, z0, x, *args, **kwargs):

        # Compute forward pass: find root of function outside autograd tape.
        with torch.no_grad():
            z_root = self.solver(
                lambda z: self.f(z, x, *args, **remove_kwargs(kwargs, 'solver_')),
                z0,
                **filter_kwargs(kwargs, 'solver_fwd_'),
            )['result']
            new_z_root = z_root

        if self.training:
            # Re-engage autograd tape (no-op in terms of value of z).
            new_z_root = z_root.requires_grad_() - self.f(z_root.requires_grad_(), x, *args, **remove_kwargs(kwargs, 'solver_'))

            # Set up backward hook for solving of linear system in backward pass.
            z_root_bwd = new_z_root.clone().detach().requires_grad_()

            if kwargs.get('solver_fwd_jac_f') is not None:
                def backward_hook(grad):
                    return torch.linalg.solve(kwargs['solver_fwd_jac_f'](z_root_bwd), grad)
            else:
                f_bwd = -self.f(z_root_bwd, x, *args, **remove_kwargs(kwargs, 'solver_'))

                def backward_hook(grad):
                    return self.solver(
                        lambda y: autograd.grad(f_bwd, z_root_bwd, y, retain_graph=True, create_graph=True)[0] + grad,
                        torch.zeros_like(grad), **filter_kwargs(kwargs, 'solver_bwd_')
                    )['result']

            new_z_root.register_hook(backward_hook)

        return new_z_root

    def forward(self, z0, x, *args, **kwargs):
        return self._root_find(z0, x, *args, **{**self.kwargs, **kwargs})
