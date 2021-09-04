from abc import ABCMeta, abstractmethod
from functools import reduce

import torch
import torch.nn as nn
import torch.autograd as autograd

from .utils import filter_kwargs


class _RootFindModule(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.shapes = None

    def pack_state(self, z_list):
        """Transform list of batched tensors into batch of vectors."""
        self.shapes = [t.shape[1:] for t in z_list]
        bsz = z_list[0].shape[0]
        z = torch.cat([t.reshape(bsz, -1) for t in z_list], dim=1)
        return z

    def unpack_state(self, z):
        """Transform batch of vectors into list of batched tensors."""
        assert self.shapes is not None
        bsz, z_list = z.shape[0], []
        start_idx, end_idx = 0, reduce(lambda x, y: x * y, self.shapes[0])
        for i in range(len(self.shapes)):
            z_list.append(z[:, start_idx:end_idx].view(bsz, *self.shapes[i]))
            if i < len(self.shapes) - 1:
                start_idx = end_idx
                end_idx += reduce(lambda x, y: x * y, self.shapes[i + 1])
        return z_list

    @abstractmethod
    def _initial_guess(self, x):
        pass

    @abstractmethod
    def forward(self, z, x, *args):
        pass


class RootFind(nn.Module):
    _default_kwargs = {
        'solver_fwd_max_iter': 30,
        'solver_fwd_tol': 1e-4,
        'solver_bwd_max_iter': 30,
        'solver_bwd_tol': 1e-4,
    }

    def __init__(self, fun, solver, output_elements=[0], **kwargs):
        super().__init__()
        self.fun = fun
        self.solver = solver
        self.output_elements = output_elements
        self.kwargs = self._default_kwargs
        self.kwargs.update(**kwargs)

    def _root_find(self, z0, x, *args, **kwargs):
        """Find root of `fun` given `z0` and `x`."""

        # Compute forward pass: find root of function outside autograd tape.
        with torch.no_grad():
            z_root = self.solver(
                lambda z: self.fun(z, x, *args),
                z0,
                **filter_kwargs(kwargs, 'solver_fwd_'),
            )['result']

        if self.training:
            # Re-engage autograd tape (no-op in terms of value of z).
            z_root = z_root + self.fun(z_root, x, *args)

            # Set up backward hook for root-solving in backward pass.
            z_bwd = z_root.clone().detach().requires_grad_()
            fun_bwd = self.fun(z_bwd, x, *args)

            def backward_hook(grad):
                new_grad = self.solver(
                    lambda y: autograd.grad(
                        fun_bwd, z_bwd, y, retain_graph=True)[0]
                    + grad,
                    torch.zeros_like(grad),
                    **filter_kwargs(kwargs, 'solver_bwd_'),
                )['result']
                return new_grad

            z_root.register_hook(backward_hook)

        return z_root

    def forward(self, x, *args, **kwargs):
        # Merge default kwargs with incoming runtime kwargs.
        kwargs = {**self.kwargs, **kwargs}
        # Get list of initial guess tensors and reshape into a batch of vectors.
        z0 = self.fun.pack_state(self.fun._initial_guess(x))
        # Find root.
        z_root = self._root_find(z0, x, *args, **kwargs)
        # Return (subset of) list of tensors of original input shapes.
        out = [self.fun.unpack_state(z_root)[i] for i in self.output_elements]
        return out[0] if len(out) == 1 else out
