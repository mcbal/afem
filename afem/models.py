import math

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .rootfind import RootFind
from .solvers import broyden
from .utils import default, batched_eye_like


class VectorSpinModel(nn.Module):

    def __init__(
        self,
        num_spins,  # max sequence length N
        dim,  # embedding dimension D
        beta=1.0,  # inverse temperature
        J_init_std=1.0,
        J_symmetric=True,
        J_traceless=True,
        J_training=True,
    ):
        super().__init__()

        self.num_spins = num_spins
        self.dim = dim
        self.beta = beta

        self._init_J(
            num_spins,
            init_std=(
                J_init_std
                if J_init_std is not None
                else 1.0 / np.sqrt(num_spins)
            ),
            training=J_training,
        )
        self.J_symmetric = J_symmetric
        self.J_traceless = J_traceless

        self.diff_root_finder = RootFind(
            self._grad_t_phi,
            broyden,
            solver_fwd_max_iter=30,
            solver_fwd_tol=1e-4,
            solver_bwd_max_iter=30,
            solver_bwd_tol=1e-4,
        )

    def _init_J(self, num_spins, init_std, training):
        """Initialize random coupling matrix."""
        J = torch.zeros(num_spins, num_spins).normal_(0, init_std)
        mask = torch.zeros(num_spins).normal_(0, 2*init_std)*torch.eye(num_spins, num_spins)
        # print(mask)
        # print(J)
        J += torch.diag(mask)

        # breakpoint()
        if training:
            self._J = nn.Parameter(J)
        else:
            self.register_buffer('_J', J)

    def J(self):
        """Return symmetrized and traceless coupling matrix."""
        num_spins, J = self._J.size(0), self._J
        if self.J_symmetric:
            J = 0.5 * (J + J.t())
        if self.J_traceless:
            mask = torch.eye(num_spins, device=J.device, dtype=J.dtype)
            J = (1.0 - mask) * J
        return J

    def _prepare_sources(self, x):
        pass

    def _phi_prep(self, t, J):
        assert t.ndim == 2, f'Tensor `t` should have either shape (batch, 1) or (batch, N) but found shape {t.shape}'
        t = t.expand(-1, self.num_spins) if t.shape[-1] == 1 else t
        V = torch.diag_embed(t) - J.unsqueeze(0).expand(t.shape[0], -1, -1)
        V_inv = torch.solve(batched_eye_like(V), V).solution
        return t, V, V_inv

    def _phi(self, t, h, beta=None, J=None):
        """Compute `phi` given partition function parameters."""
        beta, J = default(beta, self.beta), default(J, self.J())
        t, V, V_inv = self._phi_prep(t, J)

        return (
            beta * t.sum(-1) - 0.5 * torch.logdet(V)
            + beta / (4.0 * self.dim) * torch.einsum('b i f, b i j, b j f -> b', h, V_inv, h)
        )

    def _grad_t_phi(self, t, h, beta=None, J=None):
        """Compute gradient of `phi` with respect to auxiliary variables `t`."""
        beta, J = default(beta, self.beta), default(J, self.J())
        _, V, V_inv = self._phi_prep(t, J)

        if t.shape[-1] == 1:
            # Scalar t that got expanded to vector t (identical elements for every spin index)
            return (
                beta * self.num_spins * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1).sum(-1, keepdim=True)
                - beta / (4.0 * self.dim) * torch.einsum('b i f, b j f, b i k, b k j -> b', h, h, V_inv, V_inv).unsqueeze(-1)
            )
        else:
            # Vector t (different elements for every spin index)
            return (
                beta * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1)
                - beta / (4.0 * self.dim) * torch.einsum('b k f, b l f, b i k, b l i -> b i', h, h, V_inv, V_inv)
            )

    def approximate_free_energy(self, beta, phi):
        pass

    def forward(self, x, beta=None, return_responses=False):
        """Probe model with data. Returns free energy and responses."""

        # Prep by padding and (optionally) adding noise to data inputs.
        h = self._prepare_sources(x)

        # Find values for which `phi` appearing in exponential inside partition function is stationary.
        t0 = torch.zeros(self.num_spins, device=x.device, dtype=x.dtype)
        t_star = self.diff_root_finder(t0, h, beta=beta)

        # Evaluate `phi` at stationary point.
        phi_star = self._phi(self, t_star, h, beta=beta)

        # Compute approximate free energy.
        # afe = self.approximate_free_energy

        # return afe
