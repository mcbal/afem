import math

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .modules import RootFind
# from .solvers import broyden
from .utils import default


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

        self._init_weight(
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
            self._partition_function_grad_t_phi,
            broyden,
            solver_fwd_max_iter=40,
            solver_fwd_tol=1e-4,
            solver_bwd_max_iter=40,
            solver_bwd_tol=1e-4,
        )

    def _init_J(self, num_spins, init_std, training):
        """Initialize random coupling matrix."""
        J = torch.zeros(num_spins, num_spins).normal_(0, init_std)
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

    def _partition_function_phi(self, h, t, beta=None, J=None):
        """Compute `phi` given partition function parameters."""
        beta, J = default(beta, self.beta), default(J, self.J)
        t = t * torch.ones(self.num_spins, device=h.device, dtype=h.dtype) if t.numel() == 1 else t
        V = torch.diag(t) - J
        V_inv = torch.solve(torch.ones_like(V), V).solution
        return (
            repeat(beta * t.sum() - 0.5 * torch.logdet(V), '() -> b', b=h.size(0))
            + beta / (4.0 * self.dim) * torch.einsum('b i f, i j, b j f -> b', h, V_inv, h)
        )

    def _partition_function_grad_t_phi(self, h, t, beta=None, J=None):
        """Compute gradient of `phi` with respect to auxiliary variables `t`."""
        beta, J = default(beta, self.beta), default(J, self.J)
        t = t * torch.ones(self.num_spins, device=h.device, dtype=h.dtype) if t.numel() == 1 else t
        V = torch.diag(t) - J
        V_inv = torch.solve(torch.ones_like(V), V).solution
        return (
            repeat(beta - 0.5 * torch.diag(V_inv).sum(), '() -> b', b=h.size(0))
            + beta / (4.0 * self.dim) * torch.einsum('b i f, b i f -> b i', h, h)
        )

    def approximate_free_energy(self, beta, phi):
        math.pi
        return (1.0 / beta) *

    def forward(self, x, beta=None, return_responses=False):
        """Probe model with data. Returns free energy and responses."""

        # Prep by padding and (optionally) adding noise to data inputs.
        h = self._prepare_sources(x)

        # Find values for which `phi` appearing in exponential inside partition function is stationary.
        t0 = torch.zeros(self.num_spins, device=x.device, dtype=x.dtype)
        t_star = self.diff_root_finder(h, t0, beta=beta)

        # Evaluate `phi` at stationary point.
        phi_star = self._partition_function_phi(self, h, t_star, beta=beta)

        # Compute approximate free energy.
        # afe = self.approximate_free_energy

        # return afe
