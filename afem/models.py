from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from .rootfind import RootFind
from .utils import batch_eye, batch_eye_like, batch_jacobian, default, exists


@dataclass
class VectorSpinModelOutput:
    afe: torch.Tensor
    t_star: torch.Tensor
    magnetizations: Optional[torch.Tensor]
    internal_energy: Optional[torch.Tensor]
    log_prob: Optional[torch.Tensor]


class VectorSpinModel(nn.Module):
    """Implementation of differentiable steepest-descent approximation of a vector-spin model.

    See https://mcbal.github.io/post/transformers-from-spin-models-approximate-free-energy-minimization/.

    TODO: Add second-order derivatives: specific heat and susceptibilities (fluctuations).
    """

    def __init__(
        self,
        num_spins,
        dim,
        beta=1.0,
        beta_requires_grad=True,
        beta_parameter=False,
        J_init_std=None,
        J_external=False,
        J_symmetric=True,
        J_traceless=True,
        solver_fwd_max_iter=40,
        solver_fwd_tol=1e-5,
        solver_bwd_max_iter=40,
        solver_bwd_tol=1e-5,
    ):
        super().__init__()

        self.num_spins = num_spins
        self.dim = dim

        # Beta stuff.
        beta = torch.as_tensor(beta)
        if beta_parameter:
            self.beta = nn.Parameter(beta)
        elif beta_requires_grad:
            self.beta = beta.requires_grad_()
        else:
            self.register_buffer('beta', beta)

        # Coupling stuff.
        if J_external:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
        else:
            self._J = nn.Parameter(
                torch.empty(num_spins, num_spins).normal_(
                    0.0, J_init_std if exists(J_init_std) else 1.0 / np.sqrt(num_spins*dim)
                )
            )
        self.J_external = J_external
        self.J_symmetric = J_symmetric
        self.J_traceless = J_traceless

        # Initialize implicit layer for differentiable root-finding.
        self.diff_root_finder = RootFind(
            self._jac_phi,
            solver_fwd_max_iter=solver_fwd_max_iter,
            solver_fwd_tol=solver_fwd_tol,
            solver_bwd_max_iter=solver_bwd_max_iter,
            solver_bwd_tol=solver_bwd_tol,
        )

    def J(self, h):
        """Return couplings tensor.

        The functional choice of how to turn external inputs into couplings is by no means unique. The
        choice below, in combination with the weight initializations of the linear maps above, yields
        contributions of roughly the same order of magnitude in norm as the internal couplings do.

        TODO: Making this an `nn.Module` on its own might be cleaner and more composable.
        TODO: Find a way to cache the result of this function without breaking autograd. Maybe use
                the new `torch.nn.utils.parametrize` functionality in `torch>=1.9.0`.
        """
        bsz, num_spins, dim = h.shape
        if self.J_external:
            q, k = self.to_q(h), self.to_k(h)
            J = torch.tanh(torch.einsum('b i f, b j f -> b i j', q, k) * np.sqrt(dim)) / np.sqrt(num_spins*dim)
        else:
            J = repeat(self._J, 'i j -> b i j', b=bsz)
        if self.J_symmetric:
            J = 0.5 * (J + J.permute(0, 2, 1))
        if self.J_traceless:
            mask = batch_eye(bsz, num_spins, device=J.device, dtype=J.dtype)
            J = (1.0 - mask) * J
        return J

    def _phi_prep(self, t, J):
        """Construct `V` and its inverse given `t` and couplings `J`."""
        assert t.ndim == 2, f'Tensor `t` should have either shape (batch, 1) or (batch, N) but found shape {t.shape}'
        t = t.repeat(1, self.num_spins) if t.size(-1) == 1 else t
        V = torch.diag_embed(t) - J
        V_inv = torch.linalg.solve(V, batch_eye_like(V))
        return t, V, V_inv

    def _phi(self, t, h, beta=None, J=None):
        """Compute scalar `phi` given partition function parameters."""
        beta, J = default(beta, self.beta), default(J, self.J(h))
        t, V, V_inv = self._phi_prep(t, J)
        return (
            beta * t.sum(dim=-1) - 0.5 * torch.logdet(V)
            + beta / 4.0 * torch.einsum('b i f, b i j, b j f -> b', h, V_inv, h)
        )[:, None]

    def _jac_phi(self, t, h, beta=None, J=None):
        """Compute gradient of `phi` with respect to auxiliary variables `t`.

        For every example in the batch, the vector case with different auxiliary variables
        for every spin yields a vector whereas the scalar case with identical auxiliary
        variables for every spin yields just a scalar.
        """
        beta, J = default(beta, self.beta), default(J, self.J(h))
        _, _, V_inv = self._phi_prep(t, J)
        if t.size(-1) == 1:
            return (
                beta * self.num_spins - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1).sum(dim=-1)
                - beta / 4.0 * torch.einsum('b j i, b j f, b k f, b i k -> b', V_inv, h, h, V_inv)
            )[:, None]
        else:
            return (
                beta * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1)
                - beta / 4.0 * torch.einsum('b j i, b j f, b k f, b i k -> b i', V_inv, h, h, V_inv)
            )

    def _hess_phi(self, t, h, beta=None, J=None):
        """Compute (symmetric) Hessian of `phi` with respect to auxiliary variables `t`.

        For every example in the batch, the vector case with different auxiliary variables
        for every spin yields a matrix whereas the scalar case with identical auxiliary
        variables for every spin yields just a scalar.
        """
        beta, J = default(beta, self.beta), default(J, self.J(h))
        _, _, V_inv = self._phi_prep(t, J)
        if t.size(-1) == 1:
            return (
                0.5 * torch.einsum(
                    'b i j, b j i -> b', V_inv, V_inv)
                + beta / 4.0 * torch.einsum(
                    'b k i, b k f, b l f, b j l, b i j -> b', V_inv, h, h, V_inv, V_inv)
                + beta / 4.0 * torch.einsum(
                    'b j i, b l j, b l f, b k f, b i k -> b', V_inv, V_inv, h, h, V_inv)
            )[:, None, None]
        else:
            return (
                0.5 * torch.einsum(
                    'b i j, b j i -> b i j', V_inv, V_inv)
                + torch.diag_embed(
                    beta / 4.0 * torch.einsum(
                        'b k i, b k f, b l f, b j l, b i j -> b i', V_inv, h, h, V_inv, V_inv)
                    + beta / 4.0 * torch.einsum(
                        'b j i, b l j, b l f, b k f, b i k -> b i', V_inv, V_inv, h, h, V_inv)
                )
            )

    def approximate_log_Z(self, t, h, beta=None):
        """Compute steepest-descent approximation of log of parition function for large `self.dim`."""
        beta = default(beta, self.beta)
        return -0.5 * self.num_spins * (1.0 + torch.log(2.0*beta)) + self._phi(t, h, beta=beta)

    def approximate_free_energy(self, t, h, beta=None):
        """Compute steepest-descent approximation of free energy for large `self.dim`.

        Actually a free energy density since it's divided by `self.dim`". Divide by `self.num_spins`
        to get an average value per site. To calculate thermodynamic quantities, take derivatives
        with respect to gradient-requiring parameters. Careful for implicit dependencies when
        evaluating `t` away from the stationary point where phi'(t*) != 0.
        """
        beta = default(beta, self.beta)
        return -1.0 / beta * self.approximate_log_Z(t, h, beta=beta)

    def magnetizations(self, t, h, beta):
        """Differentiate log(Z) with respect to the sources `h_i` to get <sigma_i>.

        If evaluated at `t = t_star`, implicit derivative paths via `t_star(h)`
        do not contribute to the values because `phi'(t_star) = 0` at the stationary
        point, but the gradients still propagate.
        """
        if not h.requires_grad:
            h.requires_grad_()
        magnetizations = batch_jacobian(
            lambda z: self.approximate_log_Z(t, z, beta=beta),
            h, create_graph=True, swapaxes=False)[0]
        return magnetizations

    def internal_energy(self, t, h, beta):
        """Differentiate log(Z) with respect to the sources `beta` to get <E>.

        If evaluated at `t = t_star`, implicit derivative paths via `t_star(beta)`
        do not contribute to the values because `phi'(t_star) = 0` at the stationary
        point, but the gradients still propagate.
        """
        if not beta.requires_grad:
            beta.requires_grad_()
        energy = batch_jacobian(
            lambda z: -self.approximate_log_Z(t, h, beta=z),
            beta, create_graph=True, swapaxes=False)[0]
        return energy

    def log_prob(self, t, h, beta=None):
        """Compute log_prob for inputs `h` in the steepest-descent approximation.

        Returns -log[p(h)] = E(h) + log[Z] = - log[Z(h)] + log[integral dh Z(h)]
        where the first term is the steepest descent approximation and the second
        term is just a Gaussian integral which ends up cancelling the first term
        `beta * t.sum(dim=-1)` in `phi`.
        """
        beta, J = default(beta, self.beta), self.J(h)
        t, V, V_inv = self._phi_prep(t, J)
        return (
            -0.5 * torch.logdet(V)
            + beta / 4.0 * torch.einsum('b i f, b i j, b j f -> b', h, V_inv, h)
        )[:, None]

    def forward(
        self,
        h,
        t0,
        beta=None,
        return_afe=False,
        return_magnetizations=True,
        return_internal_energy=False,
        return_log_prob=False,
        use_analytical_grads=True,
    ):
        beta = default(beta, self.beta)
        t0 = repeat(t0, 'i -> b i', b=h.size(0))

        # Solve for stationary point of exponential appearing in partition function.
        if use_analytical_grads:
            t_star = self.diff_root_finder(t0, h, beta=beta, solver_fwd_jac_f=(
                lambda z: self._hess_phi(z, h, beta=beta))
            )
        else:
            t_star = self.diff_root_finder(t0, h, beta=beta)

        return VectorSpinModelOutput(
            t_star=t_star,
            afe=self.approximate_free_energy(t_star, h, beta=beta) if return_afe else None,
            magnetizations=self.magnetizations(t_star, h, beta=beta) if return_magnetizations else None,
            internal_energy=self.internal_energy(t_star, h, beta=beta)if return_internal_energy else None,
            log_prob=self.log_prob(t_star, h) if return_log_prob else None,
        )
