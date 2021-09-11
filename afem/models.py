import math

import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from .rootfind import RootFind
from .solvers import newton
from .utils import default, batch_eye_like, batch_jacobian


# TODO: Design abstract base class for couplings modules
# TODO: Add different flavours of couplings (output is always J_ij):
#           - return J_ij (symmetric traceless matrix as parameters)
#           - return J_ij + H^T W H (transformer-style based on inputs)
#               where dim^2 parameters W can be seen as W_Q^T x W_K equivalent


class VectorSpinModel(nn.Module):

    def __init__(
        self,
        num_spins,
        dim,
        beta=1.0,
        beta_requires_grad=True,
        beta_parameter=False,
        J_init_std=None,
        J_parameter=True,
        J_symmetric=True,
        J_traceless=True,
    ):
        super().__init__()

        self.num_spins = num_spins
        self.dim = dim

        # Setup inverse temperature.
        beta = torch.as_tensor(beta)
        if beta_parameter:
            self.beta = nn.Parameter(beta)
        elif beta_requires_grad:
            self.beta = beta.requires_grad_()
        else:
            self.register_buffer('beta', beta)

        # Setup couplings.
        J = torch.zeros(num_spins, num_spins).normal_(
            0, J_init_std if J_init_std is not None else 1.0 / np.sqrt(num_spins*dim**2)
        )
        if J_parameter:
            self._J = nn.Parameter(J)
        else:
            self.register_buffer('_J', J)
        self.J_symmetric = J_symmetric
        self.J_traceless = J_traceless

        # Initialize implicit layer for differentiable root-finding.
        self.diff_root_finder = RootFind(
            self._grad_t_phi,
            newton,
            solver_fwd_max_iter=50,
            solver_fwd_tol=1e-4,
            solver_bwd_max_iter=50,
            solver_bwd_tol=1e-4,
        )

    def J(self):
        """Return symmetrized and traceless coupling matrix."""
        num_spins, J = self._J.size(0), self._J
        if self.J_symmetric:
            J = 0.5 * (J + J.t())
        if self.J_traceless:
            mask = torch.eye(num_spins, device=J.device, dtype=J.dtype)
            J = (1.0 - mask) * J
        return J

    def _phi_prep(self, t, J):
        """Construct `V` and its inverse given `t` and couplings `J`."""
        assert t.ndim == 2, f'Tensor `t` should have either shape (batch, 1) or (batch, N) but found shape {t.shape}'
        t = t.repeat(1, self.num_spins) if t.size(-1) == 1 else t
        V = torch.diag_embed(t) - repeat(J, 'i j -> b i j', b=t.size(0))
        V_inv = torch.linalg.solve(V, batch_eye_like(V))
        return t, V, V_inv

    def _phi(self, t, h, beta=None, J=None):
        """Compute `phi` given partition function parameters."""
        beta, J = default(beta, self.beta), default(J, self.J())
        t, V, V_inv = self._phi_prep(t, J)
        return (
            beta * t.sum(-1) - 0.5 * torch.logdet(V)
            + beta / (4.0 * self.dim) * torch.einsum('b i f, b i j, b j f -> b', h, V_inv, h)
        )[:, None]

    def _grad_t_phi(self, t, h, beta=None, J=None):
        """Compute gradient of `phi` with respect to auxiliary variables `t`."""
        beta, J = default(beta, self.beta), default(J, self.J())
        _, _, V_inv = self._phi_prep(t, J)
        if t.size(-1) == 1:
            # scalar t broadcasted to vector t (identical auxiliaries for every spin)
            return (
                beta * self.num_spins - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1).sum(-1)
                - beta / (4.0 * self.dim) * torch.einsum('b i f, b j f, b i k, b k j -> b', h, h, V_inv, V_inv)
            )[:, None]
        else:
            # vector t (different auxiliaries for every spin): very unstable and doesn't really seem to work
            return (
                beta * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1)
                - beta / (4.0 * self.dim) * torch.einsum('b k f, b l f, b i k, b l i -> b i', h, h, V_inv, V_inv)
            )

    def approximate_log_Z(self, t, h, beta=None):
        beta = default(beta, self.beta)
        return 0.5 * self.num_spins * torch.log(math.pi / beta) + self._phi(t, h, beta=beta)

    def approximate_free_energy(self, t, h, beta=None):
        """Compute steepest-descent free energy for large `self.dim`.

        Actually a free energy density since it's divided by `self.dim`". Divide by `self.num_spins`
        to get an average value per site. To calculate thermodynamic quantities, take derivatives
        with respect to gradient-requiring parameters (careful for implicit dependencies when
        evaluating `t` away from the stationary point where phi'(t*) != 0).
        """
        return -1.0 / beta * self.approximate_log_Z(t, h, beta=beta)

    def internal_energy(self, t, h, beta, detach=False):
        """Differentiate log(Z) with respect to the sources `beta` to get <E>.

        If evaluated at `t = t_star`, implicit derivative paths via `t_star(h)`
        drop out because `phi'(t_star) = 0` at the stationary point.
        """
        if not beta.requires_grad:
            beta.requires_grad_()
        energy = batch_jacobian(
            lambda z: -self.approximate_log_Z(t, h, beta=z),
            beta, create_graph=not detach, swapaxes=False)[0]
        if detach:
            return energy.detach()
        return energy

    def magnetizations(self, t, h, beta, detach=False):
        """Differentiate log(Z) with respect to the sources `h_i` to get <sigma_i>.

        If evaluated at `t = t_star`, implicit derivative paths via `t_star(h)`
        drop out because `phi'(t_star) = 0` at the stationary point.
        """
        if not h.requires_grad:
            h.requires_grad_()
        responses = batch_jacobian(
            lambda z: self.approximate_log_Z(t, z, beta=beta),
            h, create_graph=not detach, swapaxes=False)[0]
        if detach:
            return responses.detach()
        return responses

    def forward(
        self,
        h,
        beta=None,
        t0=0.5,
        return_magnetizations=False,
        detach_magnetizations=False,
        return_internal_energy=False,
        detach_internal_energy=False,
    ):
        """Probe model with data `h` and return free energy. Optionally returns (detached) responses."""
        beta = default(beta, self.beta)
        if not h.requires_grad:
            h.requires_grad_()

        # Find t-value for which `phi` appearing in exponential in partition function is stationary.
        t_star = self.diff_root_finder(
            t0*torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype), h, beta=beta,
        )

        # Compute approximate free energy.
        afe = self.approximate_free_energy(t_star, h, beta=beta)

        # Prepare for output and calculate responses and energy if requested.
        out = (afe, t_star,)
        if return_magnetizations:
            out += (self.magnetizations(t_star, h, beta=beta, detach=detach_magnetizations),)
        if return_internal_energy:
            out += (self.internal_energy(t_star, h, beta=beta, detach=detach_internal_energy),)

        # TODO: Add second-order derivatives: specific heat and susceptibilities (fluctuations).

        return out
