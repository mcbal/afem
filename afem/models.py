import math

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .rootfind import RootFind
from .solvers import broyden, newton
from .utils import default, batched_eye_like, batched_jacobian_for_scalar_fun, batched_jacobian_for_vector_fun


class VectorSpinModel(nn.Module):

    def __init__(
        self,
        num_spins,
        dim,
        beta=1.0,
        beta_training=False,
        J_init_std=None,
        J_symmetric=True,
        J_traceless=True,
        J_training=True,
    ):
        super().__init__()

        self.num_spins = num_spins
        self.dim = dim

        # Setup inverse temperature.
        if beta_training:
            self.beta = nn.Parameter(torch.as_tensor(beta))
        else:
            self.register_buffer('beta', torch.as_tensor(beta))

        # Setup couplings.
        J = torch.zeros(num_spins, num_spins).normal_(
            0, J_init_std if J_init_std is not None else 1.0 / np.sqrt(num_spins*dim)
        )
        self.J_symmetric = J_symmetric
        self.J_traceless = J_traceless
        if J_training:
            self._J = nn.Parameter(J)
        else:
            self.register_buffer('_J', J)

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

    def _prepare_sources(self, x):
        return x

    def _phi_prep(self, t, J):
        """Construct `V` and its inverse given `t` and couplings `J`."""
        assert t.ndim == 2, f'Tensor `t` should have either shape (batch, 1) or (batch, N) but found shape {t.shape}'
        t = t.repeat(1, self.num_spins) if t.shape[-1] == 1 else t
        V = torch.diag_embed(t) - repeat(J, 'i j -> b i j', b=t.shape[0])
        V_inv = torch.linalg.solve(V, batched_eye_like(V))
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
        _, _, V_inv = self._phi_prep(t, J)
        if t.shape[-1] == 1:
            # scalar t broadcasted to vector t (identical auxiliaries for every spin)
            return (
                beta * self.num_spins * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv,
                                                                                  dim1=-2, dim2=-1).sum(-1, keepdim=True)
                - beta / (4.0 * self.dim) * torch.einsum('b i f, b j f, b i k, b k j -> b', h, h, V_inv, V_inv)[:, None]
            )
        else:
            # vector t (different auxiliaries for every spin): very unstable and doesn't really seem to work
            return (
                beta * torch.ones_like(t) - 0.5 * torch.diagonal(V_inv, dim1=-2, dim2=-1)
                - beta / (4.0 * self.dim) * torch.einsum('b k f, b l f, b i k, b l i -> b i', h, h, V_inv, V_inv)
            )

    # def _grad_grad_t_phi(self, t, h, beta=None, J=None):
    #     """Compute gradient of `phi` with respect to auxiliary variables `t`."""
    #     beta, J = default(beta, self.beta), default(J, self.J())
    #     _, _, V_inv = self._phi_prep(t, J)
    #     if t.shape[-1] == 1:
    #         # scalar t broadcasted to vector t (identical auxiliaries for every spin)
    #         return (
    #             0.5 * torch.einsum('b i k, b k i -> b', V_inv, V_inv)[:, None]
    #             + beta / (4.0 * self.dim) * torch.einsum('b j i, b j f, b k f, b k l, b i l-> b',
    #                                                      V_inv, h, h, V_inv, V_inv)[:, None]
    #             + beta / (4.0 * self.dim) * torch.einsum('b i j, b k j, b k f, b l f, b i l-> b',
    #                                                      V_inv, V_inv, h, h, V_inv)[:, None]
    #         )
    #     else:
    #         # vector t (different auxiliaries for every spin): very unstable and doesn't really seem to work
    #         return NotImplementedError()

    def approximate_free_energy(self, t, h, beta):
        beta = default(beta, self.beta)
        return - beta**-1 * (0.5*torch.log(math.pi / beta) + self.num_spins**-1 * self._phi(t, h, beta=beta))

    def forward(self, x, beta=None, return_responses=False, detach_responses=False):
        """Probe model with data `x`. Return free energy and responses."""

        beta = default(beta, self.beta)
        # Prep by padding and (optionally) adding noise to data inputs.
        h = self._prepare_sources(x)

        # Find t-value for which `phi` appearing in exponential in partition function is stationary.
        t0 = (0.5*torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
        t_star = self.diff_root_finder(
            t0, h, beta=beta,
            # solver_fwd_grad_f=lambda t: self._grad_grad_t_phi(t, h),
            # solver_bwd_grad_f=lambda t: self._grad_grad_t_phi(t, h),
        )

        # print(t_star.requires_grad, t_star.grad_fn)

        # Compute approximate free energy.
        afe = self.approximate_free_energy(t_star, h, beta=beta)

        out = (afe, t_star,)

        # _, _, V_inv = self._phi_prep(t_star, self.J())
        # print(beta / (4.0 * self.num_spins * self.dim) * 2 * (torch.einsum('b i j, b j f -> b i f', V_inv, h)))
        # print(f'V_inv: {V_inv}')

        if return_responses:
            responses = batched_jacobian_for_scalar_fun(
                lambda z: self.approximate_free_energy(t_star, z, beta=beta), h.clone(), create_graph=not detach_responses
            ) + self._grad_t_phi(t_star, h) * batched_jacobian_for_vector_fun(
                lambda z: self.diff_root_finder(t0, z, beta=beta), h.clone(), create_graph=not detach_responses
            ).squeeze(0)
            # responses = (batched_jacobian_for_scalar_fun(
            #     lambda z: self.approximate_free_energy(t_star, z, beta=beta), h, create_graph=not detach_responses
            # )) + (batched_jacobian_for_scalar_fun(
            #     lambda z: self.diff_root_finder(
            #         t0, z, beta=beta,
            #         # solver_fwd_grad_f=lambda t: self._grad_grad_t_phi(t, h),
            #         # solver_bwd_grad_f=lambda t: self._grad_grad_t_phi(t, h),
            #     ), h, create_graph=not detach_responses
            # ))
            if detach_responses:
                responses = responses.detach()
            out = out + (responses,)

        return out
