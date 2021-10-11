import numpy as np
import torch
import torch.nn as nn

from .models import VectorSpinModel
from .utils import exists


class VectorSpinAttention(nn.Module):
    """Basic attention module wrapping around `models.VectorSpinModel`."""

    def __init__(
        self,
        num_spins,
        dim,
        pre_norm=True,
        post_norm=False,
        beta=1.0,
        beta_requires_grad=False,
        beta_parameter=False,
        J_external=False,
        J_symmetric=True,
        J_traceless=True,
        solver_fwd_max_iter=40,
        solver_fwd_tol=1e-5,
        solver_bwd_max_iter=40,
        solver_bwd_tol=1e-5,
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

        self.spin_model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=beta,
            beta_requires_grad=beta_requires_grad,
            beta_parameter=beta_parameter,
            J_external=J_external,
            J_symmetric=J_symmetric,
            J_traceless=J_traceless,
            solver_fwd_max_iter=solver_fwd_max_iter,
            solver_fwd_tol=solver_fwd_tol,
            solver_bwd_max_iter=solver_bwd_max_iter,
            solver_bwd_tol=solver_bwd_tol,
        )

        self.post_norm = nn.LayerNorm(dim) if post_norm else nn.Identity()

    def forward(
            self,
            x,
            t0=None,
            beta=None,
            return_afe=False,
            return_magnetizations=True,
            return_internal_energy=False,
            return_log_prob=False,
            use_analytical_grads=True,
    ):
        h = self.pre_norm(x) / np.sqrt(self.spin_model.dim)

        out = self.spin_model(
            h,
            t0=t0 if exists(t0) else torch.ones_like(x[0, :, 0]),
            beta=beta,
            return_afe=return_afe,
            return_magnetizations=return_magnetizations,
            return_internal_energy=return_internal_energy,
            return_log_prob=return_log_prob,
            use_analytical_grads=use_analytical_grads,
        )

        out.magnetizations = self.post_norm(out.magnetizations) if exists(out.magnetizations) else None

        return out
