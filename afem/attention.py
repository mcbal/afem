from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .models import VectorSpinModel
from .modules import ScaleNorm


@dataclass
class VectorSpinAttentionOutput:
    afe: torch.Tensor
    t_star: torch.Tensor
    magnetizations: Optional[torch.Tensor]
    internal_energy: Optional[torch.Tensor]


class VectorSpinAttention(nn.Module):
    """Basic attention module wrapping around `models.VectorSpinModel`."""

    def __init__(
        self,
        num_spins,
        dim,
        use_scalenorm=True,
        pre_norm=True,
        post_norm=False,
        beta=1.0,
        beta_requires_grad=False,
        beta_parameter=False,
        J_add_external=True,
        J_init_std=None,
        J_parameter=True,
        J_symmetric=True,
        J_traceless=True,
    ):
        super().__init__()

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm

        self.pre_norm = norm_class(dim) if pre_norm else nn.Identity()

        self.spin_model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=beta,
            beta_requires_grad=beta_requires_grad,
            beta_parameter=beta_parameter,
            J_add_external=J_add_external,
            J_init_std=J_init_std,
            J_parameter=J_parameter,
            J_symmetric=J_symmetric,
            J_traceless=J_traceless,
        )

        self.post_norm = norm_class(dim) if post_norm else nn.Identity()

    def forward(
            self,
            x,
            t0,
            beta=None,
            return_magnetizations=True,
            detach_magnetizations=False,
            return_internal_energy=False,
            detach_internal_energy=False,
    ):
        h = self.pre_norm(x)

        out = self.spin_model(
            h,
            t0=t0,
            beta=beta,
            return_magnetizations=return_magnetizations,
            detach_magnetizations=detach_magnetizations,
            return_internal_energy=return_internal_energy,
            detach_internal_energy=detach_internal_energy,
        )
        out

        return VectorSpinAttentionOutput(
            afe=out[0],
            t_star=out[1],
            magnetizations=self.post_norm(out[2]) if return_magnetizations else None,
            internal_energy=out[3] if return_internal_energy else None,
        )
