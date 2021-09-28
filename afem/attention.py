import numpy as np
import torch.nn as nn

from .models import VectorSpinModel
from .modules import ScaleNorm
from .utils import exists


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
        J_add_external=False,
        J_symmetric=True,
        J_traceless=True,
        J_parameter=True,
    ):
        super().__init__()

        self.pre_norm = (ScaleNorm(scale=np.sqrt(dim)) if use_scalenorm else nn.LayerNorm(dim)
                         ) if pre_norm else nn.Identity()

        self.spin_model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=beta,
            beta_requires_grad=beta_requires_grad,
            beta_parameter=beta_parameter,
            J_add_external=J_add_external,
            J_symmetric=J_symmetric,
            J_traceless=J_traceless,
            J_parameter=J_parameter,
        )

        self.post_norm = (ScaleNorm(scale=np.sqrt(dim)) if use_scalenorm else nn.LayerNorm(dim)
                          ) if post_norm else nn.Identity()

    def forward(
            self,
            x,
            t0,
            beta=None,
            return_afe=False,
            return_magnetizations=True,
            return_internal_energy=False,
            return_log_prob=False,
            use_analytical_grads=True,
    ):
        h = self.pre_norm(x)

        out = self.spin_model(
            h,
            t0=t0,
            beta=beta,
            return_afe=return_afe,
            return_magnetizations=return_magnetizations,
            return_internal_energy=return_internal_energy,
            return_log_prob=return_log_prob,
            use_analytical_grads=use_analytical_grads,
        )

        out.magnetizations = self.post_norm(out.magnetizations) if exists(out.magnetizations) else None

        return out
