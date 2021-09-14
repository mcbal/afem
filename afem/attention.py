import torch.nn as nn

from .models import VectorSpinModel
from .modules import ScaleNorm


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
        J_symmetric=False,
        J_traceless=False,
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

    def forward(self, x, t0=0.5):
        h = self.pre_norm(x)

        afe, t_star, responses = self.spin_model(h, t0=t0, return_magnetizations=True)

        responses = self.post_norm(responses)

        return responses, afe, t_star
