# Run forward and backward pass on a `VectorSpinAttention` module and plot `phi` and its derivatives.

import matplotlib.pyplot as plt
import numpy as np
import torch

from afem.attention import VectorSpinAttention

num_spins, dim = 11, 17

# Setup model.
attention = VectorSpinAttention(
    num_spins=num_spins,
    dim=dim,
    pre_norm=False,  # otherwise plots will not match up (normalizing happens inside `VectorSpinAttention`)
    beta=1.0,
    J_add_external=True,
    J_symmetric=True,
    J_traceless=True,
)

# Create random inputs and initial value for auxiliary variable (see `afem.models.VectorSpinModel`).
x = torch.randn(1, num_spins, dim).requires_grad_()
t0 = torch.ones(1)

# Run forward pass and get output magnetizations, stationary t_star value, and approximate free energy.
out = attention(x, t0=t0, return_afe=True, return_magnetizations=True)

print(
    f'✨ t_star {out.t_star.detach()}\n✨ magnetizations: {out.magnetizations}\n✨ approximate free energy: {out.afe.detach()}'
)

# Run backward on sum of free energies across batch dimension.
out.afe.sum().backward()

#########################################################################################
# Plot internally-used function `phi(t)` and its derivatives for first element in batch. #
#########################################################################################

if t0.numel() == x.size(0):

    def filter_array(a, threshold=1e2):
        idx = np.where(np.abs(a) > threshold)
        a[idx] = np.nan
        return a

    # Pick a range and resolution for `t`.
    t_range = torch.arange(0.0, 3.0, 0.0001)[:, None]
    # Calculate function evaluations for every point on grid and plot.
    phis = np.array(attention.spin_model._phi(t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_phis = np.array(attention.spin_model._jac_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_grad_phis = np.array(attention.spin_model._hess_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy()).squeeze()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title(f"Found root of phi'(t) at t = {out.t_star[0][0].detach().numpy():.4f}")
    ax1.plot(t_range.numpy().squeeze(), filter_array(phis), 'r-')
    ax1.axvline(x=out.t_star[0].detach().numpy())
    ax1.set_ylabel("phi(t)")
    ax2.plot(t_range.numpy().squeeze(), filter_array(grad_phis), 'r-')
    ax2.axvline(x=out.t_star[0].detach().numpy())
    ax2.axhline(y=0.0)
    ax2.set_ylabel("phi'(t)")
    ax3.plot(t_range.numpy().squeeze(), filter_array(grad_grad_phis), 'r-')
    ax3.axvline(x=out.t_star[0].detach().numpy())
    ax3.axhline(y=0.0)
    ax3.set_xlim([0, 3])
    ax3.set_xlabel('t')
    ax3.set_ylabel("phi''(t)")
    plt.show()
