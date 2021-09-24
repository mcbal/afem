# Run forward and backward pass on a `VectorSpinModel` and plot `phi` and `phi'`.

import matplotlib.pyplot as plt
import numpy as np
import torch

from afem.attention import VectorSpinAttention

num_spins, dim = 11, 17

# Setup model and inputs.
attention = VectorSpinAttention(
    num_spins=num_spins,
    dim=dim,
)

x = torch.randn(1, num_spins, dim).requires_grad_()
t0 = torch.ones(1).requires_grad_()  # scalar t

# Run forward pass and get responses, approximate free energy, and stationary t_star value.
out = attention(x, t0=t0, return_magnetizations=True)

print(f'✨ afe / N: {out.afe.item()/num_spins:.4f}, \n✨ t_star {out.t_star}, \n✨ magnetizations: {out.magnetizations}')

# Run backward on sum of free energies across batch dimension.
out.afe.sum().backward()

#########################################################################################
# Plot internally-used function `phi(t)` and its derivative for first element in batch. #
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
    ax1.set_title(f"(Hopefully) Found root of phi'(t) at t = {out.t_star[0][0].detach().numpy()}")
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
