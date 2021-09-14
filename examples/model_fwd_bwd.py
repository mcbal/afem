# Run forward and backward pass on a `VectorSpinModel` and plot `phi` and `phi'`.

import matplotlib.pyplot as plt
import numpy as np
import torch

from afem.attention import VectorSpinAttention
from torchviz import make_dot

num_spins, dim = 11, 17

# Setup model and inputs.
attention = VectorSpinAttention(
    num_spins=num_spins,
    dim=dim,
    J_add_external=False,
    pre_norm=True,
    post_norm=False,
)

x = torch.randn(1, num_spins, dim).requires_grad_()

# Run forward pass and get responses, approximate free energy, and stationary t_star value.
responses, afe, t_star = attention(x)

print(f'✨ afe / N: {afe.item()/num_spins:.4f}, \n✨ t_star {t_star.item():.4f}, \n✨ responses: {responses}')

# Run backward on sum of free energies across batch dimension.
afe.sum().backward()

#########################################################################################
# Plot internally-used function `phi(t)` and its derivative for first element in batch. #
#########################################################################################


def filter_array(a, threshold=1e2):
    idx = np.where(np.abs(a) > threshold)
    a[idx] = np.nan
    return a


# Pick a range and resolution for `t`.
t_range = torch.arange(0.0, 3.0, 0.001)[:, None]
# Calculate function evaluations for every point on grid and plot.
out = np.array(attention.spin_model._phi(t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
out_grad = np.array(attention.spin_model._grad_t_phi(
    t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title(f"(Hopefully) Found root of phi'(t) at t = {t_star[0][0].detach().numpy()}")
ax1.plot(t_range.numpy().squeeze(), filter_array(out), 'r-')
ax1.axvline(x=t_star[0].detach().numpy())
ax1.set_ylabel("phi(t)")
ax2.plot(t_range.numpy().squeeze(), filter_array(out_grad), 'r-')
ax2.axvline(x=t_star[0].detach().numpy())
ax2.axhline(y=0.0)
ax2.set_xlabel('t')
ax2.set_ylabel("phi'(t)")
plt.show()
