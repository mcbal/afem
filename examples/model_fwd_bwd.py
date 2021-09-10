# Run forward and backward pass on a `VectorSpinModel` and plot `phi` and `phi'`.

import matplotlib.pyplot as plt
import numpy as np
import torch

from afem.models import VectorSpinModel
from torchviz import make_dot

num_spins, dim = 32, 128

# Setup model and inputs.
model = VectorSpinModel(
    num_spins=num_spins,
    dim=dim,
    beta=1.0,
)
x = torch.randn(1, num_spins, dim).requires_grad_()

# Run forward pass and get approximate free energy, magnetizations, and internal energy.
afe, t_star, magnetizations, energy = model(x, return_magnetizations=True, return_internal_energy=True)
print(afe, t_star, magnetizations, energy)

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
t_range = torch.arange(0.0, 2.0, 0.001)[None, :, None]
# Calculate function evaluations for every point on grid and plot.
out = [model._phi(tt, x[:1, :, :]).detach().numpy().squeeze() for _, tt in enumerate(t_range)]
out_grad = [model._grad_t_phi(tt, x[:1, :, :]).detach().numpy().squeeze() for _, tt in enumerate(t_range)]
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title(f"(Hopefully) Found root of phi'(t) at t = {t_star[0][0].detach().numpy()}")
ax1.plot(t_range.numpy().squeeze(), filter_array(out[0]), 'r-')
ax1.axvline(x=t_star[0].detach().numpy())
ax1.set_ylabel("phi(t)")
ax2.plot(t_range.numpy().squeeze(), filter_array(out_grad[0]), 'r-')
ax2.axvline(x=t_star[0].detach().numpy())
ax2.axhline(y=0.0)
ax2.set_xlabel('t')
ax2.set_ylabel("phi'(t)")
plt.show()
