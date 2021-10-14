# Run forward and backward pass on a two-spin `VectorSpinAttention` module and plot `phi`.

import matplotlib.pyplot as plt

import numpy as np
import torch

from afem.models import VectorSpinModel

num_spins, dim = 2, 64

# Setup vector-spin model.
model = VectorSpinModel(
    num_spins=num_spins,
    dim=dim,
    beta=1.0,
)

# Create random inputs and initial value for auxiliary variable (see `afem.models.VectorSpinModel`).
x = (torch.randn(1, num_spins, dim) / np.sqrt(dim)).requires_grad_()
t0 = 0.5*torch.ones(num_spins)

# Run forward pass and get output magnetizations, stationary t_star value, and approximate free energy.
out = model(x, t0=t0, return_afe=True, return_magnetizations=True)

print(
    f'✨ t_star {out.t_star.detach()}\n✨ magnetizations: {out.magnetizations}\n✨ approximate free energy: {out.afe.detach()}'
)

##############
# PLOT STUFF #
##############


def filter_array(a, threshold=50):
    idx = np.where(np.abs(a) > threshold)
    a[idx] = np.nan
    return a


fig = plt.figure()
ax = plt.axes()

t_star = out.t_star[0].detach().numpy()
t_min, t_max, t_step = 0.2, 3.0, 0.01
t_range = torch.arange(t_min, t_max, t_step)

grid_x, grid_y = torch.meshgrid(torch.arange(t_min, t_max, t_step), torch.arange(t_min, t_max, t_step))
grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

phis = np.array(model._phi(grid, x[:1, :, :].repeat(grid.shape[0], 1, 1)
                           ).detach().reshape(grid_x.shape[0], grid_x.shape[0]).numpy())

artist = ax.contourf(grid_x, grid_y, filter_array(phis), cmap='viridis', levels=20)
ax.scatter(t_star[0], t_star[1], color="red", s=50, marker="o")
ax.axvline(x=t_star[0], linestyle='--', color="red")
ax.axhline(y=t_star[1], linestyle='--', color="red")
ax.set_title(f'$\\varphi(t)$ for two spins (t = ({t_star[0]:.4f}, {t_star[1]:.4f}))')
ax.set_xlabel('$t_0$')
ax.set_ylabel('$t_1$')
plt.colorbar(artist)
# plt.savefig('plot.png', bbox_inches='tight')
plt.show()
