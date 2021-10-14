# Run forward and backward pass on a `VectorSpinModel` module and plot `phi` and its derivatives.

import matplotlib.pyplot as plt
import numpy as np
import torch

from afem.models import VectorSpinModel

num_spins, dim = 32, 128

# Setup vector-spin model.
model = VectorSpinModel(
    num_spins=num_spins,
    dim=dim,
    beta=1.0,
)

# Create random inputs and initial value for auxiliary variable (see `afem.models.VectorSpinModel`).
x = (torch.randn(1, num_spins, dim) / np.sqrt(dim)).requires_grad_()
t0 = torch.ones(1)

# Run forward pass and get output magnetizations, stationary t_star value, and approximate free energy.
out = model(x, t0=t0, return_afe=True, return_magnetizations=True)

print(
    f'✨ magnetizations: {out.magnetizations}\n✨ t_star {out.t_star.detach()}\n✨ approximate free energy: {out.afe.detach()}'
)

# Run backward on sum of free energies across batch dimension.
out.afe.sum().backward()

##############
# PLOT STUFF #
##############

if x.size(0) == 1 and t0.numel() == 1:

    def filter_array(a, threshold=250):
        idx = np.where(np.abs(a) > threshold)
        a[idx] = np.nan
        return a

    t_star = out.t_star[0][0].detach().numpy()
    afe_star = out.afe[0][0].detach().numpy()
    grad_phi_star = np.array(model._jac_phi(out.t_star, x[:1, :, :]).detach().numpy())
    # Pick a range and resolution for `t`.
    t_min, t_max, t_step = 0.0, 3.0, 0.001
    t_range = torch.arange(t_min, t_max, t_step)[:, None]
    # Calculate function evaluations for every point on grid and plot.
    phis = np.array(model._phi(t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_phis = np.array(model._jac_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_grad_phis = np.array(model._hess_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy()).squeeze()

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].set_title(
        f"Found root of $\\varphi$'(t) at t = {t_star:.4f} (afe = {afe_star:.4f})")
    axs[0].plot(t_range.numpy().squeeze(), filter_array(phis), 'r-')
    axs[0].axvline(x=out.t_star[0].detach().numpy())
    axs[0].set_ylabel("$\\varphi$(t)")

    axs[1].plot(t_range.numpy().squeeze(), filter_array(grad_phis), 'r-')
    axs[1].axvline(x=out.t_star[0].detach().numpy())
    axs[1].axhline(y=0.0)
    axs1_inset = axs[1].inset_axes([0.45, 0.20, 0.47, 0.47])
    axs1_inset.plot(t_range.numpy().squeeze(), filter_array(grad_phis), 'r-')
    axs1_inset.axvline(x=out.t_star[0].detach().numpy())
    axs1_inset.axhline(y=0.0)
    x1, x2, y1, y2 = t_star-0.05, t_star+0.05, grad_phi_star-5, grad_phi_star+5
    axs1_inset.set_xlim(x1, x2)
    axs1_inset.set_ylim(y1, y2)
    axs1_inset.set_xticklabels('')
    axs1_inset.set_yticklabels('')
    axs[1].indicate_inset_zoom(axs1_inset, edgecolor="black")
    axs[1].set_ylabel("$\\varphi$'(t)")

    axs[2].plot(t_range.numpy().squeeze(), filter_array(grad_grad_phis), 'r-')
    axs[2].axvline(x=out.t_star[0].detach().numpy())
    axs[2].set_xlim([t_min, t_max])
    axs[2].set_xlabel('t')
    axs[2].set_ylabel("$\\varphi$''(t)")

    fig.align_ylabels(axs)
    # plt.savefig('plot.png', bbox_inches='tight')
    plt.show()
