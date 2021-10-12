# Sweep across inverse temperature `beta` for fixed inputs to a `VectorSpinModel` module
# and plot `phi` and its derivatives.

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib import animation

from afem.models import VectorSpinModel

num_spins, dim = 32, 128
beta_min, beta_max, beta_num_steps = -0.6, 1.0, 100  # log10-space
plot_values_cutoff = 500

x = (torch.randn(1, num_spins, dim) / np.sqrt(dim)).requires_grad_()
t0 = 0.5*torch.ones(1)


def filter_array(a, threshold=plot_values_cutoff):
    idx = np.where(np.abs(a) > threshold)
    a[idx] = np.nan
    return a


def simple_update(frame, fig, axs):

    print(f'{frame:.4f} / {10**beta_max}')

    model = VectorSpinModel(
        num_spins=num_spins,
        dim=dim,
        beta=frame,
    )
    out = model(x, t0=t0, return_afe=True, return_magnetizations=True)

    t_star = out.t_star[0][0].detach().numpy()
    afe_star = out.afe[0][0].detach().numpy()
    t_min, t_max, t_step = 0.0, 3.0, 0.001
    t_range = torch.arange(t_min, t_max, t_step)[:, None]
    phis = np.array(model._phi(t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_phis = np.array(model._jac_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
    grad_grad_phis = np.array(model._hess_phi(
        t_range, x[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy()).squeeze()

    axs[0].clear()
    axs[0].set_title(
        f"beta = {frame:.4f}, t = {t_star:.4f}, afe = {afe_star:.4f}")
    axs[0].plot(t_range.numpy().squeeze(), filter_array(phis), 'r-')
    axs[0].axvline(x=out.t_star[0].detach().numpy())
    axs[0].set_ylim([-plot_values_cutoff, plot_values_cutoff])
    axs[0].set_ylabel("phi(t)")

    axs[1].clear()
    axs[1].plot(t_range.numpy().squeeze(), filter_array(grad_phis), 'r-')
    axs[1].axvline(x=out.t_star[0].detach().numpy())
    axs[1].axhline(y=0.0)
    axs[1].set_ylim([-plot_values_cutoff, 100])
    axs[1].set_ylabel("phi'(t)")

    axs[2].clear()
    axs[2].plot(t_range.numpy().squeeze(), filter_array(grad_grad_phis, 2*plot_values_cutoff), 'r-')
    axs[2].axvline(x=out.t_star[0].detach().numpy())
    axs[2].set_xlim([t_min, t_max])
    axs[2].set_xlabel('t')
    axs[2].set_ylim([0, 2*plot_values_cutoff])
    axs[2].set_ylabel("phi''(t)")

    fig.align_ylabels(axs)


def simple_animation():
    fig, axs = plt.subplots(3, 1, sharex=True)
    ani = animation.FuncAnimation(
        fig, simple_update, frames=np.logspace(beta_min, beta_max, beta_num_steps),
        fargs=(fig, axs,),
        interval=100,
        repeat_delay=100,
    )
    ani.save('animation.gif', writer='imagemagick')


simple_animation()
