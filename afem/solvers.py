# Root-finding algorithms.

import torch
import numpy as np
from torch.autograd.functional import jacobian


#############################################################################
#
# Broyden's method, a quasi-Newton method for finding roots in k variables.
#
# Based on the implementations in
# - https://github.com/locuslab/deq/
# - https://github.com/akbir/deq-jax
#
#############################################################################


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    while alpha1 > amin:
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    return None, phi_a1, ite


def line_search(update, x0, g0, g, n_step=0, on=True):
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, _, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (bsz, k)
    # part_Us: (bsz, k, max_iter)
    # part_VTs: (bsz, max_iter, k)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bi, bid -> bd', x, part_Us)
    return -x + torch.einsum('bd, bdi -> bi', xTU, part_VTs)


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (bsz, k)
    # part_Us: (bsz, k, max_iter)
    # part_VTs: (bsz, max_iter, k)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdi, bi -> bd', part_VTs, x)
    return -x + torch.einsum('bid, bd -> bi', part_Us, VTx)


def broyden(g, x0, max_iter, tol=1e-4, armijo_line_search=True):
    bsz, k = x0.size()
    device = x0.device

    # Initialize 0-th iteration.
    x, gx = x0, g(x0)
    # Initialize tensors for inverse Jacobian approximation.
    Us = torch.zeros(bsz, k, max_iter).to(device)
    VTs = torch.zeros(bsz, max_iter, k).to(device)
    # Propose initial update direction.
    update = -matvec(Us[:, :, :0], VTs[:, :0], gx)

    n_step, trace, min_x, min_gx, min_step, min_objective = 0, [], x, gx, 0, torch.norm(gx).item()
    prot_break = False
    protect_thres = 1e5 * k

    while n_step < max_iter:
        x, gx, delta_x, delta_gx, _ = line_search(update, x, gx, g, n_step=n_step, on=armijo_line_search)

        n_step += 1

        trace.append(torch.linalg.norm(gx).item())
        if trace[-1] < min_objective:
            min_x, min_gx = x.clone().detach(), gx.clone().detach()
            min_objective = trace[-1]
            min_step = n_step

        new_objective = trace[-1]
        print(trace)
        # breakpoint()

        if new_objective < tol:
            break
        # if new_objective < 3 * eps and n_step > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(
        #         trace_dict[stop_mode][-30:]) < 1.3:
        #     # if there's hardly been any progress in the last 30 steps
        #     break
        if new_objective > trace[0] * protect_thres:
            prot_break = True  # blowing up
            break

        # Update inverse Jacobian approximation using Sherman–Morrison formula.
        part_Us, part_VTs = Us[:, :, :n_step-1], VTs[:, :n_step-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bi, bi -> b', vT, delta_gx)[:, None]
        VTs[:, n_step-1] = vT.nan_to_num()
        Us[:, :, n_step-1] = u.nan_to_num()
        # Propose new update direction.
        update = -matvec(Us[:, :, :n_step], VTs[:, :n_step], gx)

    return {"result": min_x,
            "min_objective": min_objective,
            "min_step": min_step,
            "prot_break": prot_break,
            "trace": trace,
            "tol": tol,
            "max_iter": max_iter}


#################################################################################
#
# Basic Newton method for finding roots in k variables.
#
# Requires adding `create_graph=True` in backward hook of forward pass of
# `rootfind.RootFind` to get Jacobian of gradient function. Might be better
# to stick to Broyden's method (which does not use autograd functionalities)
# to avoid running into weird autodiff behaviour with implicit differentiation.
#
#################################################################################


@torch.enable_grad()
def batch_jacobian(f, x):
    """https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6"""
    def f_sum(x): return torch.sum(f(x), dim=0)
    return jacobian(f_sum, x).permute(1, 0, 2)


def newton(f, z_init, max_iter=40, tol=1e-4):
    def g(z):
        return z - torch.linalg.solve(batch_jacobian(f, z), f(z))
    z_prev, z, num_steps = z_init, g(z_init), 0
    while torch.linalg.norm(z_prev - z) > tol and num_steps < max_iter:
        z_prev, z = z, g(z)
        num_steps += 1
    return {'result': z}
