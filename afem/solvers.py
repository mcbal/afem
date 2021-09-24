import torch

from .utils import batch_jacobian


def _reset_singular_jacobian(x):
    """Check for singular scalars/matrices in batch; reset singular scalars/matrices to ones."""
    bad_idxs = torch.isclose(x, torch.zeros_like(
        x)) if x.size(-1) == 1 else torch.isclose(torch.linalg.det(x), torch.zeros_like(x[:, 0, 0]))
    if bad_idxs.any():
        print(
            f'🔔 Encountered {bad_idxs.sum()} singular Jacobian(s) in current batch during root-finding. Inserted NaNs so optimizer can skip batch.'
        )
        x[bad_idxs] = float('NaN')
    return x


def newton(f, z_init, grad_f=None, max_iter=40, tol=1e-4):
    """Basic Newton method for finding roots in k variables.

    When `grad_f` is None, the backward hook defined in the forward pass of
    `rootfind.RootFind` needs `create_graph=True` to compute numerical gradients.
    This can be avoided if an analytical expression for the Jacobian is available,
    which can be passed along to the solver via the `solver_fwd_grad_f` argument.
    """
    def jacobian(f, z):
        return grad_f(z) if grad_f is not None else batch_jacobian(f, z)

    def g(z):
        return z - torch.linalg.solve(_reset_singular_jacobian(jacobian(f, z)), f(z))

    z_prev, z, n_steps, trace = z_init, g(z_init), 0, []
    trace.append(torch.linalg.norm(f(z_init)).detach())
    trace.append(torch.linalg.norm(f(z)).detach())

    while torch.linalg.norm(z_prev - z) > tol and n_steps < max_iter:
        z_prev, z = z, g(z)
        trace.append(torch.linalg.norm(f(z)).detach())
        n_steps += 1

    return {
        'result': z,
        'n_steps': n_steps,
        'trace': trace,
        'max_iter': max_iter,
        'tol': tol,
    }
