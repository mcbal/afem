import torch

from .utils import batch_jacobian


def _reset_singular_jacobian(x):
    """Check for singular scalars/matrices in batch; reset singular scalars/matrices to ones."""
    bad_idxs = torch.isclose(x, torch.zeros_like(
        x)) if x.size(-1) == 1 else torch.isclose(torch.det(x), torch.zeros_like(x))
    if bad_idxs.any():
        print(
            f'🔔 Encountered {bad_idxs.sum()} singular Jacobian(s) in current batch during root-finding. Jumping to somewhere else.'
        )
        breakpoint()
        x[bad_idxs] = 1.0
    return x


def newton(f, z_init, analytical_jac_f=None, max_iter=40, tol=1e-4):
    """Basic Newton method for finding roots in k variables.

    Requires adding `create_graph=True` in backward hook of forward pass of
    `rootfind.RootFind` to get Jacobian of gradient function. This can be avoided
    if an analytical expression for the Jacobian is available, which can be passed
    along to the solver via the `analytical_jac_f` argument.

    See https://implicit-layers-tutorial.org/introduction/.
    """
    def jacobian(f, z):
        return analytical_jac_f(z) if analytical_jac_f is not None else batch_jacobian(f, z)

    def g(z):
        kaka = _reset_singular_jacobian(jacobian(f, z))
        # print(kaka)
        bla = torch.linalg.solve(kaka, f(z))
        # print('inside g', z, bla, f(z))
        return z - bla

    z_prev, z, n_steps, trace = z_init, g(z_init), 0, []

    trace.append(torch.linalg.norm(f(z_init)).detach())
    trace.append(torch.linalg.norm(f(z)).detach())

    while torch.linalg.norm(z_prev - z) > tol and n_steps < max_iter:
        z_prev, z = z, g(z)
        n_steps += 1
        trace.append(torch.linalg.norm(f(z)).detach())

    # print(z_init, trace, z)
    # print(trace)

    return {
        'result': z,
        'n_steps': n_steps,
        'trace': trace,
        'max_iter': max_iter,
        'tol': tol,
    }
