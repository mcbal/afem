import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from afem.models import VectorSpinModel
import numpy as np
import matplotlib.pyplot as plt

# class TestAnalyticalGradients(unittest.TestCase):
#     def test_phi_t_scalar(self):
#         num_spins, dim = 11, 7

#         model = VectorSpinModel(
#             num_spins=num_spins,
#             dim=dim,
#             beta=1.0 / np.sqrt(dim),
#         ).double()

#         h = torch.randn(1, num_spins, dim).double()
#         t0 = torch.rand(1, 1).double().requires_grad_()

#         analytical_grad = model._grad_t_phi(t0, h)
#         numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

#         self.assertTrue(
#             torch.allclose(analytical_grad, numerical_grad)
#         )

#     def test_phi_t_vector(self):
#         num_spins, dim = 11, 7

#         model = VectorSpinModel(
#             num_spins=num_spins,
#             dim=dim,
#             beta=1.0 / np.sqrt(dim),
#         ).double()

#         h = torch.randn(1, num_spins, dim).double()
#         t0 = torch.rand(1, num_spins).double().requires_grad_()

#         analytical_grad = model._grad_t_phi(t0, h)
#         numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

#         self.assertTrue(
#             torch.allclose(analytical_grad, numerical_grad)
#         )


class TestRootFinding(unittest.TestCase):
    def test_phi_root_finder(self):
        num_spins, dim = 32, 128

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
            J_init_std=1.0 / np.sqrt(num_spins*dim**2),
        )

        h = torch.randn(1, num_spins, dim)  # / np.sqrt(num_spins*dim**2)
        t0 = torch.ones(1, 1).requires_grad_()
        t0 = 1.0*t0

        print(f't_0: {t0}')
        print(t0.is_leaf)

        print(f'phi_0: {model._phi(t0, h)}')

        t_star = model.diff_root_finder(t0, h)

        # Evaluate `phi` at stationary point.
        print(f't_star: {t_star}')
        print(f'grad_t_star_phi (0?): {model._grad_t_phi(t_star, h)}')

        bla = model._phi(t_star, h)

        print(f'phi_star: {bla}')

        bla.sum().backward()

        print(model._J)
        print(model._J.grad)

        t = torch.arange(0.25, 3.0, 0.01).unsqueeze(-1).unsqueeze(0)
        out = [model._phi(x, h).detach().numpy().squeeze() for _, x in enumerate(t)]
        out_grad = [model._grad_t_phi(x, h).detach().numpy().squeeze() for _, x in enumerate(t)]
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(t.numpy().squeeze(), out[0], 'r.-')
        ax1.axvline(x=t_star.detach().numpy())
        ax2.plot(t.numpy().squeeze(), out_grad[0], 'r.-')
        ax2.axvline(x=t_star.detach().numpy())
        plt.show()

        # print([x for x in model.parameters()])

        # self.assertTrue(
        #     torch.allclose(analytical_grad, numerical_grad)
        # )


# class TestRootFindingGradients(unittest.TestCase):
#     def test_phi_rootfinding_gradients(self):
#         pass


if __name__ == '__main__':
    unittest.main()
