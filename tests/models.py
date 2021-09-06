import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from afem.models import VectorSpinModel


class TestAnalyticalGradients(unittest.TestCase):
    def test_phi_t_scalar(self):
        num_spins, dim = 11, 7

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0 / np.sqrt(dim),
        ).double()

        h = torch.randn(1, num_spins, dim).double()
        t0 = torch.rand(1, 1).double().requires_grad_()

        analytical_grad = model._grad_t_phi(t0, h)
        numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

        self.assertTrue(
            torch.allclose(analytical_grad, numerical_grad)
        )

    def test_phi_t_vector(self):
        num_spins, dim = 11, 7

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0 / np.sqrt(dim),
        ).double()

        h = torch.randn(1, num_spins, dim).double()
        t0 = torch.rand(1, num_spins).double().requires_grad_()

        analytical_grad = model._grad_t_phi(t0, h)
        numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

        self.assertTrue(
            torch.allclose(analytical_grad, numerical_grad)
        )


class TestRootFinding(unittest.TestCase):
    def test_phi_root_finder(self):
        num_spins, dim = 11, 7

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0 / np.sqrt(dim),
        ).double()

        h = torch.randn(1, num_spins, dim).double()
        t0 = 10*torch.ones(1, 1).double().requires_grad_()

        print(f't_0: {t0}')

        print(f'phi_0: {model._phi(t0, h)}')

        t_star = model.diff_root_finder(t0, h)

        # Evaluate `phi` at stationary point.
        print(t_star)
        print(f'grad_t_star_phi (0?): {model._grad_t_phi(t_star, h)}')
        print(f'phi_star: {model._phi(t_star, h)}')

        # self.assertTrue(
        #     torch.allclose(analytical_grad, numerical_grad)
        # )


# class TestRootFindingGradients(unittest.TestCase):
#     def test_phi_rootfinding_gradients(self):
#         pass


if __name__ == '__main__':
    unittest.main()
