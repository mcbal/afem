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


class TestRootFindingGradients(unittest.TestCase):

    def test_vector_spin_model_forward(self):
        num_spins, dim = 11, 32

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
        ).double()

        x = (torch.randn(1, num_spins, dim) / np.sqrt(dim)).double()

        self.assertTrue(
            gradcheck(
                model,
                x.requires_grad_(),
                eps=1e-4,
                atol=1e-3,
                check_undefined_grad=True,
            )
        )


if __name__ == '__main__':
    unittest.main()
