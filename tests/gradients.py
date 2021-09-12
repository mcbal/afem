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
            beta=1.0,
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
            beta=1.0,
        ).double()

        h = torch.randn(1, num_spins, dim).double()
        t0 = torch.rand(1, num_spins).double().requires_grad_()

        analytical_grad = model._grad_t_phi(t0, h)
        numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

        self.assertTrue(
            torch.allclose(analytical_grad, numerical_grad)
        )


class TestRootFindingGradients(unittest.TestCase):
    def test_vector_spin_model_forward_afe(self):
        num_spins, dim = 11, 17

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
        ).double()

        x = torch.randn(3, num_spins, dim).double()

        self.assertTrue(
            gradcheck(
                lambda x: model(x)[0],
                x.requires_grad_(),
                eps=1e-5,
                atol=1e-4,
                check_undefined_grad=False,
            )
        )

    def test_vector_spin_model_forward_afe_asym(self):
        num_spins, dim = 11, 17

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
            J_symmetric=False,
        ).double()

        x = torch.randn(3, num_spins, dim).double()

        self.assertTrue(
            gradcheck(
                lambda x: model(x)[0],
                x.requires_grad_(),
                eps=1e-5,
                atol=1e-4,
                check_undefined_grad=False,
            )
        )

    def test_vector_spin_model_forward_responses(self):
        num_spins, dim = 11, 17

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
        ).double()

        x = torch.randn(1, num_spins, dim).double()

        self.assertTrue(
            gradcheck(
                lambda x: model(x, return_magnetizations=True)[2],
                x.requires_grad_(),
                eps=1e-5,
                atol=1e-4,
                check_undefined_grad=False,
            )
        )

    def test_vector_spin_model_forward_responses_asym(self):
        num_spins, dim = 11, 17

        model = VectorSpinModel(
            num_spins=num_spins,
            dim=dim,
            beta=1.0,
            J_symmetric=False,
        ).double()

        x = torch.randn(1, num_spins, dim).double()

        self.assertTrue(
            gradcheck(
                lambda x: model(x, return_magnetizations=True)[2],
                x.requires_grad_(),
                eps=1e-5,
                atol=1e-4,
                check_undefined_grad=False,
            )
        )


if __name__ == '__main__':
    unittest.main()
