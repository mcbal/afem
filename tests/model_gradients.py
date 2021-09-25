import itertools
import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from afem.models import VectorSpinModel


class TestAnalyticalGradients(unittest.TestCase):
    def test_phi_t(self):
        num_spins, dim = 11, 17

        for (t_vector, J_add_external, J_symmetric) in itertools.product([True, False], repeat=3):
            with self.subTest(t_vector=t_vector, J_add_external=J_add_external, J_symmetric=J_symmetric):
                model = VectorSpinModel(
                    num_spins=num_spins,
                    dim=dim,
                    beta=1.0,
                    J_add_external=J_add_external,
                    J_symmetric=J_symmetric,
                ).double()

                h = torch.randn(1, num_spins, dim).double()
                t0 = torch.ones(1, num_spins) if t_vector else torch.ones(1, 1)  # (batch explicit)
                t0 = t0.double().requires_grad_()

                analytical_grad = model._jac_phi(t0, h)
                numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

                self.assertTrue(
                    torch.allclose(analytical_grad, numerical_grad)
                )

    def test_grad_phi_t(self):
        num_spins, dim = 11, 17

        for (t_vector, J_add_external, J_symmetric) in itertools.product([True, False], repeat=3):
            with self.subTest(t_vector=t_vector, J_add_external=J_add_external, J_symmetric=J_symmetric):
                model = VectorSpinModel(
                    num_spins=num_spins,
                    dim=dim,
                    beta=1.0,
                    J_add_external=J_add_external,
                    J_symmetric=J_symmetric,
                ).double()

                h = torch.randn(1, num_spins, dim).double()
                t0 = torch.ones(1, num_spins) if t_vector else torch.ones(1, 1)  # (batch explicit)
                t0 = t0.double().requires_grad_()

                analytical_grad = model._hess_phi(t0, h).sum(dim=-1)
                numerical_grad = torch.autograd.grad(model._jac_phi(t0, h).sum(dim=-1), t0)[0]

                self.assertTrue(
                    torch.allclose(analytical_grad, numerical_grad)
                )


class TestRootFindingGradients(unittest.TestCase):
    # def test_vector_spin_model_afe(self):
    #     num_spins, dim = 11, 17

    #     for (t_vector, use_analytical_grads, J_add_external, J_symmetric) in itertools.product([True, False], repeat=4):
    #         with self.subTest(
    #             t_vector=t_vector,
    #             use_analytical_grads=use_analytical_grads,
    #             J_add_external=J_add_external,
    #             J_symmetric=J_symmetric
    #         ):
    #             model = VectorSpinModel(
    #                 num_spins=num_spins,
    #                 dim=dim,
    #                 beta=1.0,
    #                 J_add_external=J_add_external,
    #                 J_symmetric=J_symmetric,
    #             ).double()

    #             x = torch.randn(1, num_spins, dim).double()
    #             t0 = torch.ones(num_spins) if t_vector else torch.ones(1)
    #             t0 = t0.double().requires_grad_()

    #             self.assertTrue(
    #                 gradcheck(
    #                     lambda z: model(z, t0, use_analytical_grads=use_analytical_grads)[0],
    #                     x.requires_grad_(),
    #                     eps=1e-5,
    #                     atol=1e-4,
    #                     check_undefined_grad=False,
    #                 )
    #             )

    def test_vector_spin_model_magnetizations(self):
        num_spins, dim = 11, 17

        for (t_vector, use_analytical_grads, J_add_external, J_symmetric) in itertools.product([True, False], repeat=4):
            with self.subTest(
                t_vector=t_vector,
                use_analytical_grads=use_analytical_grads,
                J_add_external=J_add_external,
                J_symmetric=J_symmetric
            ):
                model = VectorSpinModel(
                    num_spins=num_spins,
                    dim=dim,
                    beta=1.0,
                    J_add_external=J_add_external,
                    J_symmetric=J_symmetric,
                ).double()

                x = torch.randn(1, num_spins, dim).double()
                t0 = torch.ones(num_spins) if t_vector else torch.ones(1)
                t0 = t0.double().requires_grad_()

                self.assertTrue(
                    gradcheck(
                        lambda z: model(z, t0, return_magnetizations=True, use_analytical_grads=use_analytical_grads)[2],
                        x.requires_grad_(),
                        eps=1e-5,
                        atol=1e-3,
                        check_undefined_grad=False,
                    )
                )


if __name__ == '__main__':
    unittest.main()
