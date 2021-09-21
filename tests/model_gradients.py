import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from afem.models import VectorSpinModel


class TestAnalyticalGradients(unittest.TestCase):
    def test_phi_t(self):
        num_spins, dim = 11, 17

        for J_add_external in [True, False]:
            for J_symmetric in [True, False]:
                with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
                    model = VectorSpinModel(
                        num_spins=num_spins,
                        dim=dim,
                        beta=1.0,
                        J_add_external=J_add_external,
                        J_symmetric=J_symmetric,
                    ).double()

                    h = torch.randn(1, num_spins, dim).double()
                    t0 = torch.rand(1, 1).double().requires_grad_()  # scalar t (batch explicit)

                    analytical_grad = model._jac_phi(t0, h)
                    numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

                    self.assertTrue(
                        torch.allclose(analytical_grad, numerical_grad)
                    )

    def test_phi_t_vector(self):
        num_spins, dim = 11, 17

        for J_add_external in [True, False]:
            for J_symmetric in [True, False]:
                with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
                    model = VectorSpinModel(
                        num_spins=num_spins,
                        dim=dim,
                        beta=1.0,
                        J_add_external=J_add_external,
                        J_symmetric=J_symmetric,
                    ).double()

                    h = torch.randn(1, num_spins, dim).double()
                    t0 = torch.rand(1, num_spins).double().requires_grad_()  # vector t (batch explicit)

                    analytical_grad = model._jac_phi(t0, h)
                    numerical_grad = torch.autograd.grad(model._phi(t0, h), t0)[0]

                    self.assertTrue(
                        torch.allclose(analytical_grad, numerical_grad)
                    )

    def test_grad_phi_t_scalar(self):
        num_spins, dim = 11, 17

        for J_add_external in [True, False]:
            for J_symmetric in [True, False]:
                with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
                    model = VectorSpinModel(
                        num_spins=num_spins,
                        dim=dim,
                        beta=1.0,
                        J_add_external=J_add_external,
                        J_symmetric=J_symmetric,
                    ).double()

                    h = torch.randn(1, num_spins, dim).double()
                    t0 = torch.rand(1, 1).double().requires_grad_()  # scalar t (batch explicit)

                    analytical_grad = model._hess_phi(t0, h).sum(dim=-1)
                    numerical_grad = torch.autograd.grad(model._jac_phi(t0, h).sum(dim=-1), t0)[0]

                    self.assertTrue(
                        torch.allclose(analytical_grad, numerical_grad)
                    )

    def test_grad_phi_t_vector(self):
        num_spins, dim = 11, 17

        for J_add_external in [True, False]:
            for J_symmetric in [True, False]:
                with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
                    model = VectorSpinModel(
                        num_spins=num_spins,
                        dim=dim,
                        beta=1.0,
                        J_add_external=J_add_external,
                        J_symmetric=J_symmetric,
                    ).double()

                    h = torch.randn(1, num_spins, dim).double()
                    t0 = torch.rand(1, num_spins).double().requires_grad_()  # vector t (batch explicit)

                    analytical_grad = model._hess_phi(t0, h).sum(dim=-1)
                    numerical_grad = torch.autograd.grad(model._jac_phi(t0, h).sum(dim=-1), t0)[0]

                    self.assertTrue(
                        torch.allclose(analytical_grad, numerical_grad)
                    )


class TestRootFindingGradients(unittest.TestCase):
    def test_vector_spin_model_forward_afe(self):
        num_spins, dim = 11, 17

        for J_add_external in [False]:
            for J_symmetric in [True, False]:
                with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
                    model = VectorSpinModel(
                        num_spins=num_spins,
                        dim=dim,
                        beta=1.0,
                        J_add_external=J_add_external,
                        J_symmetric=J_symmetric,
                    ).double()

                    x = torch.randn(3, num_spins, dim).double()
                    t0 = torch.ones(num_spins).double().requires_grad_()

                    self.assertTrue(
                        gradcheck(
                            lambda x: model(x, t0)[0],
                            x.requires_grad_(),
                            eps=1e-5,
                            atol=1e-4,
                            check_undefined_grad=False,
                        )
                    )

    # def test_vector_spin_model_forward_afe(self):
    #     num_spins, dim = 11, 17

    #     for J_add_external in [False]:
    #         for J_symmetric in [True]:
    #             with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
    #                 model = VectorSpinModel(
    #                     num_spins=num_spins,
    #                     dim=dim,
    #                     beta=1.0,
    #                     J_add_external=J_add_external,
    #                     J_symmetric=J_symmetric,
    #                 ).double()

    #                 x = torch.randn(3, num_spins, dim).double()
    #                 t0 = torch.ones(num_spins).double().requires_grad_()

    #                 self.assertTrue(
    #                     gradcheck(
    #                         lambda x: model(x, t0, solver_fwd_analytic_jac=True)[0],
    #                         x.requires_grad_(),
    #                         eps=1e-5,
    #                         atol=1e-4,
    #                         check_undefined_grad=False,
    #                     )
    #                 )

    #     def test_vector_spin_model_forward_responses(self):
    #         num_spins, dim = 11, 17

    #         for J_add_external in [True, False]:
    #             for J_symmetric in [True, False]:
    #                 with self.subTest(J_add_external=J_add_external, J_symmetric=J_symmetric):
    #                     model = VectorSpinModel(
    #                         num_spins=num_spins,
    #                         dim=dim,
    #                         beta=1.0,
    #                         J_add_external=J_add_external,
    #                         J_symmetric=J_symmetric,
    #                     ).double()

    #                     x = torch.randn(1, num_spins, dim).double()
    #                       t0 = torch.ones(num_spins).double().requires_grad_()

    #                     self.assertTrue(
    #                         gradcheck(
    #                             lambda x: model(x, return_magnetizations=True)[2],
    #                             x.requires_grad_(),
    #                             eps=1e-5,
    #                             atol=1e-4,
    #                             check_undefined_grad=False,
    #                         )
    #                     )


if __name__ == '__main__':
    unittest.main()
