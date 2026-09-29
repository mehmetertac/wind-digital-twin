"""Tests for 1-D heat PINN (skipped when torch is not installed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch unavailable", allow_module_level=True)

from wind_digital_twin.pinn.heat1d import (  # noqa: E402
    ALPHA,
    MLP,
    analytical_solution,
    make_model,
    pde_residual,
    relative_l2,
    sample_training_data,
    train,
)


def test_analytical_satisfies_ic_and_bc() -> None:
    x = np.linspace(0.0, 1.0, 50)
    t = np.linspace(0.0, 1.0, 50)
    u_ic = analytical_solution(x, np.zeros_like(x))
    np.testing.assert_allclose(u_ic, np.sin(math.pi * x), rtol=1e-10)
    u_left = analytical_solution(0.0, t)
    u_right = analytical_solution(1.0, t)
    np.testing.assert_allclose(u_left, 0.0, atol=1e-12)
    np.testing.assert_allclose(u_right, 0.0, atol=1e-12)


class AnalyticalNet(torch.nn.Module):
    """Wrap analytical u(x,t) for derivative checks."""

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return analytical_solution(x, t).reshape(-1, 1)


def test_pde_residual_near_zero_for_analytical() -> None:
    model = AnalyticalNet()
    xt = torch.rand(40, 2, requires_grad=True)
    f = pde_residual(model, xt, alpha=ALPHA)
    assert float(torch.mean(f**2).detach()) < 1e-4


def test_training_reduces_loss_quick() -> None:
    data = sample_training_data(n_ic=20, n_bc=20, n_data=5, n_f=200, seed=0)
    model = make_model(seed=1)
    hist = train(
        model,
        data,
        w_f=1.0,
        adam_steps=150,
        lbfgs_max_iter=0,
        log_every=50,
    )
    assert hist["total"][-1] < hist["total"][0]


def test_w_f_zero_skips_physics_term_in_total() -> None:
    data = sample_training_data(n_ic=10, n_bc=10, n_data=0, n_f=100, seed=2)
    model = make_model(seed=3)
    train(model, data, w_f=0.0, adam_steps=30, lbfgs_max_iter=0, log_every=30)
    # Interior should not match truth without physics or interior labels.
    l2 = relative_l2(model, n=51)
    assert l2 > 0.05


def test_pinn_beats_data_only_on_quick_run() -> None:
    data = sample_training_data(n_ic=25, n_bc=25, n_data=10, n_f=500, seed=42)
    m_pinn = make_model(seed=42)
    train(m_pinn, data, w_f=1.0, adam_steps=800, lbfgs_max_iter=80)
    m_data = make_model(seed=42)
    train(m_data, data, w_f=0.0, adam_steps=800, lbfgs_max_iter=80)
    l2_pinn = relative_l2(m_pinn)
    l2_data = relative_l2(m_data)
    assert l2_pinn < 0.05
    assert l2_data > l2_pinn * 2
