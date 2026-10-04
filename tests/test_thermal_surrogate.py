"""Tests for ODE thermal surrogate (torch optional)."""

from __future__ import annotations

import pytest

from wind_digital_twin.surrogate.thermal_surrogate import (
    build_surrogate_dataset,
    simulate_trajectory,
)

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch unavailable", allow_module_level=True)

from wind_digital_twin.surrogate.thermal_surrogate import train_surrogate  # noqa: E402


def test_build_surrogate_dataset_shapes():
    ds = build_surrogate_dataset(horizon=24, n_profiles=3, n_params=4, seed=2)
    assert ds.X.shape[0] == 12
    assert ds.y.shape == (12, 24)
    assert ds.X.shape[1] == ds.param_dim + ds.load_dim


def test_train_surrogate_holdout_rmse_reasonable():
    ds = build_surrogate_dataset(horizon=32, n_profiles=4, n_params=6, seed=3)
    _model, _mean, _std, rmse = train_surrogate(ds, epochs=25, val_fraction=0.2)
    assert rmse < 15.0


def test_simulate_trajectory_length():
    from wind_digital_twin.surrogate.thermal_surrogate import random_load_profiles
    from wind_digital_twin.surrogate.thermal_surrogate import sample_thermal_params

    profile = random_load_profiles(horizon=20, n_profiles=1)[0]
    params = sample_thermal_params(1)[0]
    traj = simulate_trajectory(profile, params)
    assert len(traj) == 20
