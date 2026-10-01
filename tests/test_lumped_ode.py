"""Tests for lumped thermal ODE fit and simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_digital_twin.config import POWER_COLUMN
from wind_digital_twin.physics.lumped_ode import (
    LumpedThermalParams,
    fit_lumped_thermal,
    simulate_temperature,
    tau_plausible,
)
from wind_digital_twin.physics.gearbox_thermal import (
    fit_gearbox_thermal,
    validate_thermal_model,
    healthy_train_validate_split,
)


def _generate_closed_loop_ode(
    n: int = 400,
    tau_minutes: float = 45.0,
    seed: int = 0,
) -> pd.DataFrame:
    """SCADA-like drivers with temperatures from a known lumped ODE."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    t_hours = np.arange(n) / 6.0
    power = 400.0 + 800.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t_hours / 48.0))
    power += rng.normal(0, 30.0, n)
    power = np.clip(power, 100.0, 1800.0)
    rpm = 8.0 + 0.004 * power + rng.normal(0, 0.15, n)
    nac = 14.0 + 3.0 * np.sin(2 * np.pi * t_hours / 72.0) + rng.normal(0, 0.3, n)

    params = LumpedThermalParams(
        tau_seconds=tau_minutes * 60.0,
        beta_0=30.0,
        beta_p=0.004,
        beta_rpm=0.05,
    )
    t_eq = nac + params.beta_0 + params.beta_p * power + params.beta_rpm * rpm
    tau = params.tau_seconds
    state = float(t_eq[0])
    oil = np.empty(n)
    for k in range(n):
        oil[k] = state + rng.normal(0, 0.15)
        if k + 1 < n:
            state = t_eq[k] + (state - t_eq[k]) * np.exp(-600.0 / tau)

    return pd.DataFrame(
        {
            "Gear_Oil_Temp_Avg": oil,
            "Gear_Bear_Temp_Avg": oil + 5.0,
            POWER_COLUMN: power,
            "Rtr_RPM_Avg": rpm,
            "Nac_Temp_Avg": nac,
        },
        index=idx,
    )


def test_fit_recovers_tau_minutes():
    df = _generate_closed_loop_ode(tau_minutes=45.0, n=500)
    fitted = fit_lumped_thermal(df.iloc[:400], "Gear_Oil_Temp_Avg")
    assert 25.0 <= fitted.tau_minutes <= 70.0


def test_tau_plausible_bands():
    assert tau_plausible("Gear_Oil_Temp_Avg", 60.0)
    assert not tau_plausible("Gear_Oil_Temp_Avg", 5.0)
    assert tau_plausible("Gear_Bear_Temp_Avg", 20.0)


def test_simulate_resets_on_gap():
    idx = pd.to_datetime(
        ["2016-01-01 00:00", "2016-01-01 00:10", "2016-01-02 00:10"], utc=True
    )
    df = pd.DataFrame(
        {
            "Gear_Oil_Temp_Avg": [50.0, 51.0, 99.0],
            POWER_COLUMN: [500.0, 500.0, 500.0],
            "Rtr_RPM_Avg": [10.0, 10.0, 10.0],
            "Nac_Temp_Avg": [15.0, 15.0, 15.0],
        },
        index=idx,
    )
    params = LumpedThermalParams(tau_seconds=3600.0, beta_0=30.0, beta_p=0.0, beta_rpm=0.0)
    pred = simulate_temperature(df, params, "Gear_Oil_Temp_Avg")
    assert pred[2] == 99.0


def test_nn_correction_bounded_and_frozen_physics():
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("torch not installed or broken")

    df = _generate_closed_loop_ode(n=300)
    train = df.iloc[:200]
    params = fit_lumped_thermal(train, "Gear_Oil_Temp_Avg")
    corrected = fit_gearbox_thermal(train, "Gear_Oil_Temp_Avg", use_nn_correction=True)
    assert isinstance(corrected.model, LumpedThermalParams)
    assert corrected.model.tau_seconds == pytest.approx(params.tau_seconds)
    assert corrected.equilibrium_correction is not None
    corr = corrected.equilibrium_correction(
        train[POWER_COLUMN].to_numpy(),
        train["Rtr_RPM_Avg"].to_numpy(),
        train["Nac_Temp_Avg"].to_numpy(),
    )
    assert np.all(np.abs(corr) <= 3.0 + 1e-6)


def test_ode_validate_on_held_out():
    df = _generate_closed_loop_ode(n=600)
    train_df, val_df = healthy_train_validate_split(df)
    model = fit_gearbox_thermal(train_df, "Gear_Oil_Temp_Avg")
    metrics = validate_thermal_model(model, val_df)
    assert metrics["rmse"] < 2.0
    assert "tau_minutes" in metrics
