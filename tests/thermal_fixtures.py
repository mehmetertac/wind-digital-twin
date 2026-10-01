"""Shared SCADA-like frames for thermal / hybrid tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.config import POWER_COLUMN, THERMAL_TARGET_COLUMNS


def synthetic_thermal_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Time-correlated drivers and oil/bear temps from a lumped ODE (10-min steps).

    Independent per-row noise breaks open-loop ODE fit; this generator is
    consistent with the physics backbone.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    t_hours = np.arange(n) / 6.0
    power = 400.0 + 900.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t_hours / 36.0))
    power += rng.normal(0, 40.0, n)
    power = np.clip(power, 150.0, 1800.0)
    rpm = 8.0 + 0.004 * power + rng.normal(0, 0.15, n)
    nac = 14.0 + 3.0 * np.sin(2 * np.pi * t_hours / 72.0) + rng.normal(0, 0.25, n)

    tau_sec = 45.0 * 60.0
    beta_0_oil, beta_p, beta_rpm = 30.0, 0.004, 0.02
    t_eq_oil = nac + beta_0_oil + beta_p * power + beta_rpm * rpm
    state = float(t_eq_oil[0])
    oil = np.empty(n)
    for k in range(n):
        oil[k] = state + rng.normal(0, 0.2)
        if k + 1 < n:
            state = t_eq_oil[k] + (state - t_eq_oil[k]) * np.exp(-600.0 / tau_sec)

    bear = t_eq_oil + 5.0 + 0.002 * power
    for k in range(n):
        bear[k] = bear[k] + rng.normal(0, 0.2)

    return pd.DataFrame(
        {
            THERMAL_TARGET_COLUMNS[0]: oil,
            THERMAL_TARGET_COLUMNS[1]: bear,
            POWER_COLUMN: power,
            "Rtr_RPM_Avg": rpm,
            "Nac_Temp_Avg": nac,
        },
        index=idx,
    )
