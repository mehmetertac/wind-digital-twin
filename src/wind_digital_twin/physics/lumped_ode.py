"""First-order lumped thermal ODE for gearbox oil/bearing temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from wind_digital_twin.config import (
    POWER_COLUMN,
    THERMAL_DRIVER_COLUMNS,
    THERMAL_GAP_RESET_HOURS,
    THERMAL_ODE_TAU_INIT_MINUTES,
    THERMAL_ODE_TAU_MAX_MINUTES,
    THERMAL_ODE_TAU_MIN_MINUTES,
    THERMAL_TARGET_COLUMNS,
)

EquilibriumCorrectionFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LumpedThermalParams:
    """Identifiable ODE parameters: tau and equilibrium gains."""

    tau_seconds: float
    beta_0: float
    beta_p: float
    beta_rpm: float

    @property
    def tau_minutes(self) -> float:
        return self.tau_seconds / 60.0


def equilibrium_temperature(
    nac: np.ndarray,
    power: np.ndarray,
    rpm: np.ndarray,
    params: LumpedThermalParams,
    correction: EquilibriumCorrectionFn | None = None,
) -> np.ndarray:
    """T_eq = T_nac + beta_0 + beta_P * P + beta_rpm * RPM (+ optional correction)."""
    t_eq = nac + params.beta_0 + params.beta_p * power + params.beta_rpm * rpm
    if correction is not None:
        t_eq = t_eq + correction(power, rpm, nac)
    return t_eq


def _gap_reset_mask(index: pd.DatetimeIndex, gap_hours: float) -> np.ndarray:
    """True at row k when dt from k-1 exceeds gap_hours (includes k=0)."""
    n = len(index)
    if n == 0:
        return np.array([], dtype=bool)
    if n == 1:
        return np.array([True])
    dt_sec = index.to_series().diff().dt.total_seconds().to_numpy()
    reset = np.zeros(n, dtype=bool)
    reset[0] = True
    reset[1:] = dt_sec[1:] > gap_hours * 3600.0
    return reset


def simulate_temperature(
    df: pd.DataFrame,
    params: LumpedThermalParams,
    target_column: str,
    driver_columns: list[str] | None = None,
    gap_reset_hours: float = THERMAL_GAP_RESET_HOURS,
    correction: EquilibriumCorrectionFn | None = None,
) -> np.ndarray:
    """
    Open-loop integrate: T_0 = first observed target; reset state on long gaps.

    Drivers are piecewise constant between SCADA stamps.
    """
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    power = df[POWER_COLUMN].to_numpy(dtype=float)
    rpm = df["Rtr_RPM_Avg"].to_numpy(dtype=float)
    nac = df["Nac_Temp_Avg"].to_numpy(dtype=float)
    observed = df[target_column].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return np.array([])

    t_eq = equilibrium_temperature(nac, power, rpm, params, correction=correction)
    reset = _gap_reset_mask(df.index, gap_reset_hours)
    dt_sec = df.index.to_series().diff().dt.total_seconds().to_numpy()
    dt_sec[0] = 0.0

    pred = np.empty(n, dtype=float)
    tau = max(params.tau_seconds, 1.0)
    state = float(observed[0])

    for k in range(n):
        if k > 0:
            if reset[k]:
                state = float(observed[k])
            else:
                dt = dt_sec[k]
                if dt > 0:
                    decay = np.exp(-dt / tau)
                    state = t_eq[k] + (state - t_eq[k]) * decay
        pred[k] = state

    return pred


def _initial_gains(
    train_df: pd.DataFrame,
    target_column: str,
) -> tuple[float, float, float]:
    """Ridge-free OLS on (T - T_nac) ~ P + RPM."""
    y = (train_df[target_column] - train_df["Nac_Temp_Avg"]).to_numpy(dtype=float)
    power = train_df[POWER_COLUMN].to_numpy(dtype=float)
    rpm = train_df["Rtr_RPM_Avg"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), power, rpm])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def fit_lumped_thermal(
    train_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str] | None = None,
    tau_init_minutes: float = THERMAL_ODE_TAU_INIT_MINUTES,
    correction: EquilibriumCorrectionFn | None = None,
) -> LumpedThermalParams:
    """Fit tau and equilibrium gains by nonlinear least squares on training rows."""
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    del driver_columns  # drivers fixed in config for ODE

    if len(train_df) < 10:
        raise ValueError("Need at least 10 training rows for lumped ODE fit")

    beta_0, beta_p, beta_rpm = _initial_gains(train_df, target_column)
    tau_init = tau_init_minutes * 60.0
    tau_min = THERMAL_ODE_TAU_MIN_MINUTES * 60.0
    tau_max = THERMAL_ODE_TAU_MAX_MINUTES * 60.0
    lower = np.array([tau_min, -50.0, 0.0, -2.0])
    upper = np.array([tau_max, 80.0, 0.05, 2.0])
    x0 = np.array([tau_init, beta_0, beta_p, beta_rpm], dtype=float)
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)

    y_obs = train_df[target_column].to_numpy(dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        params = LumpedThermalParams(
            tau_seconds=float(x[0]),
            beta_0=float(x[1]),
            beta_p=float(x[2]),
            beta_rpm=float(x[3]),
        )
        pred = simulate_temperature(
            train_df,
            params,
            target_column,
            correction=correction,
        )
        return pred - y_obs

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
    x = result.x
    return LumpedThermalParams(
        tau_seconds=float(x[0]),
        beta_0=float(x[1]),
        beta_p=float(x[2]),
        beta_rpm=float(x[3]),
    )


def tau_plausible(
    target_column: str,
    tau_minutes: float,
) -> bool:
    """Field sanity bands for oil vs bearing time constants."""
    from wind_digital_twin.config import (
        THERMAL_BEAR_TAU_PLAUSIBLE_MAX_MIN,
        THERMAL_BEAR_TAU_PLAUSIBLE_MIN_MIN,
        THERMAL_OIL_TAU_PLAUSIBLE_MAX_MIN,
        THERMAL_OIL_TAU_PLAUSIBLE_MIN_MIN,
    )

    if target_column == THERMAL_TARGET_COLUMNS[0]:
        lo, hi = THERMAL_OIL_TAU_PLAUSIBLE_MIN_MIN, THERMAL_OIL_TAU_PLAUSIBLE_MAX_MIN
    else:
        lo, hi = THERMAL_BEAR_TAU_PLAUSIBLE_MIN_MIN, THERMAL_BEAR_TAU_PLAUSIBLE_MAX_MIN
    return lo <= tau_minutes <= hi


__all__ = [
    "EquilibriumCorrectionFn",
    "LumpedThermalParams",
    "equilibrium_temperature",
    "fit_lumped_thermal",
    "simulate_temperature",
    "tau_plausible",
]
