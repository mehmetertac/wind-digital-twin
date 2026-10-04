"""Tests for learned hybrid thermal corrector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.thermal_fixtures import synthetic_thermal_df
from wind_digital_twin.config import THERMAL_TARGET_COLUMNS
from wind_digital_twin.physics.gearbox_thermal import fit_gearbox_thermal
from wind_digital_twin.residual.learned_residual import fit_hybrid_thermal_corrector
from wind_digital_twin.residual.residual_features import degradation_signal


def _biased_thermal_df(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """Healthy SCADA with P^2 bias the linear-in-P ODE underfits."""
    df = synthetic_thermal_df(n=n, seed=seed)
    power = df["Grd_Prod_Pwr_Avg"]
    bias = 1.5e-6 * (power - power.mean()) ** 2
    for col in THERMAL_TARGET_COLUMNS:
        df[col] = df[col] + bias
    return df


def test_hybrid_rmse_beats_physics_on_healthy_val():
    df = _biased_thermal_df(n=700)
    healthy = df.iloc[:600]
    _, metrics = fit_hybrid_thermal_corrector(healthy, train_fraction=0.75)
    for slug in ("oil", "bear"):
        t = metrics["targets"][slug]
        assert t["hybrid_val_rmse"] < t["physics_val_rmse"]
        assert t["rmse_reduction_pct"] > 0.0


def test_fault_offset_stays_visible_after_hybrid():
    df = _biased_thermal_df(n=650)
    train = df.iloc[:500]
    test = df.iloc[500:].copy()
    hybrid, _ = fit_hybrid_thermal_corrector(train, train_fraction=0.8)

    healthy_res = hybrid.dual_hybrid_residuals(test)
    shifted = test.copy()
    shifted[THERMAL_TARGET_COLUMNS[0]] += 5.0
    shifted[THERMAL_TARGET_COLUMNS[1]] += 5.0
    fault_res = hybrid.dual_hybrid_residuals(shifted)

    healthy_deg = degradation_signal(healthy_res["oil_residual"]).mean()
    fault_deg = degradation_signal(fault_res["oil_residual"]).mean()
    assert fault_deg > healthy_deg + 3.0


def test_corrector_does_not_use_gear_temps_in_features():
    df = _biased_thermal_df(n=400)
    hybrid, _ = fit_hybrid_thermal_corrector(df.iloc[:350])
    for corr in (hybrid.oil_corrector, hybrid.bear_corrector):
        for col in THERMAL_TARGET_COLUMNS:
            assert col not in corr.feature_columns
        assert "theta_oil" not in corr.feature_columns
