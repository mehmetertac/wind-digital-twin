"""Tests for gearbox thermal normal-behavior model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.config import POWER_COLUMN, THERMAL_DRIVER_COLUMNS
from tests.thermal_fixtures import synthetic_thermal_df as _synthetic_thermal_df
from wind_digital_twin.physics.gearbox_thermal import (
    fit_gearbox_thermal,
    fit_gearbox_thermal_with_selection,
    healthy_train_validate_split,
    validate_thermal_model,
)


def test_fit_and_residual_near_zero_on_train():
    df = _synthetic_thermal_df()
    model = fit_gearbox_thermal(df.iloc[:400], "Gear_Oil_Temp_Avg")
    residuals = model.residual(df.iloc[:400])
    assert len(residuals) == 400
    assert abs(residuals.mean()) < 1.0
    assert residuals.std() < 2.0


def test_injected_offset_shifts_residual_mean():
    df = _synthetic_thermal_df()
    train_df, val_df = healthy_train_validate_split(df)
    model = fit_gearbox_thermal(train_df, "Gear_Oil_Temp_Avg")

    val_shifted = val_df.copy()
    val_shifted["Gear_Oil_Temp_Avg"] = val_shifted["Gear_Oil_Temp_Avg"] + 5.0
    residuals = model.residual(val_shifted)
    assert residuals.mean() < -4.0


def test_time_ordered_split_no_future_in_train():
    df = _synthetic_thermal_df(n=100)
    train_df, val_df = healthy_train_validate_split(df, train_fraction=0.8)
    assert train_df.index.max() <= val_df.index.min()
    assert len(train_df) == 80
    assert len(val_df) == 20


def test_validate_flags_structured_residuals():
    df = _synthetic_thermal_df()
    train_df, val_df = healthy_train_validate_split(df)
    model = fit_gearbox_thermal(train_df, "Gear_Oil_Temp_Avg")

    good = validate_thermal_model(model, val_df)
    assert good["n_samples"] == len(val_df)
    assert good["rmse"] < 2.0

    # Inject systematic residual slope vs power
    val_bad = val_df.copy()
    val_bad["Gear_Oil_Temp_Avg"] = (
        val_bad["Gear_Oil_Temp_Avg"] - 0.01 * val_bad[POWER_COLUMN]
    )
    bad = validate_thermal_model(model, val_bad)
    assert bad["driver_correlations"][POWER_COLUMN] != 0.0


def test_model_selection_returns_info():
    df = _synthetic_thermal_df(n=600)
    model, train_df, val_df, info = fit_gearbox_thermal_with_selection(
        df, "Gear_Bear_Temp_Avg", driver_columns=THERMAL_DRIVER_COLUMNS
    )
    assert info["chosen_model"] == "lumped_ode"
    assert "linear_val_rmse" in info
    assert "ode_val_rmse" in info
    assert info["train_rows"] > 0
    assert info["val_rows"] > 0
    preds = model.predict(df)
    assert len(preds) == len(df)


def test_residual_frame_columns():
    df = _synthetic_thermal_df(n=50)
    model = fit_gearbox_thermal(df.iloc[:40], "Gear_Oil_Temp_Avg")
    frame = model.residual_frame(df.iloc[40:])
    assert list(frame.columns) == ["actual", "predicted", "residual"]
    assert (frame["residual"] == frame["predicted"] - frame["actual"]).all()
