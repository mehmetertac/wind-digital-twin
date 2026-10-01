"""Physics-informed normal-behavior model for gearbox oil/bearing temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wind_digital_twin.config import (
    POWER_COLUMN,
    THERMAL_DRIVER_COLUMNS,
    THERMAL_SEASONAL_TERMS,
    THERMAL_TRAIN_FRACTION,
)
from wind_digital_twin.physics.lumped_ode import (
    EquilibriumCorrectionFn,
    LumpedThermalParams,
    fit_lumped_thermal,
    simulate_temperature,
    tau_plausible,
)

ModelKind = Literal["linear", "lumped_ode"]

THERMAL_VALIDATION_THRESHOLDS = {
    "max_abs_mean_residual_c": 0.5,
    "max_driver_correlation": 0.1,
    "max_time_trend_per_day_c": 0.01,
}


@dataclass
class GearboxThermalModel:
    """Per-turbine thermal model: predict expected gearbox temperature."""

    model: RegressorMixin | LumpedThermalParams
    model_kind: ModelKind
    driver_columns: list[str]
    target_column: str
    feature_names: list[str] | None = None
    seasonal_terms: bool = False
    equilibrium_correction: EquilibriumCorrectionFn | None = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predicted temperature (°C)."""
        if self.model_kind == "lumped_ode":
            assert isinstance(self.model, LumpedThermalParams)
            pred = simulate_temperature(
                X,
                self.model,
                self.target_column,
                self.driver_columns,
                correction=self.equilibrium_correction,
            )
            return pd.Series(pred, index=X.index, name=f"{self.target_column}_pred")

        X_arr = self._build_features(X)
        pred = self.model.predict(X_arr)  # type: ignore[union-attr]
        return pd.Series(pred, index=X.index, name=f"{self.target_column}_pred")

    def residual(self, X: pd.DataFrame) -> pd.Series:
        """Return predicted minus actual temperature (°C)."""
        actual = X[self.target_column]
        predicted = self.predict(X)
        return pd.Series(
            predicted.values - actual.values,
            index=X.index,
            name=f"{self.target_column}_residual",
        )

    def residual_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return actual, predicted, and residual columns."""
        actual = X[self.target_column]
        predicted = self.predict(X)
        residual = predicted - actual
        return pd.DataFrame(
            {
                "actual": actual,
                "predicted": predicted,
                "residual": residual,
            },
            index=X.index,
        )

    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        drivers = X[self.driver_columns]
        if self.model_kind == "linear":
            power = drivers[POWER_COLUMN].values.reshape(-1, 1)
            rpm = drivers["Rtr_RPM_Avg"].values.reshape(-1, 1)
            nac = drivers["Nac_Temp_Avg"].values.reshape(-1, 1)
            parts = [power, rpm, nac, power**2]
            if self.seasonal_terms:
                month = np.asarray(X.index.month, dtype=float)
                month_rad = 2.0 * np.pi * month / 12.0
                parts.extend(
                    [np.sin(month_rad).reshape(-1, 1), np.cos(month_rad).reshape(-1, 1)]
                )
            return np.hstack(parts)
        return drivers.values


def _linear_feature_names(seasonal_terms: bool) -> list[str]:
    names = [POWER_COLUMN, "Rtr_RPM_Avg", "Nac_Temp_Avg", f"{POWER_COLUMN}_sq"]
    if seasonal_terms:
        names.extend(["month_sin", "month_cos"])
    return names


def healthy_train_validate_split(
    healthy_df: pd.DataFrame,
    train_fraction: float = THERMAL_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split of healthy rows: first fraction train, remainder validate."""
    if healthy_df.empty:
        return healthy_df, healthy_df.iloc[0:0]
    n_train = max(1, int(len(healthy_df) * train_fraction))
    if n_train >= len(healthy_df):
        n_train = max(1, len(healthy_df) - 1)
    train_df = healthy_df.iloc[:n_train]
    val_df = healthy_df.iloc[n_train:]
    return train_df, val_df


def _fit_linear(
    train_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str],
    seasonal_terms: bool = THERMAL_SEASONAL_TERMS,
) -> GearboxThermalModel:
    model = GearboxThermalModel(
        model=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        ),
        model_kind="linear",
        driver_columns=driver_columns,
        target_column=target_column,
        feature_names=_linear_feature_names(seasonal_terms),
        seasonal_terms=seasonal_terms,
    )
    X_train = model._build_features(train_df)
    y_train = train_df[target_column].values
    model.model.fit(X_train, y_train)  # type: ignore[union-attr]
    return model


def _validation_rmse(model: GearboxThermalModel, val_df: pd.DataFrame) -> float:
    if val_df.empty:
        return float("inf")
    y_true = val_df[model.target_column].values
    y_pred = model.predict(val_df).values
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_gearbox_thermal(
    train_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str] | None = None,
    seasonal_terms: bool = THERMAL_SEASONAL_TERMS,
    use_nn_correction: bool = False,
) -> GearboxThermalModel:
    """
    Fit lumped-parameter thermal ODE on healthy training rows.

    Expects train_df to be the time-ordered training portion of healthy data.
    """
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    del seasonal_terms  # ODE uses configured driver set only
    params = fit_lumped_thermal(train_df, target_column, driver_columns)
    correction = None
    if use_nn_correction:
        from wind_digital_twin.physics.lumped_ode_correction import (
            fit_equilibrium_correction,
        )

        mlp = fit_equilibrium_correction(train_df, params, target_column)
        correction = mlp.correction_fn()
    return GearboxThermalModel(
        model=params,
        model_kind="lumped_ode",
        driver_columns=driver_columns,
        target_column=target_column,
        feature_names=["tau", "beta_0", "beta_p", "beta_rpm"],
        equilibrium_correction=correction,
    )


def fit_gearbox_thermal_with_selection(
    healthy_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str] | None = None,
    train_fraction: float = THERMAL_TRAIN_FRACTION,
    seasonal_terms: bool = THERMAL_SEASONAL_TERMS,
    use_nn_correction: bool = False,
) -> tuple[GearboxThermalModel, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Fit ODE on healthy data with time-ordered train/val split.

    Returns (final_model, train_df, val_df, selection_info).
    The final model is refit on all healthy rows. Linear RMSE is kept as baseline.
    """
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    train_df, val_df = healthy_train_validate_split(healthy_df, train_fraction)

    linear_model = _fit_linear(
        train_df, target_column, driver_columns, seasonal_terms=seasonal_terms
    )
    linear_rmse = _validation_rmse(linear_model, val_df)

    ode_model = fit_gearbox_thermal(
        train_df,
        target_column,
        driver_columns,
        use_nn_correction=use_nn_correction,
    )
    ode_rmse = _validation_rmse(ode_model, val_df)

    final_model = fit_gearbox_thermal(
        healthy_df,
        target_column,
        driver_columns,
        use_nn_correction=use_nn_correction,
    )

    selection_info = {
        "chosen_model": "lumped_ode",
        "linear_val_rmse": linear_rmse,
        "ode_val_rmse": ode_rmse,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "seasonal_terms": seasonal_terms,
        "nn_correction": use_nn_correction,
    }
    return final_model, train_df, val_df, selection_info


def validate_thermal_model(
    model: GearboxThermalModel,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check held-out healthy residuals: small, unbiased, structureless.

    Returns metrics dict with pass/fail flags per criterion.
    """
    if val_df.empty:
        return {
            "n_samples": 0,
            "passed": False,
            "reason": "empty validation set",
        }

    residuals = model.residual(val_df)
    actual = val_df[model.target_column]
    predicted = model.predict(val_df)

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    mean_res = float(residuals.mean())

    driver_corrs: dict[str, float] = {}
    for col in model.driver_columns:
        if val_df[col].std() > 0 and residuals.std() > 0:
            driver_corrs[col] = float(np.corrcoef(val_df[col], residuals)[0, 1])
        else:
            driver_corrs[col] = 0.0

    t_days = (val_df.index - val_df.index[0]).total_seconds().values / 86400.0
    if len(t_days) > 1 and np.std(t_days) > 0:
        time_trend = float(np.polyfit(t_days, residuals.values, 1)[0])
    else:
        time_trend = 0.0

    unbiased = abs(mean_res) < THERMAL_VALIDATION_THRESHOLDS["max_abs_mean_residual_c"]
    structureless_drivers = all(
        abs(c) < THERMAL_VALIDATION_THRESHOLDS["max_driver_correlation"]
        for c in driver_corrs.values()
    )
    structureless_time = (
        abs(time_trend) < THERMAL_VALIDATION_THRESHOLDS["max_time_trend_per_day_c"]
    )

    model_details: dict[str, Any] = {"model_kind": model.model_kind}
    if model.model_kind == "linear" and model.feature_names:
        pipeline = model.model
        ridge = pipeline.named_steps["ridge"]  # type: ignore[union-attr]
        coefs = ridge.coef_.tolist()
        model_details["coefficients"] = dict(zip(model.feature_names, coefs))
        model_details["intercept"] = float(ridge.intercept_)
    elif model.model_kind == "lumped_ode" and isinstance(model.model, LumpedThermalParams):
        model_details["tau_minutes"] = model.model.tau_minutes
        model_details["tau_plausible"] = tau_plausible(
            model.target_column, model.model.tau_minutes
        )
        model_details["beta_0"] = model.model.beta_0
        model_details["beta_p"] = model.model.beta_p
        model_details["beta_rpm"] = model.model.beta_rpm
        model_details["nn_correction"] = model.equilibrium_correction is not None

    passed = unbiased and structureless_drivers and structureless_time

    result: dict[str, Any] = {
        "n_samples": len(val_df),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_residual": mean_res,
        "driver_correlations": driver_corrs,
        "time_trend_per_day": time_trend,
        "unbiased": unbiased,
        "structureless_drivers": structureless_drivers,
        "structureless_time": structureless_time,
        "passed": passed,
        "model_details": model_details,
    }
    if model.model_kind == "lumped_ode" and isinstance(model.model, LumpedThermalParams):
        result["tau_minutes"] = model.model.tau_minutes
        result["tau_plausible"] = tau_plausible(
            model.target_column, model.model.tau_minutes
        )
    return result


__all__ = [
    "GearboxThermalModel",
    "fit_gearbox_thermal",
    "fit_gearbox_thermal_with_selection",
    "healthy_train_validate_split",
    "validate_thermal_model",
]
