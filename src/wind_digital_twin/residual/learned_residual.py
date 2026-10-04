"""Gradient-boosting corrector for systematic physics residual on healthy data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from wind_digital_twin.config import THERMAL_TARGET_COLUMNS, THERMAL_TRAIN_FRACTION
from wind_digital_twin.physics.gearbox_thermal import (
    GearboxThermalModel,
    fit_gearbox_thermal,
    healthy_train_validate_split,
)
from wind_digital_twin.residual.fusion_features import (
    build_corrector_matrix,
    corrector_feature_columns,
)


@dataclass
class TargetResidualCorrector:
    """Predicts physics residual (pred - actual) from exogenous fusion features."""

    model: RegressorMixin
    feature_columns: list[str]
    target_column: str

    def predict_correction(self, df: pd.DataFrame) -> pd.Series:
        """Predict systematic physics residual on rows with corrector features."""
        X = build_corrector_matrix(df)
        if X.empty:
            return pd.Series(dtype=float, name=f"{self.target_column}_corr")
        pred = self.model.predict(X[self.feature_columns].values)
        return pd.Series(pred, index=X.index, name=f"{self.target_column}_corr")


@dataclass
class HybridThermalCorrector:
    """Physics ODE + per-target learned residual correctors."""

    oil_model: GearboxThermalModel
    bear_model: GearboxThermalModel
    oil_corrector: TargetResidualCorrector
    bear_corrector: TargetResidualCorrector

    def hybrid_predict(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """T_hybrid = T_physics - correction_hat."""
        if target_column == THERMAL_TARGET_COLUMNS[0]:
            thermal = self.oil_model
            corrector = self.oil_corrector
        else:
            thermal = self.bear_model
            corrector = self.bear_corrector
        physics = thermal.predict(df)
        correction = corrector.predict_correction(df)
        aligned = physics.reindex(correction.index)
        hybrid = aligned - correction
        return pd.Series(hybrid.values, index=correction.index, name=f"{target_column}_hybrid")

    def hybrid_residual(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """Hybrid pred minus actual (same sign convention as GearboxThermalModel.residual)."""
        sub = df.loc[self.hybrid_predict(df, target_column).index]
        actual = sub[target_column]
        predicted = self.hybrid_predict(df, target_column)
        return pd.Series(
            predicted.values - actual.values,
            index=predicted.index,
            name=f"{target_column}_hybrid_residual",
        )

    def dual_hybrid_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oil and bear hybrid residuals on intersected operating rows."""
        oil = self.hybrid_residual(df, THERMAL_TARGET_COLUMNS[0])
        bear = self.hybrid_residual(df, THERMAL_TARGET_COLUMNS[1])
        common = oil.index.intersection(bear.index)
        return pd.DataFrame(
            {"oil_residual": oil.loc[common], "bear_residual": bear.loc[common]},
            index=common,
        )


def _make_gbm_regressor(random_state: int) -> RegressorMixin:
    """
    Tabular booster for residual correction (LightGBM-class).

    Uses sklearn ``HistGradientBoostingRegressor`` for portable CI; optional
    ``lightgbm`` can be wired behind ``WIND_DT_USE_LIGHTGBM=1`` when stable.
    """
    import os

    if os.environ.get("WIND_DT_USE_LIGHTGBM") == "1":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=80,
                learning_rate=0.08,
                max_depth=4,
                num_leaves=16,
                random_state=random_state,
                verbosity=-1,
                n_jobs=1,
            )
        except ImportError:
            pass
    return HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.08,
        max_depth=4,
        random_state=random_state,
    )


def _fit_target_corrector(
    train_df: pd.DataFrame,
    thermal_model: GearboxThermalModel,
    target_column: str,
    random_state: int = 42,
) -> TargetResidualCorrector:
    X = build_corrector_matrix(train_df)
    if len(X) < 30:
        raise ValueError(f"Insufficient corrector rows for {target_column} ({len(X)})")
    feature_cols = corrector_feature_columns(X)
    y = thermal_model.residual(train_df).reindex(X.index)
    mask = y.notna()
    X_arr = X.loc[mask, feature_cols].values
    y_arr = y.loc[mask].values

    reg = _make_gbm_regressor(random_state)
    reg.fit(X_arr, y_arr)
    return TargetResidualCorrector(
        model=reg,
        feature_columns=feature_cols,
        target_column=target_column,
    )


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = actual.reindex(predicted.index).dropna()
    pred = predicted.reindex(aligned.index)
    if len(aligned) == 0:
        return float("inf")
    return float(np.sqrt(mean_squared_error(aligned, pred)))


def fit_hybrid_thermal_corrector(
    healthy_df: pd.DataFrame,
    train_fraction: float = THERMAL_TRAIN_FRACTION,
    random_state: int = 42,
) -> tuple[HybridThermalCorrector, dict[str, Any]]:
    """
    Fit ODE + LightGBM correctors on healthy data with time-ordered validation.

    Returns final corrector refit on all healthy rows and validation metrics.
    """
    train_df, val_df = healthy_train_validate_split(healthy_df, train_fraction)
    oil_target, bear_target = THERMAL_TARGET_COLUMNS

    oil_model = fit_gearbox_thermal(train_df, oil_target)
    bear_model = fit_gearbox_thermal(train_df, bear_target)
    oil_corr = _fit_target_corrector(train_df, oil_model, oil_target, random_state)
    bear_corr = _fit_target_corrector(train_df, bear_model, bear_target, random_state)

    interim = HybridThermalCorrector(
        oil_model=oil_model,
        bear_model=bear_model,
        oil_corrector=oil_corr,
        bear_corrector=bear_corr,
    )

    metrics: dict[str, Any] = {"targets": {}, "train_rows": len(train_df), "val_rows": len(val_df)}
    for target, thermal in ((oil_target, oil_model), (bear_target, bear_model)):
        slug = "oil" if "Oil" in target else "bear"
        if val_df.empty:
            metrics["targets"][slug] = {
                "physics_val_rmse": float("inf"),
                "hybrid_val_rmse": float("inf"),
                "rmse_reduction_pct": 0.0,
            }
            continue
        actual = val_df[target]
        physics_pred = thermal.predict(val_df)
        hybrid_pred = interim.hybrid_predict(val_df, target)
        phys_rmse = _rmse(actual, physics_pred)
        hybrid_rmse = _rmse(actual, hybrid_pred)
        reduction = (
            100.0 * (phys_rmse - hybrid_rmse) / phys_rmse if phys_rmse > 0 else 0.0
        )
        metrics["targets"][slug] = {
            "physics_val_rmse": phys_rmse,
            "hybrid_val_rmse": hybrid_rmse,
            "rmse_reduction_pct": reduction,
        }

    final_oil = fit_gearbox_thermal(healthy_df, oil_target)
    final_bear = fit_gearbox_thermal(healthy_df, bear_target)
    final = HybridThermalCorrector(
        oil_model=final_oil,
        bear_model=final_bear,
        oil_corrector=_fit_target_corrector(
            healthy_df, final_oil, oil_target, random_state
        ),
        bear_corrector=_fit_target_corrector(
            healthy_df, final_bear, bear_target, random_state
        ),
    )
    return final, metrics


__all__ = [
    "HybridThermalCorrector",
    "TargetResidualCorrector",
    "fit_hybrid_thermal_corrector",
]
