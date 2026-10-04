"""Anomaly detector on learned hybrid residuals + fusion context."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wind_digital_twin.anomaly.isolation_forest import (
    IsolationForestPipeline,
    fit_isolation_forest,
)
from wind_digital_twin.config import (
    DEFAULT_CONTAMINATION,
    RESIDUAL_EWMA_SPAN,
    RESIDUAL_ROLLING_WINDOWS,
)
from wind_digital_twin.residual.fusion_features import (
    DETECTOR_FUSION_CONTEXT_COLUMNS,
    build_fusion_feature_frame,
)
from wind_digital_twin.residual.learned_residual import (
    HybridThermalCorrector,
    fit_hybrid_thermal_corrector,
)
from wind_digital_twin.residual.residual_features import (
    build_residual_feature_frame_from_series,
)


def build_hybrid_detector_feature_frame(
    df: pd.DataFrame,
    hybrid: HybridThermalCorrector,
    ewma_span: int = RESIDUAL_EWMA_SPAN,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """EWMA/rolling on hybrid residuals plus theta / delta context."""
    rolling_windows = rolling_windows or RESIDUAL_ROLLING_WINDOWS
    residuals = hybrid.dual_hybrid_residuals(df)
    fusion = build_fusion_feature_frame(df)
    context_cols = [c for c in DETECTOR_FUSION_CONTEXT_COLUMNS if c in fusion.columns]
    extra = fusion[context_cols].reindex(residuals.index)
    return build_residual_feature_frame_from_series(
        residuals["oil_residual"],
        residuals["bear_residual"],
        ewma_span=ewma_span,
        rolling_windows=rolling_windows,
        extra_columns=extra,
    )


@dataclass
class HybridResidualPipeline:
    """Learned hybrid thermal corrector + IF on hybrid residual windows."""

    hybrid: HybridThermalCorrector
    if_pipeline: IsolationForestPipeline
    feature_columns: list[str]
    ewma_span: int = RESIDUAL_EWMA_SPAN
    rolling_windows: list[int] | None = None

    def score(self, X: pd.DataFrame) -> pd.Series:
        features = build_hybrid_detector_feature_frame(
            X,
            self.hybrid,
            ewma_span=self.ewma_span,
            rolling_windows=self.rolling_windows,
        )
        if features.empty:
            return pd.Series(dtype=float, name="anomaly_score")
        raw = self.if_pipeline.score(features)
        return pd.Series(raw.values, index=features.index, name="anomaly_score")


def fit_hybrid_residual_detector(
    X_train: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    ewma_span: int = RESIDUAL_EWMA_SPAN,
    rolling_windows: list[int] | None = None,
) -> HybridResidualPipeline:
    """Fit hybrid corrector and IF on hybrid-residual features (healthy rows only)."""
    rolling_windows = rolling_windows or RESIDUAL_ROLLING_WINDOWS
    hybrid, _metrics = fit_hybrid_thermal_corrector(X_train)
    train_features = build_hybrid_detector_feature_frame(
        X_train,
        hybrid,
        ewma_span=ewma_span,
        rolling_windows=rolling_windows,
    )
    if len(train_features) < 50:
        raise ValueError(
            f"Insufficient hybrid feature rows for IF training ({len(train_features)})"
        )
    feature_cols = train_features.columns.tolist()
    if_pipeline = fit_isolation_forest(
        train_features,
        feature_cols,
        contamination=contamination,
    )
    return HybridResidualPipeline(
        hybrid=hybrid,
        if_pipeline=if_pipeline,
        feature_columns=feature_cols,
        ewma_span=ewma_span,
        rolling_windows=rolling_windows,
    )


__all__ = [
    "HybridResidualPipeline",
    "build_hybrid_detector_feature_frame",
    "fit_hybrid_residual_detector",
]
