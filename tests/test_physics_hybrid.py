"""Tests for physics-residual hybrid detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.config import THERMAL_TARGET_COLUMNS
from tests.thermal_fixtures import synthetic_thermal_df as _synthetic_thermal_df
from wind_digital_twin.physics.gearbox_thermal import fit_gearbox_thermal
from wind_digital_twin.anomaly.physics_hybrid import PhysicsHybridPipeline, fit_physics_hybrid
from wind_digital_twin.residual.residual_features import (
    build_residual_feature_frame,
    compute_dual_residuals,
    degradation_signal,
)


def test_degradation_signal_direction():
    residuals = pd.Series([1.0, -3.0, 0.0])
    signal = degradation_signal(residuals)
    assert signal.iloc[0] == -1.0
    assert signal.iloc[1] == 3.0
    assert signal.iloc[2] == 0.0


def test_build_residual_feature_frame_drops_incomplete_windows():
    df = _synthetic_thermal_df(n=300)
    oil_model = fit_gearbox_thermal(df.iloc[:200], THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(df.iloc[:200], THERMAL_TARGET_COLUMNS[1])
    features = build_residual_feature_frame(df, oil_model, bear_model)
    assert not features.empty
    assert "oil_deg" in features.columns
    assert "bear_ewma" in features.columns
    assert "oil_roll_mean_144" in features.columns
    assert len(features) < len(df)


def test_injected_offset_increases_degradation_features():
    df = _synthetic_thermal_df(n=400)
    train = df.iloc[:250]
    test = df.iloc[250:].copy()
    oil_model = fit_gearbox_thermal(train, THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(train, THERMAL_TARGET_COLUMNS[1])

    healthy_features = build_residual_feature_frame(test, oil_model, bear_model)
    shifted = test.copy()
    shifted[THERMAL_TARGET_COLUMNS[0]] += 5.0
    shifted[THERMAL_TARGET_COLUMNS[1]] += 5.0
    degraded_features = build_residual_feature_frame(shifted, oil_model, bear_model)

    assert degraded_features["oil_deg"].mean() > healthy_features["oil_deg"].mean()
    assert degraded_features["bear_deg"].mean() > healthy_features["bear_deg"].mean()


def test_compute_dual_residuals_aligned():
    df = _synthetic_thermal_df(n=100)
    oil_model = fit_gearbox_thermal(df.iloc[:60], THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(df.iloc[:60], THERMAL_TARGET_COLUMNS[1])
    residuals = compute_dual_residuals(df, oil_model, bear_model)
    assert list(residuals.columns) == ["oil_residual", "bear_residual"]
    assert len(residuals) == len(df)


def test_physics_hybrid_scores_higher_on_degraded_data():
    df = _synthetic_thermal_df(n=800)
    train = df.iloc[:400]
    test_healthy = df.iloc[400:600]
    test_shifted = test_healthy.copy()
    test_shifted[THERMAL_TARGET_COLUMNS[0]] += 6.0
    test_shifted[THERMAL_TARGET_COLUMNS[1]] += 6.0

    pipeline = fit_physics_hybrid(train)
    assert isinstance(pipeline, PhysicsHybridPipeline)

    healthy_scores = pipeline.score(test_healthy)
    degraded_scores = pipeline.score(test_shifted)
    assert len(healthy_scores) > 0
    assert len(degraded_scores) > 0
    assert degraded_scores.mean() > healthy_scores.mean()


def test_physics_hybrid_end_to_end_smoke():
    df = _synthetic_thermal_df(n=500)
    pipeline = fit_physics_hybrid(df.iloc[:300])
    scores = pipeline.score(df.iloc[300:])
    assert len(scores) > 0
    assert scores.notna().all()
