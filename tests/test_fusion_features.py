"""Tests for fusion feature frame."""

from __future__ import annotations

import pandas as pd

from tests.thermal_fixtures import synthetic_thermal_df
from wind_digital_twin.residual.fusion_features import (
    build_corrector_matrix,
    build_fusion_feature_frame,
    corrector_feature_columns,
)


def test_build_fusion_feature_frame_theta_and_delta():
    df = synthetic_thermal_df(n=200)
    fusion = build_fusion_feature_frame(df)
    assert not fusion.empty
    assert "theta_oil" in fusion.columns
    assert "theta_bear" in fusion.columns
    assert "delta_bear_oil" in fusion.columns
    assert "torque_proxy" in fusion.columns
    assert fusion["theta_oil"].notna().all()


def test_corrector_features_exclude_theta():
    df = synthetic_thermal_df(n=100)
    fusion = build_fusion_feature_frame(df)
    cols = corrector_feature_columns(fusion)
    assert "theta_oil" not in cols
    assert "theta_bear" not in cols
    assert "delta_bear_oil" not in cols
    assert "power_sq" in cols


def test_build_corrector_matrix_aligned():
    df = synthetic_thermal_df(n=150)
    X = build_corrector_matrix(df)
    assert len(X) <= len(df)
    assert X.index.is_monotonic_increasing


def test_resolve_feature_columns_keeps_optional_when_present():
    from wind_digital_twin.data.clean import resolve_feature_columns

    df = synthetic_thermal_df(n=10)
    df["Amb_WindSpeed_Avg"] = 8.0
    df["Amb_WindDir_Relative_Avg"] = 180.0
    df["Gen_RPM_Avg"] = 1000.0
    cols = resolve_feature_columns(df)
    assert "Gen_RPM_Avg" in cols
