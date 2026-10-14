"""End-to-end twin pass on synthetic T06."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from wind_digital_twin.config import DEFAULT_BUFFER_DAYS, RESIDUAL_ROLLING_WINDOWS
from wind_digital_twin.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_digital_twin.data.load_edp import load_edp_dataset
from wind_digital_twin.data.synthetic_edp import generate_synthetic_edp_dataset
from wind_digital_twin.eval.twin_pass import (
    build_twin_layer_frame,
    run_twin_pass,
    threshold_from_healthy_holdout,
)
from wind_digital_twin.anomaly.hybrid_residual_detector import fit_hybrid_residual_detector
from wind_digital_twin.physics.gearbox_thermal import healthy_train_validate_split


def test_twin_pass_synthetic_t06(tmp_path):
    generate_synthetic_edp_dataset(tmp_path, random_state=11)
    fig_dir = tmp_path / "figures"
    res_dir = tmp_path / "results"

    result = run_twin_pass(
        raw_dir=tmp_path,
        turbine_id="T06",
        lookback_days=120,
        figures_dir=fig_dir,
        results_dir=res_dir,
    )

    layer = result.layer_frame
    assert not layer.empty
    turbines, failures = load_edp_dataset(tmp_path)
    df = clean_turbine_df(turbines["T06"], min_power_kw=50.0)
    failure = get_failure_for_turbine("T06", failures)
    assert failure is not None
    assert layer.index.min() <= pd.Timestamp(failure.timestamp)
    assert layer.index.max() >= pd.Timestamp(failure.timestamp) - timedelta(days=30)

    for col in (
        "bear_actual",
        "bear_physics_pred",
        "bear_hybrid_pred",
        "bear_physics_residual",
        "bear_hybrid_residual",
    ):
        assert col in layer.columns
        assert layer[col].notna().all()

    scored = layer["anomaly_score"].dropna()
    assert len(scored) > 100
    warmup = max(RESIDUAL_ROLLING_WINDOWS)
    assert layer["anomaly_score"].isna().sum() >= warmup - 1

    assert np.isfinite(result.threshold)

    healthy = df.loc[
        healthy_training_mask(df.index, failure, DEFAULT_BUFFER_DAYS)
    ]
    _train, val_df = healthy_train_validate_split(healthy)
    buffer_cut = pd.Timestamp(failure.timestamp) - timedelta(days=DEFAULT_BUFFER_DAYS)
    assert val_df.index.max() < buffer_cut

    assert result.timeline_figure.is_file()
    assert result.rmse_figure.is_file()
    assert result.frame_path.is_file()


def test_build_layer_frame_index_alignment(tmp_path):
    generate_synthetic_edp_dataset(tmp_path, random_state=3)
    turbines, failures = load_edp_dataset(tmp_path)
    df = clean_turbine_df(turbines["T06"], min_power_kw=50.0)
    failure = get_failure_for_turbine("T06", failures)
    healthy = df.loc[healthy_training_mask(df.index, failure, 90)]
    pipe = fit_hybrid_residual_detector(healthy.iloc[:3000])
    layer = build_twin_layer_frame(df, pipe)
    assert layer.index.is_monotonic_increasing
    assert (layer["bear_physics_residual"] == layer["bear_physics_pred"] - layer["bear_actual"]).all()
    assert (layer["bear_hybrid_residual"] == layer["bear_hybrid_pred"] - layer["bear_actual"]).all()


def test_threshold_not_from_buffer(tmp_path):
    generate_synthetic_edp_dataset(tmp_path, random_state=5)
    turbines, failures = load_edp_dataset(tmp_path)
    df = clean_turbine_df(turbines["T06"], min_power_kw=50.0)
    failure = get_failure_for_turbine("T06", failures)
    healthy = df.loc[healthy_training_mask(df.index, failure, 90)]
    pipe = fit_hybrid_residual_detector(healthy)
    _thresh, val_df = threshold_from_healthy_holdout(pipe, healthy)
    buffer_cut = pd.Timestamp(failure.timestamp) - timedelta(days=90)
    assert val_df.index.max() < buffer_cut
