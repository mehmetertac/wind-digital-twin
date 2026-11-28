"""Rise residual score: season shift, one-sided alarms, and fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import SplineTransformer

from wind_digital_twin.anomaly.residual_z import one_sided_max_score
from wind_digital_twin.eval.care_hybrid_eval import numeric_feature_matrix
from wind_digital_twin.eval.care_nbm import CachedEvent, bootstrap_care, care_from_cached
from wind_digital_twin.eval.care_rise_score import (
    apply_bonferroni_z,
    degradation,
    fit_rise,
    fuse_prediction_z,
    mean_residual_by_power_bin,
    operating_frame,
)


def test_one_sided_score_ignores_colder_than_expected():
    fit = {"oil": pd.Series(np.zeros(40))}
    hot = one_sided_max_score(fit, {"oil": pd.Series(np.full(20, 5.0))})
    cold = one_sided_max_score(fit, {"oil": pd.Series(np.full(20, -5.0))})
    assert float(hot.mean()) > 1.0
    assert float(cold.max()) == 0.0


def test_nacelle_term_explains_a_larger_rise_when_the_nacelle_is_cooler():
    index = pd.date_range("2022-01-01", periods=400, freq="10min", tz="UTC")
    power = np.linspace(400.0, 1600.0, len(index))
    rpm = np.full(len(index), 14.0)
    nacelle = 25.0 + 8.0 * np.sin(np.linspace(0.0, 12.0, len(index)))
    rise = 12.0 + 0.004 * power + 0.25 * (30.0 - nacelle)
    fit = pd.DataFrame(
        {
            "Grd_Prod_Pwr_Avg": power,
            "Rtr_RPM_Avg": rpm,
            "Nac_Temp_Avg": nacelle,
            "Gear_Oil_Temp_Avg": nacelle + rise,
        },
        index=index,
    )
    model = fit_rise(fit, "Gear_Oil_Temp_Avg")
    assert model is not None
    assert "Nac_Temp_Avg" in model.columns
    colder = fit.copy()
    colder["Nac_Temp_Avg"] = 16.0
    colder_rise = 12.0 + 0.004 * power + 0.25 * (30.0 - 16.0)
    colder["Gear_Oil_Temp_Avg"] = 16.0 + colder_rise
    assert abs(float(degradation(colder, model).mean())) < 0.5
    colder["Gear_Oil_Temp_Avg"] = colder["Gear_Oil_Temp_Avg"] + 8.0
    assert float(degradation(colder, model).mean()) > 7.0


def test_fusion_keeps_the_finite_score():
    fused = fuse_prediction_z(
        np.array([np.nan, 1.0, 3.0]),
        np.array([2.0, np.nan, 1.0]),
    )
    assert fused.tolist() == [2.0, 1.0, 3.0]


def test_non_operating_rows_hold_the_criticality_counter():
    index = pd.date_range("2023-01-01", periods=80, freq="10min", tz="UTC")
    start, end = index[0], index[-1]

    def _cached(event_id: int, asset: str, label: str, hold: bool) -> CachedEvent:
        return CachedEvent(
            event_id=event_id,
            asset_id=asset,
            label=label,
            description="Gearbox failure" if label == "anomaly" else "nan",
            event_start=start,
            event_end=end,
            pred_index=index,
            z_pred={"hybrid_rise": np.full(len(index), 5.0 if label == "anomaly" else 0.0)},
            z_cal={"hybrid_rise": np.array([0.0, 0.1])},
            pred_normal=np.full(len(index), not hold if label == "anomaly" else True),
        )

    held = care_from_cached(
        [_cached(1, "0", "anomaly", hold=True), _cached(2, "10", "normal", hold=False)],
        "hybrid_rise",
        alpha=0.4,
        mode="loto",
        use_normal_index=True,
    )["summary"]
    counted = care_from_cached(
        [_cached(1, "0", "anomaly", hold=False), _cached(2, "10", "normal", hold=False)],
        "hybrid_rise",
        alpha=0.4,
        mode="loto",
        use_normal_index=True,
    )["summary"]
    assert held["care"] == 0.0
    assert counted["care"] is not None and counted["care"] > 0.0


def test_power_spline_keeps_quadratic_load_bias_under_0_2_c():
    index = pd.date_range("2022-01-01", periods=600, freq="10min", tz="UTC")
    power = np.linspace(200.0, 1800.0, len(index))
    nacelle = 18.0 + 4.0 * np.sin(np.linspace(0.0, 8.0, len(index)))
    rise = 6.0 + 1.5e-6 * (power - 1000.0) ** 2 + 0.15 * (25.0 - nacelle)
    frame = pd.DataFrame(
        {
            "Grd_Prod_Pwr_Avg": power,
            "Rtr_RPM_Avg": np.full(len(index), 12.0),
            "Nac_Temp_Avg": nacelle,
            "Gear_Oil_Temp_Avg": nacelle + rise,
        },
        index=index,
    )
    model = fit_rise(frame, "Gear_Oil_Temp_Avg")
    assert model is not None
    spline = model.pipeline.named_steps["features"].named_transformers_["power"]
    assert isinstance(spline, SplineTransformer)
    bins = mean_residual_by_power_bin(frame, model, rated_kw=2000.0, width=0.2)
    assert bins
    assert max(abs(value) for value in bins.values()) < 0.2


def test_status_type_id_never_enters_features():
    raw = pd.DataFrame(
        {
            "time_stamp": pd.date_range("2022-06-01", periods=6, freq="10min", tz="UTC"),
            "Grd_Prod_Pwr_Avg": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "Rtr_RPM_Avg": [12.0] * 6,
            "Nac_Temp_Avg": [15.0] * 6,
            "Gear_Oil_Temp_Avg": [40.0] * 6,
            "sensor_99_avg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "status_type_id": [0, 2, 3, 4, 3, 0],
        }
    )
    channel_map = pd.DataFrame({"feature": pd.Series(dtype=str), "twin_role": pd.Series(dtype=str)})
    frame = operating_frame(raw, channel_map, ("Gear_Oil_Temp_Avg",))
    matrix = numeric_feature_matrix(raw)
    assert "status_type_id" not in frame.columns
    assert "status_type_id" not in matrix.columns
    assert "sensor_99_avg" in matrix.columns


def _rise_event(event_id: int, asset: str, label: str, oil_pred: float, oil_cal: np.ndarray) -> CachedEvent:
    index = pd.date_range("2024-01-01", periods=40, freq="10min", tz="UTC")
    return CachedEvent(
        event_id=event_id,
        asset_id=asset,
        label=label,
        description="Gearbox failure" if label == "anomaly" else "nan",
        event_start=index[0],
        event_end=index[-1],
        pred_index=index,
        channel_pred={
            "hybrid_rise": {
                "Gear_Oil_Temp_Avg": np.full(len(index), oil_pred),
                "Gear_Bear_Temp_Avg": np.zeros(len(index)),
            }
        },
        channel_cal={
            "hybrid_rise": {
                "Gear_Oil_Temp_Avg": oil_cal,
                "Gear_Bear_Temp_Avg": np.zeros(20),
            }
        },
    )


def test_bonferroni_min_p_ignores_the_test_asset_calibration():
    events = [
        _rise_event(1, "1", "normal", 0.0, np.zeros(80)),
        _rise_event(2, "2", "anomaly", 8.0, np.full(200, 50.0)),
    ]
    apply_bonferroni_z(events, "hybrid_rise", alpha=0.2, n_channels=2)
    report = care_from_cached(events, "hybrid_rise", alpha=0.2, mode="loto")
    alarms = {row["event_id"]: row["n_alarms"] for row in report["events"]}
    assert alarms[1] == 0
    assert alarms[2] == 40


def test_bootstrap_interval_matches_a_two_event_score():
    events = [
        _rise_event(1, "1", "normal", 0.0, np.zeros(80)),
        _rise_event(2, "2", "anomaly", 8.0, np.full(200, 50.0)),
    ]
    apply_bonferroni_z(events, "hybrid_rise", alpha=0.2, n_channels=2)
    boot = bootstrap_care(events, "hybrid_rise", n=40, seed=0, alpha=0.2, mode="loto")
    assert boot["ci90"][0] == boot["care"]
    assert boot["ci90"][1] == boot["care"]
