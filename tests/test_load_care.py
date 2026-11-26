"""Tests for CARE dataset loader and channel map."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wind_digital_twin.data.care_channels import (
    build_channel_map,
    classify_feature,
    load_feature_description,
    role_rename_columns,
)
from wind_digital_twin.data.load_care import load_care_dataset, load_event_info, to_twin_frame

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "care_wfa"


def test_classify_feature_groups():
    assert classify_feature("Grid power", "kW") == "electrical"
    assert classify_feature("Temperature oil in gearbox", "C") == "thermal"
    assert classify_feature("Rotor rpm", "rpm") == "mechanical"


def test_feature_description_expands_multi_stat():
    df = load_feature_description(FIXTURE_DIR / "feature_description.csv")
    assert "wind_speed_3_avg" in df["feature"].values


def test_load_event_info_v6_timestamps():
    events = load_event_info(FIXTURE_DIR)
    assert len(events) == 2
    assert events[0].event_id == 99
    assert events[0].label == "anomaly"


def test_twin_frame_rename_and_rescale(tmp_path: Path):
    df = load_feature_description(FIXTURE_DIR / "feature_description.csv")
    channel_map = build_channel_map(df)
    rename = role_rename_columns(channel_map)
    assert rename["power_30_avg"] == "Grd_Prod_Pwr_Avg"
    frame = pd.DataFrame(
        {
            "time_stamp": pd.date_range("2023-01-01", periods=3, freq="10min", tz="UTC"),
            "power_30_avg": [0.5, 0.25, 0.1],
            "sensor_12_avg": [40.0, 41.0, 42.0],
        }
    )
    twin = to_twin_frame(frame, channel_map, rescale_power=True, rated_kw=2000.0)
    assert "Grd_Prod_Pwr_Avg" in twin.columns
    assert twin["Grd_Prod_Pwr_Avg"].iloc[0] == pytest.approx(1000.0)


def test_load_care_dataset_split(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (tmp_path / "event_info.csv").write_text(
        (FIXTURE_DIR / "event_info.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    header = (
        "time_stamp;asset_id;id;train_test;status_type_id;"
        "sensor_0_avg;power_30_avg;sensor_12_avg\n"
    )
    rows = []
    idx = pd.date_range("2023-01-01", periods=120, freq="10min", tz="UTC")
    for i, ts in enumerate(idx):
        split = "train" if i < 80 else "prediction"
        status = 0 if i < 80 else 4
        rows.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')};10;{i};{split};{status};20;0.5;40\n")
    (datasets / "99.csv").write_text(header + "".join(rows), encoding="utf-8")

    ds = load_care_dataset(tmp_path, 99)
    assert len(ds.train) == 80
    assert len(ds.prediction) == 40
    assert int(ds.train_normal_mask.sum()) == 80