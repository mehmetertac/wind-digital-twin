"""Tests for CARE hybrid / AE evaluation helpers."""

from __future__ import annotations

import pandas as pd

from wind_digital_twin.data.load_care import CareEvent
from wind_digital_twin.eval.care_hybrid_eval import (
    event_in_care_scope,
    is_gearbox_anomaly,
    scores_to_persistent_alarms,
)


def test_is_gearbox_anomaly():
    assert is_gearbox_anomaly("Gearbox bearings damaged")
    assert not is_gearbox_anomaly("Hydraulic group")


def test_event_in_care_scope_gearbox_only():
    ds_anom = type(
        "DS",
        (),
        {
            "event": CareEvent(
                1, "10", "anomaly", pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC"), "Hydraulic group"
            )
        },
    )()
    ds_gb = type(
        "DS",
        (),
        {
            "event": CareEvent(
                2, "10", "anomaly", pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC"), "Gearbox failure"
            )
        },
    )()
    assert not event_in_care_scope(ds_anom, gearbox_only=True)
    assert event_in_care_scope(ds_gb, gearbox_only=True)
    assert event_in_care_scope(ds_anom, gearbox_only=False)


def test_scores_to_persistent_alarms():
    idx = pd.date_range("2020-01-01", periods=12, freq="10min", tz="UTC")
    scores = pd.Series([0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2], index=idx)
    alarms = scores_to_persistent_alarms(scores, threshold=1.0, persistence_samples=3)
    assert alarms.iloc[2:8].all()
    assert not alarms.iloc[0:2].any()