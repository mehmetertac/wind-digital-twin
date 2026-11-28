"""Identity-disjoint thresholds, CARE event merges, and lead-time text."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.data.load_care import CareEvent
from wind_digital_twin.eval.care_hybrid_eval import is_generator_bearing
from wind_digital_twin.eval.care_nbm import CachedEvent, care_from_cached
from wind_digital_twin.eval.care_rul import (
    failure_log_frame,
    independent_onsets,
    select_failure_events,
)
from wind_digital_twin.eval.identity_calibration import (
    chronological_fit_cal_split,
    chronological_fit_cal_split_uncapped,
    conformal_pvalues,
    conformal_rank_threshold,
    format_crossed_lead,
    threshold_excluding_assets,
)


def _event(event_id: int, asset: str, start: str, end: str, description: str) -> CareEvent:
    return CareEvent(
        event_id,
        asset,
        "anomaly",
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        description,
    )


def test_chronological_split_is_disjoint_and_capped():
    index = pd.date_range("2020-01-01", periods=40_000, freq="10min", tz="UTC")
    frame = pd.DataFrame({"value": np.arange(len(index))}, index=index)
    fit, cal = chronological_fit_cal_split(frame)
    assert len(fit) == 12_000
    assert len(cal) == 6_000
    assert fit.index.max() < cal.index.min()


def test_uncapped_split_keeps_the_earlier_block():
    index = pd.date_range("2020-01-01", periods=40_000, freq="10min", tz="UTC")
    frame = pd.DataFrame({"value": np.arange(len(index))}, index=index)
    fit, cal = chronological_fit_cal_split_uncapped(frame)
    assert len(fit) == 32_000
    assert len(cal) == 6_000
    assert fit.index.max() < cal.index.min()
    assert fit.index.min() == index[0]


def test_threshold_does_not_use_the_test_asset():
    pool = {
        "0": np.array([0.0, 0.2, 0.4, 0.6]),
        "21": np.array([100.0, 100.0, 100.0, 100.0]),
    }
    threshold = threshold_excluding_assets(pool, {"21"}, alpha=0.25)
    assert threshold < 10.0
    assert threshold == conformal_rank_threshold(pool["0"], alpha=0.25)


def test_conformal_pvalues_are_an_upper_tail_rank():
    cal = np.array([0.0, 1.0, 2.0, 3.0])
    p = conformal_pvalues(cal, np.array([3.0, 10.0, np.nan]))
    assert p[0] == 2.0 / 5.0
    assert p[1] == 1.0 / 5.0
    assert np.isnan(p[2])
    assert conformal_pvalues(np.array([]), np.array([4.0]))[0] == 1.0


def test_lead_text_is_an_integer_or_an_explicit_miss():
    assert format_crossed_lead(14.2, "51") == "14"
    assert format_crossed_lead(None, "51") == "never crossed on event 51"
    assert format_crossed_lead(float("nan"), "10") == "never crossed on event 10"
    assert "none" not in format_crossed_lead(None, "72").lower()


def test_overlapping_gearbox_labels_collapse_to_the_earlier_onset():
    events = [
        _event(51, "21", "2023-10-06", "2023-10-17", "Gearbox bearings damaged"),
        _event(72, "21", "2023-10-10", "2023-10-17", "Gearbox failure"),
        _event(10, "10", "2023-10-11", "2023-10-18", "Gearbox failure"),
        _event(0, "0", "2023-08-06", "2023-08-20", "Generator bearing failure"),
    ]
    kept, dropped = independent_onsets(select_failure_events(events, "gearbox"))
    assert [event.event_id for event in kept] == [10, 51]
    assert dropped[0]["event_id"] == 72
    drivetrain, _ = independent_onsets(select_failure_events(events, "drivetrain"))
    assert [event.event_id for event in drivetrain] == [0, 10, 51]
    log = failure_log_frame(kept)
    assert "component" not in log.columns
    assert log["remarks"].str.startswith("51|").any()
    assert is_generator_bearing("Generator bearing failure")


def test_cached_care_threshold_changes_with_other_assets_only():
    index = pd.date_range("2023-01-01", periods=4, freq="10min", tz="UTC")
    start, end = index[1], index[2]

    def _cached(event_id: int, asset: str, label: str, pred_z: list[float], cal_z: list[float]) -> CachedEvent:
        return CachedEvent(
            event_id=event_id,
            asset_id=asset,
            label=label,
            description="Gearbox failure" if label == "anomaly" else "nan",
            event_start=start,
            event_end=end,
            pred_index=index,
            z_pred={"hybrid": np.array(pred_z, dtype=float)},
            z_cal={"hybrid": np.array(cal_z, dtype=float)},
        )

    events = [
        _cached(1, "0", "anomaly", [0.0, 5.0, 5.0, 0.0], [0.0, 0.1]),
        _cached(2, "10", "normal", [0.0, 0.0, 0.0, 0.0], [0.0, 0.2]),
    ]
    summary = care_from_cached(events, "hybrid", alpha=0.4, mode="loto")["summary"]
    assert summary["care"] is not None
    assert summary["accuracy"] == 1.0
