"""Tests for vendored CARE score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_digital_twin.eval.care_score import CAREScore, calculate_criticality


def test_criticality_ramps_and_decays():
    idx = pd.date_range("2020-01-01", periods=6, freq="10min", tz="UTC")
    anomalies = pd.Series([True, True, False, False, True, True], index=idx)
    crit = calculate_criticality(anomalies)
    assert crit.tolist() == [1, 2, 1, 0, 1, 2]


def test_criticality_ignores_abnormal_status():
    idx = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
    anomalies = pd.Series([True, True, True, True], index=idx)
    normal_idx = pd.Series([True, True, False, False], index=idx)
    crit = calculate_criticality(anomalies, normal_idx=normal_idx)
    assert crit.tolist()[2:] == [2, 2]


def test_special_cases_final_score():
    scorer = CAREScore(criticality_threshold=2)
    idx = pd.date_range("2020-01-01", periods=10, freq="10min", tz="UTC")
    preds = pd.Series([False] * 10, index=idx)
    scorer.evaluate_event(
        event_start=idx[2],
        event_end=idx[8],
        event_label="anomaly",
        predicted_anomalies=preds,
        ignore_normal_index=True,
        event_id=1,
    )
    scorer.evaluate_event(
        event_start=idx[0],
        event_end=idx[5],
        event_label="normal",
        predicted_anomalies=preds,
        ignore_normal_index=True,
        event_id=2,
    )
    assert scorer.get_final_score() == 0.0

    scorer2 = CAREScore(criticality_threshold=2)
    rand = pd.Series(np.random.default_rng(0).random(10) < 0.5, index=idx)
    scorer2.evaluate_event(idx[2], idx[8], "anomaly", rand, ignore_normal_index=True, event_id=1)
    scorer2.evaluate_event(idx[0], idx[5], "normal", pd.Series(False, index=idx), ignore_normal_index=True, event_id=2)
    if scorer2.calculate_avg_accuracy() <= 0.5:
        assert scorer2.get_final_score() == scorer2.calculate_avg_accuracy()


def test_all_normal_strategy_gets_zero():
    idx = pd.date_range("2020-01-01", periods=20, freq="10min", tz="UTC")
    scorer = CAREScore(criticality_threshold=72)
    preds = pd.Series(False, index=idx)
    scorer.evaluate_event(idx[5], idx[15], "anomaly", preds, ignore_normal_index=True, event_id=1)
    scorer.evaluate_event(idx[0], idx[10], "normal", preds, ignore_normal_index=True, event_id=2)
    assert scorer.get_final_score() == 0.0


def test_random_pointwise_accuracy_near_half():
    rng = np.random.default_rng(42)
    scorer = CAREScore(criticality_threshold=72)
    for eid, label in enumerate(["anomaly", "normal"] * 4):
        idx = pd.date_range("2020-01-01", periods=500, freq="10min", tz="UTC") + pd.Timedelta(days=eid)
        preds = pd.Series(rng.random(len(idx)) < 0.5, index=idx)
        scorer.evaluate_event(idx[50], idx[200], label, preds, ignore_normal_index=True, event_id=eid)
    assert scorer.calculate_avg_accuracy() == pytest.approx(0.5, abs=0.08)


def test_paper_v2_earliness_preset():
    assert CAREScore.paper_v2().ws_start_of_descend == (1, 2)