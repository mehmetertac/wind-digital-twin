"""CARE score for CARE-to-Compare benchmark evaluation.

Vendored from AEFDI/EnergyFaultDetector (MIT), commit cfcae85f82fab8a6bcec3319837ac7db21d05a70
and surrounding care_score / analysis.calculate_criticality on main as of 2025-10.
Paper: arXiv:2404.10320 / DOI 10.3390/data9120138.

Copyright 2025, Fraunhofer Institute for Energy Economics and Energy System Technology (IEE), Kassel.
See module docstring in upstream for full MIT license text.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from wind_digital_twin.config import CARE_CRITICALITY_THRESHOLD


def calculate_criticality(
    anomalies: pd.Series,
    normal_idx: pd.Series | None = None,
    init_criticality: int = 0,
    max_criticality: int = 1000,
) -> pd.Series:
    """Criticality counter for event-level reliability (Algorithm 1, corrected)."""
    anomalies = anomalies.sort_index()

    if normal_idx is None:
        normal_idx = pd.Series(np.full(len(anomalies), True), index=anomalies.index)
    else:
        normal_idx = normal_idx.sort_index().reindex(anomalies.index)
        normal_idx = normal_idx.fillna(True).astype(bool)

    if len(anomalies) != len(normal_idx):
        raise ValueError("length mismatch between anomalies and normal_idx.")

    group_level = _detect_group_level(anomalies.index)
    if group_level is not None:
        parts = []
        for _, group_anomalies in anomalies.groupby(level=group_level):
            group_normal = normal_idx.loc[group_anomalies.index]
            parts.append(
                _compute_criticality_array(
                    group_anomalies, group_normal, init_criticality, max_criticality
                )
            )
        return pd.concat(parts)

    return _compute_criticality_array(
        anomalies, normal_idx, init_criticality, max_criticality
    )


def _detect_group_level(index: pd.Index) -> int | None:
    if not isinstance(index, pd.MultiIndex):
        return None
    for i, level in enumerate(index.levels):
        if not isinstance(level, pd.DatetimeIndex):
            return i
    return None


def _compute_criticality_array(
    anomalies: pd.Series,
    normal_idx: pd.Series,
    init_criticality: int,
    max_criticality: int,
) -> pd.Series:
    normal_arr = normal_idx.to_numpy(dtype=bool)
    anomaly_arr = anomalies.to_numpy(dtype=bool)
    deltas = np.where(
        normal_arr & anomaly_arr,
        1,
        np.where(normal_arr & ~anomaly_arr, -1, 0),
    )

    crit = np.empty(len(deltas), dtype=np.int64)
    c = init_criticality
    for i, d in enumerate(deltas):
        c = max(0, min(max_criticality, c + int(d)))
        crit[i] = c
    return pd.Series(crit, index=anomalies.index)


class CAREScore:
    """Coverage, Accuracy, Reliability, Earliness for early fault detection."""

    def __init__(
        self,
        coverage_beta: float = 0.5,
        reliability_beta: float = 0.5,
        coverage_w: float = 1.0,
        accuracy_w: float = 2.0,
        earliness_w: float = 1.0,
        reliability_w: float = 1.0,
        criticality_threshold: int = CARE_CRITICALITY_THRESHOLD,
        min_fraction_anomalous: float = 0.1,
        ws_start_of_descend: tuple[int, int] = (1, 4),
        anomaly_detection_method: str = "criticality",
    ) -> None:
        if anomaly_detection_method not in ("criticality", "fraction"):
            raise ValueError("anomaly_detection_method must be 'criticality' or 'fraction'")

        self.coverage_beta = coverage_beta
        self.reliability_beta = reliability_beta
        self.coverage_w = coverage_w
        self.accuracy_w = accuracy_w
        self.earliness_w = earliness_w
        self.reliability_w = reliability_w
        self.min_fraction_anomalous = min_fraction_anomalous
        self.criticality_threshold = criticality_threshold
        self.anomaly_detection_method = anomaly_detection_method
        self.ws_start_of_descend = ws_start_of_descend
        self._evaluated_events: list[dict[str, Any]] = []

    @classmethod
    def paper_v2(cls, **kwargs: Any) -> CAREScore:
        """Earliness decay at 1/2 of event length (arXiv v2 text)."""
        return cls(ws_start_of_descend=(1, 2), **kwargs)

    @property
    def evaluated_events(self) -> pd.DataFrame:
        df = pd.DataFrame(self._evaluated_events)
        if df.empty:
            return df
        if self.anomaly_detection_method == "criticality":
            df["anomaly_detected"] = df["max_criticality"] >= self.criticality_threshold
        else:
            df["anomaly_detected"] = (
                (df["tp"] + df["fp"]) / (df["tp"] + df["fp"] + df["fn"] + df["tn"])
            ) >= self.min_fraction_anomalous
        return df

    def evaluate_event(
        self,
        event_start: int | pd.Timestamp,
        event_end: int | pd.Timestamp,
        event_label: str,
        predicted_anomalies: pd.Series,
        normal_index: pd.Series | None = None,
        evaluate_until_event_end: bool | str = False,
        event_id: int | None = None,
        ignore_normal_index: bool = False,
    ) -> dict[str, Any]:
        if event_label not in ("anomaly", "normal"):
            raise ValueError("event_label must be 'anomaly' or 'normal'")

        allowed = (True, False, "normal_only", "anomaly_only", "True", "False")
        if evaluate_until_event_end not in allowed:
            raise ValueError(f"evaluate_until_event_end must be one of {allowed}")

        predicted_anomalies = predicted_anomalies.sort_index()
        if normal_index is None or ignore_normal_index:
            normal_index = pd.Series(True, index=predicted_anomalies.index)
        else:
            normal_index = normal_index.sort_index()

        cap_data = (
            evaluate_until_event_end in (True, "True")
            or (event_label == "anomaly" and evaluate_until_event_end == "anomaly_only")
            or (event_label == "normal" and evaluate_until_event_end == "normal_only")
        )
        if cap_data:
            predicted_anomalies = predicted_anomalies.loc[:event_end]
            normal_index = normal_index.loc[:event_end]

        ground_truth = self.create_ground_truth(
            event_start, event_end, normal_index, event_label
        )
        predicted_anomalies_event = predicted_anomalies.loc[event_start:event_end]
        if predicted_anomalies_event.empty:
            raise ValueError("No event data could be selected.")

        weighted_score_value = (
            self._calculate_weighted_score(predicted_anomalies_event)
            if event_label == "anomaly"
            else np.nan
        )

        criticality = calculate_criticality(
            anomalies=predicted_anomalies.loc[:event_end],
            normal_idx=normal_index.loc[:event_end],
        )
        max_criticality = float(np.max(criticality))

        scored_truth = ground_truth[normal_index]
        scored_pred = predicted_anomalies[normal_index]
        if scored_truth.empty:
            # Every row was held out of the operating mask, so there is no
            # confusion matrix. Criticality above already stayed at zero.
            metrics = {
                "f_beta_score": np.nan,
                "accuracy": np.nan,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
        else:
            metrics = self._calculate_basic_metrics(
                ground_truth=scored_truth,
                predicted_anomalies=scored_pred,
                event_label=event_label,
            )

        if event_id is None:
            event_id = len(self._evaluated_events)

        evaluation = {
            "event_id": event_id,
            "event_label": event_label,
            "weighted_score": weighted_score_value,
            "max_criticality": max_criticality,
            **metrics,
        }
        self._evaluated_events.append(evaluation)
        return evaluation

    def get_final_score(
        self,
        event_selection: list[int] | None = None,
        criticality_threshold: int | None = None,
        min_fraction_anomalous: float | None = None,
    ) -> float:
        if self.anomaly_detection_method == "criticality" and criticality_threshold is not None:
            self.criticality_threshold = criticality_threshold
        if self.anomaly_detection_method == "fraction" and min_fraction_anomalous is not None:
            self.min_fraction_anomalous = min_fraction_anomalous

        events_to_evaluate = self._select_events(event_selection)
        if events_to_evaluate["event_label"].nunique() < 2:
            raise ValueError("Need at least one normal and one anomalous event.")

        if np.sum(events_to_evaluate["anomaly_detected"]) == 0:
            return 0.0

        is_anomaly_event = events_to_evaluate["event_label"] == "anomaly"
        avg_accuracy = float(events_to_evaluate.loc[~is_anomaly_event, "accuracy"].mean())
        if avg_accuracy <= 0.5:
            return avg_accuracy

        avg_f_score = float(events_to_evaluate.loc[is_anomaly_event, "f_beta_score"].mean())
        avg_weighted_score = float(events_to_evaluate.loc[is_anomaly_event, "weighted_score"].mean())
        eventwise_fscore = self.calculate_reliability(event_selection)

        care_score = (
            avg_accuracy * self.accuracy_w
            + avg_weighted_score * self.earliness_w
            + avg_f_score * self.coverage_w
            + eventwise_fscore * self.reliability_w
        )
        sum_of_weights = self.accuracy_w + self.earliness_w + self.coverage_w + self.reliability_w
        return care_score / sum_of_weights

    def calculate_avg_coverage(self, event_selection: list[int] | None = None) -> float:
        events = self._select_events(event_selection)
        mask = events["event_label"] == "anomaly"
        return float(events.loc[mask, "f_beta_score"].mean())

    def calculate_avg_accuracy(self, event_selection: list[int] | None = None) -> float:
        events = self._select_events(event_selection)
        mask = events["event_label"] == "anomaly"
        return float(events.loc[~mask, "accuracy"].mean())

    def calculate_reliability(self, event_selection: list[int] | None = None, **kwargs: Any) -> float:
        events = self._select_events(event_selection)
        return float(
            fbeta_score(
                y_true=events["event_label"] == "anomaly",
                y_pred=events["anomaly_detected"],
                beta=self.reliability_beta,
                **kwargs,
            )
        )

    def calculate_avg_earliness(self, event_selection: list[int] | None = None) -> float:
        events = self._select_events(event_selection)
        mask = events["event_label"] == "anomaly"
        return float(events.loc[mask, "weighted_score"].mean())

    @staticmethod
    def create_ground_truth(
        event_start: int | pd.Timestamp,
        event_end: int | pd.Timestamp,
        normal_index: pd.Series,
        event_label: str,
    ) -> pd.Series:
        ground_truth = pd.Series(data=~normal_index, index=normal_index.index, name="ground_truth")
        ground_truth = ground_truth.sort_index()
        if event_label == "anomaly":
            ground_truth.loc[event_start:event_end] = True
        return ground_truth

    def _calculate_basic_metrics(
        self,
        ground_truth: pd.Series,
        predicted_anomalies: pd.Series,
        event_label: str,
    ) -> dict[str, float | int]:
        f_score = np.nan
        if event_label == "anomaly":
            if all(ground_truth):
                warnings.warn(
                    "Ill-defined anomaly event: ground truth is all True; F-score may be degenerate.",
                    stacklevel=2,
                )
            f_score = float(
                fbeta_score(
                    y_true=ground_truth,
                    y_pred=predicted_anomalies,
                    beta=self.coverage_beta,
                )
            )

        accuracy = float(accuracy_score(y_true=ground_truth, y_pred=predicted_anomalies))
        tn, fp, fn, tp = confusion_matrix(
            y_true=ground_truth, y_pred=predicted_anomalies, labels=[False, True]
        ).ravel()

        return {
            "f_beta_score": f_score,
            "accuracy": accuracy,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    def _calculate_weighted_score(self, event_prediction: pd.Series) -> float:
        event_length = len(event_prediction)
        numerator, denominator = self.ws_start_of_descend
        if denominator <= 0:
            raise ValueError("ws_start_of_descend denominator must be > 0")
        start_of_descend = numerator / denominator
        if not (0 <= start_of_descend < 1):
            raise ValueError("ws_start_of_descend must satisfy 0 <= numerator/denominator < 1")

        scale = int(denominator)
        scaled_event_length = event_length * scale
        cp = int(scaled_event_length * start_of_descend)

        x_values = np.linspace(0, scale, scaled_event_length)
        weights = np.zeros(scaled_event_length)
        weights[:cp] = scale
        slope = 1 / (1 - start_of_descend)
        offset = scale / (1 - start_of_descend)
        weights[cp:scaled_event_length] = offset - slope * x_values[cp:scaled_event_length]

        final_weights = weights[::scale] / scale
        denom = np.sum(final_weights)
        if denom == 0:
            return 0.0
        return float(np.sum(event_prediction.to_numpy(dtype=bool) * final_weights) / denom)

    def save_evaluated_events(self, file_path: Path | str) -> None:
        self.evaluated_events.to_csv(Path(file_path), index=False)

    def load_evaluated_events(self, file_path: Path | str) -> None:
        path = Path(file_path)
        try:
            self._evaluated_events = pd.read_csv(path).to_dict(orient="records")
        except Exception as exc:
            raise ValueError(f"Failed to read evaluated events from {path}: {exc}") from exc

    def _select_events(self, event_ids: list[int] | None = None) -> pd.DataFrame:
        if event_ids is None:
            return self.evaluated_events
        return self.evaluated_events[self.evaluated_events["event_id"].isin(event_ids)]