"""CARE evaluation helpers for hybrid-residual and minimal AE on WF-A."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from wind_digital_twin.anomaly.hybrid_residual_detector import fit_hybrid_residual_detector
from wind_digital_twin.anomaly.scoring import threshold_from_training
from wind_digital_twin.config import (
    DEFAULT_PERSISTENCE_SAMPLES,
    DEFAULT_THRESHOLD_PERCENTILE,
    THERMAL_MIN_POWER_KW,
)
from wind_digital_twin.data.clean import clean_turbine_df
from wind_digital_twin.data.load_care import CareDataset, to_twin_frame
from wind_digital_twin.eval.care_score import CAREScore
from wind_digital_twin.eval.twin_pass import threshold_from_healthy_holdout
from wind_digital_twin.physics.gearbox_thermal import healthy_train_validate_split

CARE_META = frozenset({"id", "time_stamp", "asset_id", "train_test", "status_type_id"})


def is_gearbox_anomaly(description: str) -> bool:
    text = str(description).lower()
    return "gearbox" in text or "gear box" in text


def is_generator_bearing(description: str) -> bool:
    text = str(description).lower()
    return "generator" in text and "bearing" in text


def is_drivetrain_anomaly(description: str) -> bool:
    return is_gearbox_anomaly(description) or is_generator_bearing(description)


def event_in_care_scope(ds: CareDataset, *, gearbox_only: bool) -> bool:
    if ds.event.label == "normal":
        return True
    if not gearbox_only:
        return True
    return is_gearbox_anomaly(ds.event.description)


def raw_to_operating(raw: pd.DataFrame, channel_map: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "time_stamp" not in frame.columns:
        frame = frame.reset_index()
        if "time_stamp" not in frame.columns and "index" in frame.columns:
            frame = frame.rename(columns={"index": "time_stamp"})
    twin = to_twin_frame(frame, channel_map=channel_map, rescale_power=True)
    return clean_turbine_df(twin, min_power_kw=THERMAL_MIN_POWER_KW)


def numeric_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in CARE_META]
    return df[cols].select_dtypes(include=[np.number]).fillna(0.0)


def scores_to_persistent_alarms(
    scores: pd.Series,
    threshold: float,
    persistence_samples: int = DEFAULT_PERSISTENCE_SAMPLES,
) -> pd.Series:
    """Pointwise True for samples inside a sustained above-threshold run."""
    out = pd.Series(False, index=scores.index, dtype=bool)
    if scores.empty or persistence_samples < 1:
        return out
    above = scores > threshold
    n = len(scores)
    i = 0
    while i <= n - persistence_samples:
        if above.iloc[i : i + persistence_samples].all():
            j = i + persistence_samples
            while j < n and above.iloc[j]:
                j += 1
            out.iloc[i:j] = True
            i = j
        else:
            i += 1
    return out


def _reindex_scores_to_prediction(
    scores: pd.Series,
    pred_idx: pd.DatetimeIndex,
    threshold: float,
    *,
    persistence_samples: int = 0,
) -> pd.Series:
    aligned = scores.reindex(pred_idx)
    if persistence_samples > 0:
        valid = aligned.dropna()
        mask = scores_to_persistent_alarms(valid, threshold, persistence_samples)
        alarms = pd.Series(False, index=pred_idx)
        if not mask.empty:
            alarms.loc[mask.index] = mask.astype(bool)
    else:
        alarms = aligned.gt(threshold).fillna(False)
    return alarms.astype(bool)


class MinimalMultivariateAE:
    """Sklearn MLP autoencoder surrogate (86-dim avg numeric features)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: MLPRegressor | None = None
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame, *, max_rows: int = 8000) -> None:
        if len(X) > max_rows:
            X = X.sample(max_rows, random_state=self.random_state)
        self.feature_columns = X.columns.tolist()
        Xs = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 16, 64),
            activation="relu",
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=self.random_state,
        )
        self.model.fit(Xs, Xs)

    def reconstruction_error(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model not fit")
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        Xs = self.scaler.transform(X)
        pred = self.model.predict(Xs)
        err = np.mean((Xs - pred) ** 2, axis=1)
        return pd.Series(err, index=X.index, name="recon_error")


def predict_minimal_ae_alarms(
    ds: CareDataset,
    *,
    percentile: float = 99.0,
    persistence_samples: int = 0,
    seed: int = 42,
) -> pd.Series:
    pred_idx = pd.to_datetime(ds.prediction["time_stamp"], utc=True)
    train = ds.train.loc[ds.train_normal_mask]
    x_train = numeric_feature_matrix(train)
    x_train.index = pd.to_datetime(train["time_stamp"], utc=True)
    x_pred = numeric_feature_matrix(ds.prediction)
    if len(x_train) < 200 or x_train.shape[1] < 10:
        return pd.Series(False, index=pred_idx)

    ae = MinimalMultivariateAE(random_state=seed)
    ae.fit(x_train)
    _tr, val = healthy_train_validate_split(x_train)
    if val.empty:
        val = x_train.iloc[max(0, len(x_train) - 500) :]
    holdout_err = ae.reconstruction_error(val)
    threshold = threshold_from_training(holdout_err, percentile=percentile)
    pred_err = ae.reconstruction_error(x_pred)
    scores = pd.Series(pred_err.values, index=pred_idx)
    return _reindex_scores_to_prediction(scores, pred_idx, threshold, persistence_samples=persistence_samples)


def predict_hybrid_alarms(
    ds: CareDataset,
    channel_map: pd.DataFrame,
    *,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    persistence_samples: int = 0,
) -> pd.Series:
    pred_idx = pd.to_datetime(ds.prediction["time_stamp"], utc=True)
    train_normal = ds.train.loc[ds.train_normal_mask]
    train_df = raw_to_operating(train_normal, channel_map)
    if len(train_df) < 100:
        return pd.Series(False, index=pred_idx)

    pipeline = fit_hybrid_residual_detector(train_df)
    threshold, _ = threshold_from_healthy_holdout(pipeline, train_df, percentile=percentile)
    pred_df = raw_to_operating(ds.prediction, channel_map)
    scores = pipeline.score(pred_df)
    if scores.empty:
        return pd.Series(False, index=pred_idx)
    return _reindex_scores_to_prediction(
        scores, pred_idx, threshold, persistence_samples=persistence_samples
    )


@dataclass
class CareStrategy:
    name: str
    description: str
    gearbox_only: bool = False
    predict: Callable[[CareDataset], pd.Series] = lambda _ds: pd.Series(dtype=bool)  # noqa: E731


def run_care_strategy(
    events: list[CareDataset],
    strategy: CareStrategy,
    predict_fn: Callable[[CareDataset], pd.Series],
) -> dict:
    scorer = CAREScore()
    per_event: list[dict] = []

    for ds in events:
        if not event_in_care_scope(ds, gearbox_only=strategy.gearbox_only):
            continue
        try:
            predictions = predict_fn(ds)
        except Exception as exc:
            per_event.append(
                {
                    "event_id": ds.event.event_id,
                    "label": ds.event.label,
                    "error": str(exc),
                }
            )
            continue

        scorer.evaluate_event(
            event_start=ds.event.event_start,
            event_end=ds.event.event_end,
            event_label=ds.event.label,
            predicted_anomalies=predictions,
            normal_index=None,
            ignore_normal_index=True,
            event_id=ds.event.event_id,
        )
        per_event.append(
            {
                "event_id": ds.event.event_id,
                "label": ds.event.label,
                "description": ds.event.description,
                "n_alarms": int(predictions.sum()),
                "in_scope": True,
            }
        )

    try:
        summary = {
            "name": strategy.name,
            "description": strategy.description,
            "care": scorer.get_final_score(),
            "coverage": scorer.calculate_avg_coverage(),
            "accuracy": scorer.calculate_avg_accuracy(),
            "reliability": scorer.calculate_reliability(),
            "earliness": scorer.calculate_avg_earliness(),
        }
    except ValueError as exc:
        summary = {
            "name": strategy.name,
            "description": strategy.description,
            "error": str(exc),
            "care": None,
        }

    return {"summary": summary, "events": per_event}