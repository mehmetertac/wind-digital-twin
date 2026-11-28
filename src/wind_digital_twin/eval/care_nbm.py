"""Normal-behaviour scores on CARE Wind Farm A, then identity-disjoint alarms.

The lumped thermal model and the learned residual see only status-normal
training rows (CARE status ids 0 and 2). Prediction rows never enter the fit.
Dense and LSTM autoencoders, and this repo's Isolation Forest, use the same
row split on the numeric SCADA matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from wind_digital_twin.anomaly.hybrid_residual_detector import fit_hybrid_residual_detector
from wind_digital_twin.anomaly.isolation_forest import fit_isolation_forest
from wind_digital_twin.config import CARE_NORMAL_STATUS_IDS
from wind_digital_twin.data.load_care import CareDataset
from wind_digital_twin.eval.care_hybrid_eval import (
    MinimalMultivariateAE,
    event_in_care_scope,
    numeric_feature_matrix,
    raw_to_operating,
)
from wind_digital_twin.eval.care_score import CAREScore
from wind_digital_twin.eval.identity_calibration import (
    DEFAULT_ALERT_ALPHA,
    chronological_fit_cal_split,
    fit_location_scale,
    threshold_excluding_assets,
    to_z,
)

MODEL_NAMES = ("hybrid", "isolation_forest", "dense_ae", "lstm_ae")


@dataclass
class CachedEvent:
    """Standardized scores for one CARE sub-dataset."""

    event_id: int
    asset_id: str
    label: str
    description: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    pred_index: pd.DatetimeIndex
    z_pred: dict[str, np.ndarray] = field(default_factory=dict)
    z_cal: dict[str, np.ndarray] = field(default_factory=dict)
    cal_index: dict[str, pd.DatetimeIndex] = field(default_factory=dict)
    pred_normal: np.ndarray | None = None
    errors: dict[str, str] = field(default_factory=dict)
    channel_pred: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    channel_cal: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def _sorted_normal_train(ds: CareDataset) -> pd.DataFrame:
    normal = ds.train.loc[ds.train["status_type_id"].isin(CARE_NORMAL_STATUS_IDS)].copy()
    if "time_stamp" not in normal.columns or normal.empty:
        return normal
    normal = normal.sort_values("time_stamp")
    return normal


def _prediction_index(ds: CareDataset) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ds.prediction["time_stamp"], utc=True))


def attach_prediction_normal(event: CachedEvent, prediction: pd.DataFrame) -> None:
    """CARE status ids 0 and 2, aligned to the prediction index.

    Rows that are not status-normal hold the criticality counter when scoring
    passes this mask as ``normal_index``.
    """
    if prediction.empty or "time_stamp" not in prediction.columns or "status_type_id" not in prediction.columns:
        event.pred_normal = np.ones(len(event.pred_index), dtype=bool)
        return
    times = pd.DatetimeIndex(pd.to_datetime(prediction["time_stamp"], utc=True))
    flags = prediction["status_type_id"].isin(CARE_NORMAL_STATUS_IDS).to_numpy(dtype=bool)
    series = pd.Series(flags, index=times)
    series = series[~series.index.duplicated(keep="last")]
    event.pred_normal = series.reindex(event.pred_index).fillna(False).to_numpy(dtype=bool)


def _align_z(
    scores: pd.Series,
    center: float,
    scale: float,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    if scores.empty:
        return np.full(len(index), np.nan)
    z = pd.Series(to_z(scores.to_numpy(), center, scale), index=scores.index)
    aligned = z.reindex(index)
    return aligned.to_numpy(dtype=float)


def _store_model(
    event: CachedEvent,
    name: str,
    fit_scores: pd.Series,
    cal_scores: pd.Series,
    pred_scores: pd.Series,
) -> None:
    center, scale = fit_location_scale(fit_scores.to_numpy())
    event.z_cal[name] = to_z(cal_scores.to_numpy(), center, scale) if len(cal_scores) else np.array([])
    event.z_pred[name] = _align_z(pred_scores, center, scale, event.pred_index)


def _score_hybrid(
    event: CachedEvent,
    fit_raw: pd.DataFrame,
    cal_raw: pd.DataFrame,
    pred_raw: pd.DataFrame,
    channel_map: pd.DataFrame,
) -> None:
    fit_df = raw_to_operating(fit_raw, channel_map)
    cal_df = raw_to_operating(cal_raw, channel_map) if not cal_raw.empty else fit_df.iloc[0:0]
    pred_df = raw_to_operating(pred_raw, channel_map)
    if len(fit_df) < 100:
        raise ValueError(f"hybrid fit block has {len(fit_df)} operating rows")
    pipeline = fit_hybrid_residual_detector(fit_df)
    fit_scores = pipeline.score(fit_df)
    cal_scores = pipeline.score(cal_df) if not cal_df.empty else pd.Series(dtype=float)
    pred_scores = pipeline.score(pred_df) if not pred_df.empty else pd.Series(dtype=float)
    if fit_scores.empty:
        raise ValueError("hybrid produced no fit-block scores")
    _store_model(event, "hybrid", fit_scores, cal_scores, pred_scores)


def _with_time_index(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "time_stamp" not in raw.columns:
        return raw
    out = raw.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out["time_stamp"], utc=True))
    return out


def _score_matrix_model(
    event: CachedEvent,
    name: str,
    fit_raw: pd.DataFrame,
    cal_raw: pd.DataFrame,
    pred_raw: pd.DataFrame,
    scorer,
) -> None:
    fit_raw = _with_time_index(fit_raw)
    cal_raw = _with_time_index(cal_raw)
    pred_raw = _with_time_index(pred_raw)
    x_fit = numeric_feature_matrix(fit_raw)
    x_cal = numeric_feature_matrix(cal_raw) if not cal_raw.empty else x_fit.iloc[0:0]
    x_pred = numeric_feature_matrix(pred_raw)
    fit_scores, cal_scores, pred_scores = scorer(x_fit, x_cal, x_pred)
    _store_model(event, name, fit_scores, cal_scores, pred_scores)


def _isolation_forest_scores(x_fit: pd.DataFrame, x_cal: pd.DataFrame, x_pred: pd.DataFrame):
    if len(x_fit) < 50 or x_fit.shape[1] < 2:
        raise ValueError("isolation forest fit block is too small")
    pipeline = fit_isolation_forest(
        x_fit,
        x_fit.columns.tolist(),
        contamination=0.01,
        n_estimators=100,
        random_state=42,
    )
    cal_scores = pipeline.score(x_cal) if len(x_cal) else pd.Series(dtype=float)
    return pipeline.score(x_fit), cal_scores, pipeline.score(x_pred)


def _dense_ae_scores(x_fit: pd.DataFrame, x_cal: pd.DataFrame, x_pred: pd.DataFrame, seed: int):
    if len(x_fit) < 200 or x_fit.shape[1] < 10:
        raise ValueError("dense autoencoder fit block is too small")
    model = MinimalMultivariateAE(random_state=seed)
    model.fit(x_fit)
    cal_err = model.reconstruction_error(x_cal) if len(x_cal) else pd.Series(dtype=float)
    return model.reconstruction_error(x_fit), cal_err, model.reconstruction_error(x_pred)


def _window_errors(reconstructed: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.mean((reconstructed - original) ** 2, axis=(1, 2))


def _lstm_ae_scores(
    x_fit: pd.DataFrame,
    x_cal: pd.DataFrame,
    x_pred: pd.DataFrame,
    *,
    seed: int,
    seq_len: int = 6,
    hidden: int = 16,
    epochs: int = 4,
    max_windows: int = 2000,
):
    import torch
    from torch import nn

    if len(x_fit) < seq_len + 30 or x_fit.shape[1] < 2:
        raise ValueError("LSTM autoencoder fit block is too small")

    class _LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.proj = nn.Linear(hidden, n_features)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            _, (state, _) = self.encoder(batch)
            repeated = state[-1].unsqueeze(1).expand(-1, batch.size(1), -1).contiguous()
            decoded, _ = self.decoder(repeated)
            return self.proj(decoded)

    scaler_mean = x_fit.mean().to_numpy(dtype=np.float64)
    scaler_std = x_fit.std().replace(0, 1).to_numpy(dtype=np.float64)

    def _scale(frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=x_fit.columns, fill_value=0.0).to_numpy(dtype=np.float64)
        return (values - scaler_mean) / scaler_std

    def _windows(values: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
        n = len(values) - seq_len + 1
        if n <= 0:
            return np.empty((0, seq_len, values.shape[1])), np.array([], dtype=int)
        end = np.arange(seq_len - 1, len(values))
        if limit is not None and n > limit:
            choose = np.linspace(0, n - 1, limit).astype(int)
            end = end[choose]
        batch = np.stack([values[i - seq_len + 1 : i + 1] for i in end])
        return batch, end

    torch.manual_seed(seed)
    fit_x = _scale(x_fit)
    train_w, _ = _windows(fit_x, max_windows)
    if len(train_w) < 32:
        raise ValueError("LSTM autoencoder has too few windows")
    device = torch.device("cpu")
    net = _LSTMAutoencoder(fit_x.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(train_w, dtype=torch.float32, device=device)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(tensor)
        loss = loss_fn(pred, tensor)
        loss.backward()
        opt.step()

    def _error_series(frame: pd.DataFrame, limit: int | None) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=float)
        values = _scale(frame)
        windows, ends = _windows(values, limit)
        if len(windows) == 0:
            return pd.Series(dtype=float)
        net.eval()
        pieces: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(windows), 512):
                batch = torch.tensor(windows[start : start + 512], dtype=torch.float32, device=device)
                pieces.append(net(batch).cpu().numpy())
        recon = np.concatenate(pieces, axis=0)
        err = _window_errors(recon, windows)
        return pd.Series(err, index=frame.index[ends])

    return (
        _error_series(x_fit, max_windows),
        _error_series(x_cal, None),
        _error_series(x_pred, None),
    )


def score_care_event(
    ds: CareDataset,
    channel_map: pd.DataFrame,
    *,
    seed: int = 42,
    models: tuple[str, ...] = MODEL_NAMES,
) -> CachedEvent:
    """Fit each requested model on the status-normal fit block and store z-scores."""
    normal = _sorted_normal_train(ds)
    fit_raw, cal_raw = chronological_fit_cal_split(normal)
    pred_raw = ds.prediction.sort_values("time_stamp") if "time_stamp" in ds.prediction.columns else ds.prediction
    event = CachedEvent(
        event_id=ds.event.event_id,
        asset_id=str(ds.event.asset_id),
        label=ds.event.label,
        description=ds.event.description,
        event_start=ds.event.event_start,
        event_end=ds.event.event_end,
        pred_index=_prediction_index(ds),
    )
    if "hybrid" in models:
        try:
            _score_hybrid(event, fit_raw, cal_raw, pred_raw, channel_map)
        except Exception as exc:
            event.errors["hybrid"] = str(exc)
            event.z_pred["hybrid"] = np.full(len(event.pred_index), np.nan)
            event.z_cal["hybrid"] = np.array([])
    if "isolation_forest" in models:
        try:
            _score_matrix_model(
                event,
                "isolation_forest",
                fit_raw,
                cal_raw,
                pred_raw,
                _isolation_forest_scores,
            )
        except Exception as exc:
            event.errors["isolation_forest"] = str(exc)
            event.z_pred["isolation_forest"] = np.full(len(event.pred_index), np.nan)
            event.z_cal["isolation_forest"] = np.array([])
    if "dense_ae" in models:
        try:
            _score_matrix_model(
                event,
                "dense_ae",
                fit_raw,
                cal_raw,
                pred_raw,
                lambda a, b, c: _dense_ae_scores(a, b, c, seed),
            )
        except Exception as exc:
            event.errors["dense_ae"] = str(exc)
            event.z_pred["dense_ae"] = np.full(len(event.pred_index), np.nan)
            event.z_cal["dense_ae"] = np.array([])
    if "lstm_ae" in models:
        try:
            _score_matrix_model(
                event,
                "lstm_ae",
                fit_raw,
                cal_raw,
                pred_raw,
                lambda a, b, c: _lstm_ae_scores(a, b, c, seed=seed),
            )
        except Exception as exc:
            event.errors["lstm_ae"] = str(exc)
            event.z_pred["lstm_ae"] = np.full(len(event.pred_index), np.nan)
            event.z_cal["lstm_ae"] = np.array([])
    rise_names = tuple(name for name in ("hybrid_rise", "hybrid_multi") if name in models)
    if rise_names:
        from wind_digital_twin.eval.care_rise_score import score_rise_into_event

        score_rise_into_event(event, ds, channel_map, rise_names)
    attach_prediction_normal(event, ds.prediction)
    return event


def calibration_by_asset(events: list[CachedEvent], model: str) -> dict[str, np.ndarray]:
    """One calibration vector per asset: the longest finite vector wins."""
    best: dict[str, np.ndarray] = {}
    for event in events:
        values = np.asarray(event.z_cal.get(model, []), dtype=float)
        values = values[np.isfinite(values)]
        current = best.get(event.asset_id)
        if current is None or values.size > current.size:
            best[event.asset_id] = values
    return best


def _smooth(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()


def _prediction_z(event: CachedEvent, model: str, ewma_span: int | None) -> pd.Series:
    z = np.asarray(event.z_pred.get(model, []), dtype=float)
    if z.size != len(event.pred_index):
        z = np.full(len(event.pred_index), np.nan)
    series = pd.Series(z, index=event.pred_index)
    if not ewma_span:
        return series
    ordered = series.sort_index().ewm(span=ewma_span, adjust=False, ignore_na=True).mean()
    return ordered.reindex(event.pred_index)


def _alarms(event: CachedEvent, model: str, threshold: float, ewma_span: int | None = None) -> pd.Series:
    z = _prediction_z(event, model, ewma_span)
    return (z.notna() & (z > threshold)).astype(bool)


def care_from_cached(
    events: list[CachedEvent],
    model: str,
    *,
    alpha: float = DEFAULT_ALERT_ALPHA,
    mode: str = "loto",
    locked_assets: frozenset[str] = frozenset(),
    gearbox_only: bool = False,
    use_normal_index: bool = False,
    ewma_span: int | None = None,
    include_frame: bool = False,
) -> dict:
    """Score cached z-values with a threshold that excludes the test asset.

    ``loto`` drops only the asset being scored. ``locked`` scores
    ``locked_assets`` with one threshold built from every other asset.
    """
    if mode not in {"loto", "locked"}:
        raise ValueError("mode must be 'loto' or 'locked'")
    pool = calibration_by_asset(events, model)
    if ewma_span:
        pool = {asset: _smooth(values, ewma_span) for asset, values in pool.items()}
    scorer = CAREScore()
    per_event: list[dict] = []
    locked_threshold = None
    if mode == "locked":
        locked_threshold = threshold_excluding_assets(pool, set(locked_assets), alpha)

    for event in events:
        dummy = type("DS", (), {})()
        dummy.event = type(
            "E",
            (),
            {"label": event.label, "description": event.description},
        )()
        if not event_in_care_scope(dummy, gearbox_only=gearbox_only):
            continue
        if mode == "locked" and event.asset_id not in locked_assets:
            continue
        if mode == "loto":
            threshold = threshold_excluding_assets(pool, {event.asset_id}, alpha)
        else:
            threshold = float(locked_threshold)
        alarms = _alarms(event, model, threshold, ewma_span)
        normal_index = None
        ignore_normal = True
        if (
            use_normal_index
            and event.pred_normal is not None
            and len(event.pred_normal) == len(event.pred_index)
        ):
            normal_index = pd.Series(np.asarray(event.pred_normal, dtype=bool), index=event.pred_index)
            ignore_normal = False
        try:
            scorer.evaluate_event(
                event_start=event.event_start,
                event_end=event.event_end,
                event_label=event.label,
                predicted_anomalies=alarms,
                normal_index=normal_index,
                ignore_normal_index=ignore_normal,
                event_id=event.event_id,
            )
        except ValueError as exc:
            per_event.append({"event_id": event.event_id, "error": str(exc)})
            continue
        per_event.append(
            {
                "event_id": event.event_id,
                "asset_id": event.asset_id,
                "label": event.label,
                "description": event.description,
                "n_alarms": int(alarms.sum()),
                "threshold_z": threshold,
            }
        )

    try:
        summary = {
            "model": model,
            "mode": mode,
            "alpha": alpha,
            "gearbox_only": gearbox_only,
            "use_normal_index": use_normal_index,
            "ewma_span": ewma_span,
            "care": scorer.get_final_score(),
            "coverage": scorer.calculate_avg_coverage(),
            "accuracy": scorer.calculate_avg_accuracy(),
            "reliability": scorer.calculate_reliability(),
            "earliness": scorer.calculate_avg_earliness(),
            "n_events": len(per_event),
        }
    except ValueError as exc:
        summary = {
            "model": model,
            "mode": mode,
            "alpha": alpha,
            "gearbox_only": gearbox_only,
            "use_normal_index": use_normal_index,
            "ewma_span": ewma_span,
            "error": str(exc),
            "care": None,
            "n_events": len(per_event),
        }
    payload = {"summary": summary, "events": per_event}
    if include_frame:
        payload["frame"] = scorer.evaluated_events
    return payload


def _care_from_frame(frame: pd.DataFrame) -> float:
    """CARE on a resampled event table. Same weights as ``CAREScore.get_final_score``."""
    spec = CAREScore()
    if frame.empty or frame["event_label"].nunique() < 2:
        return float("nan")
    detected = frame["max_criticality"] >= spec.criticality_threshold
    if not bool(detected.any()):
        return 0.0
    is_anomaly = frame["event_label"].to_numpy() == "anomaly"
    avg_accuracy = float(frame.loc[~is_anomaly, "accuracy"].mean())
    if avg_accuracy <= 0.5:
        return avg_accuracy
    avg_coverage = float(frame.loc[is_anomaly, "f_beta_score"].mean())
    avg_earliness = float(frame.loc[is_anomaly, "weighted_score"].mean())
    reliability = float(
        fbeta_score(
            y_true=is_anomaly,
            y_pred=detected.to_numpy(dtype=bool),
            beta=spec.reliability_beta,
            zero_division=0,
        )
    )
    numerator = (
        avg_accuracy * spec.accuracy_w
        + avg_earliness * spec.earliness_w
        + avg_coverage * spec.coverage_w
        + reliability * spec.reliability_w
    )
    denom = spec.accuracy_w + spec.earliness_w + spec.coverage_w + spec.reliability_w
    return float(numerator / denom)


def _quantile_ci(draws: list[float], level: float) -> list[float | None]:
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return [float(lo), float(hi)]


def _bootstrap_frame(frame: pd.DataFrame, *, n: int, seed: int, level: float) -> list[float | None]:
    if frame.empty:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = [
        _care_from_frame(frame.iloc[rng.integers(0, len(frame), size=len(frame))])
        for _ in range(n)
    ]
    return _quantile_ci(draws, level)


def bootstrap_care(
    events: list[CachedEvent],
    model: str,
    *,
    n: int = 200,
    seed: int = 0,
    level: float = 0.9,
    **kwargs,
) -> dict:
    """Percentile interval for CARE by resampling the scored events with replacement."""
    kwargs.pop("include_frame", None)
    scored = care_from_cached(events, model, include_frame=True, **kwargs)
    frame = scored["frame"]
    return {
        "n": n,
        "seed": seed,
        "level": level,
        "care": scored["summary"].get("care"),
        "ci90": _bootstrap_frame(frame, n=n, seed=seed, level=level),
    }


def bootstrap_interval(frame: pd.DataFrame, *, n: int = 200, seed: int = 0, level: float = 0.9) -> list[float | None]:
    """90% percentile interval for a table of already scored events."""
    return _bootstrap_frame(frame, n=n, seed=seed, level=level)


def bootstrap_care_difference(
    events: list[CachedEvent],
    left: str,
    right: str,
    *,
    n: int = 200,
    seed: int = 0,
    level: float = 0.9,
    **kwargs,
) -> dict:
    """Paired interval for ``CARE(left) - CARE(right)`` on the same event resamples."""
    kwargs.pop("include_frame", None)
    left_frame = care_from_cached(events, left, include_frame=True, **kwargs)["frame"]
    right_frame = care_from_cached(events, right, include_frame=True, **kwargs)["frame"]
    left_rows = left_frame.drop_duplicates("event_id").set_index("event_id")
    right_rows = right_frame.drop_duplicates("event_id").set_index("event_id")
    common = left_rows.index.intersection(right_rows.index)
    left_rows = left_rows.loc[common]
    right_rows = right_rows.loc[common]
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    size = len(common)
    if size:
        for _ in range(n):
            pick = rng.integers(0, size, size=size)
            diffs.append(
                _care_from_frame(left_rows.iloc[pick]) - _care_from_frame(right_rows.iloc[pick])
            )
    return {
        "left": left,
        "right": right,
        "n": n,
        "seed": seed,
        "level": level,
        "ci90": _quantile_ci(diffs, level),
    }


def care_from_alarm_union(
    events: list[CachedEvent],
    models: tuple[str, ...],
    *,
    alpha: float = DEFAULT_ALERT_ALPHA,
    gearbox_only: bool = False,
    ewma_span: int | None = None,
    include_frame: bool = False,
) -> dict:
    """Alarm when any member model crosses its own identity-disjoint threshold.

    Max-of-z fusion misses this case: the residual z and the autoencoder z
    are not on a common scale, so the quieter score never wins the max.
    """
    pools = {model: calibration_by_asset(events, model) for model in models}
    if ewma_span:
        pools = {
            model: {asset: _smooth(values, ewma_span) for asset, values in pool.items()}
            for model, pool in pools.items()
        }
    scorer = CAREScore()
    per_event: list[dict] = []
    for event in events:
        dummy = type("DS", (), {})()
        dummy.event = type("E", (), {"label": event.label, "description": event.description})()
        if not event_in_care_scope(dummy, gearbox_only=gearbox_only):
            continue
        alarms = pd.Series(False, index=event.pred_index)
        for model in models:
            threshold = threshold_excluding_assets(pools[model], {event.asset_id}, alpha)
            alarms = alarms | _alarms(event, model, threshold, ewma_span)
        try:
            scorer.evaluate_event(
                event_start=event.event_start,
                event_end=event.event_end,
                event_label=event.label,
                predicted_anomalies=alarms,
                normal_index=None,
                ignore_normal_index=True,
                event_id=event.event_id,
            )
        except ValueError as exc:
            per_event.append({"event_id": event.event_id, "error": str(exc)})
            continue
        per_event.append(
            {
                "event_id": event.event_id,
                "asset_id": event.asset_id,
                "label": event.label,
                "n_alarms": int(alarms.sum()),
            }
        )
    name = " or ".join(models)
    try:
        summary = {
            "model": name,
            "mode": "loto",
            "alpha": alpha,
            "gearbox_only": gearbox_only,
            "use_normal_index": False,
            "ewma_span": ewma_span,
            "care": scorer.get_final_score(),
            "coverage": scorer.calculate_avg_coverage(),
            "accuracy": scorer.calculate_avg_accuracy(),
            "reliability": scorer.calculate_reliability(),
            "earliness": scorer.calculate_avg_earliness(),
            "n_events": len(per_event),
        }
    except ValueError as exc:
        summary = {
            "model": name,
            "mode": "loto",
            "alpha": alpha,
            "gearbox_only": gearbox_only,
            "error": str(exc),
            "care": None,
            "n_events": len(per_event),
        }
    payload = {"summary": summary, "events": per_event}
    if include_frame:
        payload["frame"] = scorer.evaluated_events
    return payload
