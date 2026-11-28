"""CARE scores from nacelle-referenced temperature rise, not the gearbox IF.

The lumped ODE forces a nacelle gain of 1 and the detector then watches only
oil and the high-speed-shaft bearing. These scores fit ``T - T_nac`` on a
power spline, rotor speed, pitch, and nacelle temperature over the whole
earlier normal block, then alarm on the one-sided residual. The nacelle term
is required: on Wind Farm A the rise itself is larger when the nacelle is
cooler, and leaving it out leaves a multi-degree bias on a later cool window.
The power spline is required because a straight power term leaves a high-load
bias that alarms hydraulic oil on both a fault and a healthy window. Extra
targets cover the labelled subsystems: generator bearings, stator, hydraulic
oil, transformer, and converter. Channels are combined as a Bonferroni min-p,
not a max of z-scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from wind_digital_twin.anomaly.residual_z import one_sided_channel_scores, one_sided_max_score
from wind_digital_twin.config import (
    CARE_NORMAL_STATUS_IDS,
    CARE_WFA_RATED_KW,
    POWER_COLUMN,
    RESIDUAL_EWMA_SPAN,
    THERMAL_MIN_POWER_KW,
    THERMAL_TARGET_COLUMNS,
)
from wind_digital_twin.data.load_care import CareDataset, to_twin_frame
from wind_digital_twin.eval.identity_calibration import (
    chronological_fit_cal_split,
    chronological_fit_cal_split_uncapped,
    conformal_pvalues,
)

GEARBOX_TARGETS: tuple[str, ...] = tuple(THERMAL_TARGET_COLUMNS)
SUBSYSTEM_TARGETS: tuple[str, ...] = (
    *GEARBOX_TARGETS,
    "Gen_Bear1_Temp_Avg",
    "Gen_Bear2_Temp_Avg",
    "Gen_Stator1_Temp_Avg",
    "Gen_Stator2_Temp_Avg",
    "Gen_Stator3_Temp_Avg",
    "Hyd_Oil_Temp_Avg",
    "HV_Trafo_L1_Temp_Avg",
    "HV_Trafo_L2_Temp_Avg",
    "HV_Trafo_L3_Temp_Avg",
    "IGBT_Grid_Temp_Avg",
    "IGBT_Rotor1_Temp_Avg",
    "IGBT_Rotor2_Temp_Avg",
    "IGBT_Rotor3_Temp_Avg",
)
RISE_MODEL_NAMES: tuple[str, ...] = ("hybrid_rise", "hybrid_multi")
FUSED_MODEL_NAME = "hybrid_ae"
NAC_COLUMN = "Nac_Temp_Avg"
RPM_COLUMN = "Rtr_RPM_Avg"
PITCH_COLUMN = "Blds_PitchAngle_Avg"
POWER_SPLINE_KNOTS = 5
_MIN_FIT_ROWS = 100
_BONFERRONI_CAL_ROWS = 32


class _RiseModel:
    def __init__(self, pipeline: Pipeline, columns: list[str], target: str) -> None:
        self.pipeline = pipeline
        self.columns = columns
        self.target = target


def _status_normal_train(ds: CareDataset) -> pd.DataFrame:
    normal = ds.train.loc[ds.train["status_type_id"].isin(CARE_NORMAL_STATUS_IDS)].copy()
    if "time_stamp" in normal.columns and not normal.empty:
        normal = normal.sort_values("time_stamp")
    return normal


def matrix_calibration_index(ds: CareDataset) -> pd.DatetimeIndex:
    """Row order of the capped calibration block used by the autoencoder cache."""
    _, cal = chronological_fit_cal_split(_status_normal_train(ds))
    if cal.empty or "time_stamp" not in cal.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(cal["time_stamp"], utc=True))


def operating_frame(
    raw: pd.DataFrame,
    channel_map: pd.DataFrame,
    targets: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Operating rows with drivers and whichever temperature targets are present."""
    if raw.empty:
        return pd.DataFrame()
    twin = to_twin_frame(raw, channel_map=channel_map, rescale_power=True)
    twin = twin[~twin.index.duplicated(keep="last")].sort_index()
    drivers = [POWER_COLUMN, RPM_COLUMN, NAC_COLUMN]
    if PITCH_COLUMN in twin.columns:
        drivers.append(PITCH_COLUMN)
    present_targets = [column for column in targets if column in twin.columns]
    columns = drivers + present_targets
    frame = twin[columns]
    frame = frame[frame[POWER_COLUMN] > THERMAL_MIN_POWER_KW]
    frame = frame.dropna(subset=drivers)
    return frame


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [POWER_COLUMN, RPM_COLUMN, NAC_COLUMN]
    if PITCH_COLUMN in frame.columns and frame[PITCH_COLUMN].notna().mean() >= 0.9:
        columns.append(PITCH_COLUMN)
    return columns


def _rise_pipeline(columns: list[str]) -> Pipeline:
    """Power spline plus scaled rpm, pitch, and nacelle. Five knots, not tuned per event."""
    other = [column for column in columns if column != POWER_COLUMN]
    spline = SplineTransformer(
        n_knots=POWER_SPLINE_KNOTS,
        degree=3,
        include_bias=False,
        extrapolation="linear",
    )
    transformers: list[tuple] = [("power", spline, [POWER_COLUMN])]
    if other:
        transformers.append(("level", StandardScaler(), other))
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def fit_rise(frame: pd.DataFrame, target: str) -> _RiseModel | None:
    """Ridge of ``T - T_nac`` on a power spline, rpm, nacelle temperature, and pitch."""
    if target not in frame.columns or NAC_COLUMN not in frame.columns or POWER_COLUMN not in frame.columns:
        return None
    columns = _feature_columns(frame)
    rise = frame[target] - frame[NAC_COLUMN]
    mask = rise.notna() & frame[columns].notna().all(axis=1)
    if int(mask.sum()) < _MIN_FIT_ROWS:
        return None
    pipeline = _rise_pipeline(columns)
    pipeline.fit(frame.loc[mask, columns], rise.loc[mask].to_numpy(dtype=float))
    return _RiseModel(pipeline, columns, target)


def degradation(frame: pd.DataFrame, model: _RiseModel) -> pd.Series:
    """Actual rise minus predicted rise. Positive means hotter than expected."""
    if frame.empty or model.target not in frame.columns:
        return pd.Series(dtype=float, name=model.target)
    mask = frame[model.columns + [model.target, NAC_COLUMN]].notna().all(axis=1)
    if not mask.any():
        return pd.Series(dtype=float, name=model.target)
    sub = frame.loc[mask]
    predicted = model.pipeline.predict(sub[model.columns])
    actual = (sub[model.target] - sub[NAC_COLUMN]).to_numpy(dtype=float)
    return pd.Series(actual - predicted, index=sub.index, name=model.target)


def mean_residual_by_power_bin(
    frame: pd.DataFrame,
    model: _RiseModel,
    *,
    rated_kw: float = CARE_WFA_RATED_KW,
    width: float = 0.2,
) -> dict[str, float]:
    """Mean fit residual (°C) in each ``width`` per-unit power bin."""
    resid = degradation(frame, model)
    if resid.empty:
        return {}
    pu = frame.loc[resid.index, POWER_COLUMN].to_numpy(dtype=float) / rated_kw
    edges = np.floor(pu / width) * width
    out: dict[str, float] = {}
    for edge in np.unique(edges[np.isfinite(edges)]):
        chosen = edges == edge
        out[f"{float(edge):.1f}"] = float(np.mean(resid.to_numpy(dtype=float)[chosen]))
    return out


def _fit_models(frame: pd.DataFrame, targets: tuple[str, ...]) -> dict[str, _RiseModel]:
    models: dict[str, _RiseModel] = {}
    for target in targets:
        model = fit_rise(frame, target)
        if model is not None:
            models[target] = model
    return models


def _channels(frame: pd.DataFrame, models: dict[str, _RiseModel]) -> dict[str, pd.Series]:
    return {target: degradation(frame, model) for target, model in models.items()}


def score_rise_blocks(
    ds: CareDataset,
    channel_map: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    span: int = RESIDUAL_EWMA_SPAN,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fit-block, calibration-block, and prediction scores for one target set."""
    normal = _status_normal_train(ds)
    fit_raw, cal_raw = chronological_fit_cal_split_uncapped(normal)
    pred_raw = ds.prediction.sort_values("time_stamp") if "time_stamp" in ds.prediction.columns else ds.prediction
    fit_df = operating_frame(fit_raw, channel_map, targets)
    cal_df = operating_frame(cal_raw, channel_map, targets)
    pred_df = operating_frame(pred_raw, channel_map, targets)
    models = _fit_models(fit_df, targets)
    if not models:
        raise ValueError(f"no rise model fitted from {len(fit_df)} operating rows")
    fit_channels = _channels(fit_df, models)
    fit_scores = one_sided_max_score(fit_channels, fit_channels, span=span)
    cal_scores = (
        one_sided_max_score(fit_channels, _channels(cal_df, models), span=span)
        if not cal_df.empty
        else pd.Series(dtype=float)
    )
    pred_scores = one_sided_max_score(fit_channels, _channels(pred_df, models), span=span)
    if fit_scores.empty:
        raise ValueError("rise score produced no fit-block values")
    return fit_scores, cal_scores, pred_scores


def gearbox_fit_bin_bias_c(
    ds: CareDataset,
    channel_map: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Fit-block mean residual (°C) by 0.2 pu power bin, for the two gearbox targets."""
    normal = _status_normal_train(ds)
    fit_raw, _ = chronological_fit_cal_split_uncapped(normal)
    fit_df = operating_frame(fit_raw, channel_map, GEARBOX_TARGETS)
    bias: dict[str, dict[str, float]] = {}
    for target, model in _fit_models(fit_df, GEARBOX_TARGETS).items():
        bias[target] = mean_residual_by_power_bin(fit_df, model)
    return bias


def gearbox_rise_bias_c(
    ds: CareDataset,
    channel_map: pd.DataFrame,
) -> dict[str, float]:
    """Mean prediction-window rise residual (°C) for the two gearbox targets."""
    normal = _status_normal_train(ds)
    fit_raw, _ = chronological_fit_cal_split_uncapped(normal)
    pred_raw = ds.prediction.sort_values("time_stamp") if "time_stamp" in ds.prediction.columns else ds.prediction
    fit_df = operating_frame(fit_raw, channel_map, GEARBOX_TARGETS)
    pred_df = operating_frame(pred_raw, channel_map, GEARBOX_TARGETS)
    bias: dict[str, float] = {}
    for target, model in _fit_models(fit_df, GEARBOX_TARGETS).items():
        values = degradation(pred_df, model).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            bias[target] = float(np.mean(values))
    return bias


def _channel_blocks(
    ds: CareDataset,
    channel_map: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    span: int = RESIDUAL_EWMA_SPAN,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """One-sided calibration and prediction z for each fitted channel."""
    normal = _status_normal_train(ds)
    fit_raw, cal_raw = chronological_fit_cal_split_uncapped(normal)
    pred_raw = ds.prediction.sort_values("time_stamp") if "time_stamp" in ds.prediction.columns else ds.prediction
    fit_df = operating_frame(fit_raw, channel_map, targets)
    cal_df = operating_frame(cal_raw, channel_map, targets)
    pred_df = operating_frame(pred_raw, channel_map, targets)
    models = _fit_models(fit_df, targets)
    if not models:
        raise ValueError(f"no rise model fitted from {len(fit_df)} operating rows")
    fit_channels = _channels(fit_df, models)
    cal_channels = (
        one_sided_channel_scores(fit_channels, _channels(cal_df, models), span=span)
        if not cal_df.empty
        else {}
    )
    pred_channels = one_sided_channel_scores(fit_channels, _channels(pred_df, models), span=span)
    if not pred_channels:
        raise ValueError("rise score produced no prediction channels")
    return cal_channels, pred_channels


def _align_series(scores: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    if scores.empty:
        return np.full(len(index), np.nan)
    values = scores[~scores.index.duplicated(keep="last")]
    return values.reindex(index).to_numpy(dtype=float)


def score_rise_into_event(event, ds: CareDataset, channel_map: pd.DataFrame, names: tuple[str, ...]) -> None:
    """Store per-channel rise z on ``event``. The combined score is applied later."""
    chosen = {
        "hybrid_rise": GEARBOX_TARGETS,
        "hybrid_multi": SUBSYSTEM_TARGETS,
    }
    for name in names:
        targets = chosen.get(name)
        if targets is None:
            continue
        try:
            cal_channels, pred_channels = _channel_blocks(ds, channel_map, targets)
            event.channel_pred[name] = {
                channel: _align_series(series, event.pred_index) for channel, series in pred_channels.items()
            }
            event.channel_cal[name] = {
                channel: series.to_numpy(dtype=float) for channel, series in cal_channels.items()
            }
            event.errors.pop(name, None)
        except Exception as exc:
            event.errors[name] = str(exc)
            event.channel_pred[name] = {}
            event.channel_cal[name] = {}
            event.z_pred[name] = np.full(len(event.pred_index), np.nan)
            event.z_cal[name] = np.array([])


def _longest_by_asset(events: list, model: str, channel: str, excluded: set[str]) -> np.ndarray:
    best: dict[str, np.ndarray] = {}
    for event in events:
        asset = str(event.asset_id)
        if asset in excluded:
            continue
        values = np.asarray(event.channel_cal.get(model, {}).get(channel, []), dtype=float)
        values = values[np.isfinite(values)]
        current = best.get(asset)
        if current is None or values.size > current.size:
            best[asset] = values
    if not best:
        return np.array([])
    return np.concatenate(list(best.values()))


def _model_pool(events: list, model: str, excluded: set[str]) -> np.ndarray:
    from wind_digital_twin.eval.care_nbm import calibration_by_asset

    parts = [
        np.asarray(values, dtype=float)
        for asset, values in calibration_by_asset(events, model).items()
        if asset not in excluded and len(values)
    ]
    if not parts:
        return np.array([])
    return np.concatenate(parts)


def _neg_log10_min(rows: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(row, dtype=float) for row in rows])
    finite = np.isfinite(stacked)
    masked = np.where(finite, stacked, np.inf)
    p_min = masked.min(axis=0)
    out = np.full(p_min.shape, np.nan)
    ok = np.isfinite(p_min) & (p_min > 0.0) & (p_min < np.inf)
    out[ok] = -np.log10(np.clip(p_min[ok], 1e-300, 1.0))
    return out


def apply_bonferroni_z(
    events: list,
    name: str,
    *,
    alpha: float,
    n_channels: int,
    source_model: str | None = None,
    extra_models: tuple[str, ...] = (),
    excluded_assets: set[str] | None = None,
    per_event_exclude: bool = True,
) -> None:
    """Store ``-log10`` of the Bonferroni min-p so the cached alarm rule is unchanged.

    Each channel's p-value uses calibration z from assets outside the excluded
    set. The decision is ``p < alpha / n_channels``. A constant calibration
    vector at ``-log10(alpha / n_channels)`` makes ``care_from_cached`` alarm
    on that same cutoff.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    source = source_model or name
    threshold = float(-np.log10(alpha / n_channels))
    held_out = {str(asset) for asset in (excluded_assets or set())}
    for event in events:
        excluded = set(held_out)
        if per_event_exclude:
            excluded.add(str(event.asset_id))
        p_rows: list[np.ndarray] = []
        for channel, pred in event.channel_pred.get(source, {}).items():
            p_rows.append(conformal_pvalues(_longest_by_asset(events, source, channel, excluded), pred))
        for extra in extra_models:
            pred = np.asarray(event.z_pred.get(extra, []), dtype=float)
            if pred.shape != (len(event.pred_index),):
                pred = np.full(len(event.pred_index), np.nan)
            p_rows.append(conformal_pvalues(_model_pool(events, extra, excluded), pred))
        if not p_rows:
            event.z_pred[name] = np.full(len(event.pred_index), np.nan)
            event.z_cal[name] = np.array([])
            continue
        event.z_pred[name] = _neg_log10_min(p_rows)
        event.z_cal[name] = np.full(_BONFERRONI_CAL_ROWS, threshold)
        event.errors.pop(name, None)


def fuse_prediction_z(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Elementwise max, keeping a finite value when the other side is missing."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"fused prediction lengths differ: {a.shape} vs {b.shape}")
    out = np.maximum(np.where(np.isnan(a), -np.inf, a), np.where(np.isnan(b), -np.inf, b))
    out = out.astype(float)
    out[np.isnan(a) & np.isnan(b)] = np.nan
    return out


def fuse_calibration_z(
    index_left: pd.DatetimeIndex,
    left: np.ndarray,
    index_right: pd.DatetimeIndex,
    right: np.ndarray,
) -> np.ndarray:
    """Max of two calibration z series on the union of their timestamps."""
    a = pd.Series(np.asarray(left, dtype=float), index=pd.DatetimeIndex(index_left))
    b = pd.Series(np.asarray(right, dtype=float), index=pd.DatetimeIndex(index_right))
    a = a[~a.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).to_numpy(dtype=float)
    fused = np.maximum(
        np.where(np.isnan(joined[:, 0]), -np.inf, joined[:, 0]),
        np.where(np.isnan(joined[:, 1]), -np.inf, joined[:, 1]),
    ).astype(float)
    fused[np.isnan(joined).all(axis=1)] = np.nan
    return fused[np.isfinite(fused)]


def attach_hybrid_ae(event, dense_cal_index: pd.DatetimeIndex) -> None:
    """Fuse the multi-target rise z with the dense autoencoder z already on ``event``."""
    if "hybrid_multi" not in event.z_pred or "dense_ae" not in event.z_pred:
        return
    try:
        event.z_pred[FUSED_MODEL_NAME] = fuse_prediction_z(
            event.z_pred["hybrid_multi"],
            event.z_pred["dense_ae"],
        )
    except ValueError as exc:
        event.errors[FUSED_MODEL_NAME] = str(exc)
        event.z_pred[FUSED_MODEL_NAME] = np.full(len(event.pred_index), np.nan)
        event.z_cal[FUSED_MODEL_NAME] = np.array([])
        return
    multi_index = event.cal_index.get("hybrid_multi", pd.DatetimeIndex([]))
    multi_z = np.asarray(event.z_cal.get("hybrid_multi", []), dtype=float)
    dense_z = np.asarray(event.z_cal.get("dense_ae", []), dtype=float)
    if len(dense_cal_index) != len(dense_z) or len(multi_index) != len(multi_z):
        event.errors[FUSED_MODEL_NAME] = (
            f"cal index mismatch dense {len(dense_z)}/{len(dense_cal_index)} "
            f"multi {len(multi_z)}/{len(multi_index)}"
        )
        event.z_cal[FUSED_MODEL_NAME] = np.array([])
        return
    event.z_cal[FUSED_MODEL_NAME] = fuse_calibration_z(
        multi_index, multi_z, dense_cal_index, dense_z
    )
    event.errors.pop(FUSED_MODEL_NAME, None)


__all__ = [
    "FUSED_MODEL_NAME",
    "GEARBOX_TARGETS",
    "POWER_SPLINE_KNOTS",
    "RISE_MODEL_NAMES",
    "SUBSYSTEM_TARGETS",
    "apply_bonferroni_z",
    "attach_hybrid_ae",
    "degradation",
    "fit_rise",
    "fuse_calibration_z",
    "fuse_prediction_z",
    "gearbox_fit_bin_bias_c",
    "gearbox_rise_bias_c",
    "matrix_calibration_index",
    "mean_residual_by_power_bin",
    "operating_frame",
    "score_rise_blocks",
    "score_rise_into_event",
]
