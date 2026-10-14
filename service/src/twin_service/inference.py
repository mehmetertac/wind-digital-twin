"""Scoring and RUL inference using loaded artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from twin_service.artifacts import LoadedArtifacts, bundle_rul_max
from twin_service.constants import DAILY_FEATURE_COLUMNS, MIN_SCADA_ROWS, RUL_MAX_DAYS
from twin_service.schema import (
    ScoredPoint,
    SensorAttribution,
    TwinRulRequest,
    TwinRulResponse,
    TwinScoreRequest,
    TwinScoreResponse,
    enforce_quantile_order,
)


def _scada_frame(body: TwinScoreRequest) -> pd.DataFrame:
    rows = [row.model_dump() for row in body.rows]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.drop(columns=["timestamp"], errors="ignore")
    return frame


def score_window(artifacts: LoadedArtifacts, body: TwinScoreRequest) -> TwinScoreResponse:
    if len(body.rows) < MIN_SCADA_ROWS:
        raise ValueError(f"SCADA window must have at least {MIN_SCADA_ROWS} rows")

    df = _scada_frame(body)
    pipeline = artifacts.hybrid_pipeline
    hybrid = pipeline.hybrid
    oil_col = "Gear_Oil_Temp_Avg"

    expected = hybrid.oil_model.predict(df)
    hybrid_pred = hybrid.hybrid_predict(df, oil_col)
    actual = df[oil_col]
    common = expected.index.intersection(hybrid_pred.index).intersection(actual.index)
    scores = pipeline.score(df)

    points: list[ScoredPoint] = []
    for ts in common:
        ts_iso = pd.Timestamp(ts).isoformat().replace("+00:00", "Z")
        score_val = scores.loc[ts] if ts in scores.index else float("nan")
        if pd.isna(score_val):
            continue
        points.append(
            ScoredPoint(
                timestamp=ts_iso,
                expected_temp_c=float(expected.loc[ts]),
                residual_c=float(hybrid_pred.loc[ts] - actual.loc[ts]),
                anomaly_score=float(score_val),
            )
        )

    return TwinScoreResponse(
        turbine_id=body.turbine_id,
        anomaly_threshold=artifacts.anomaly_threshold,
        points=points,
    )


def _feature_matrix(row: dict[str, float], feature_columns: tuple[str, ...]) -> np.ndarray:
    return np.array([[float(row[col]) for col in feature_columns]], dtype=float)


def _model_map(bundle: dict[str, Any]) -> dict[float, Any]:
    raw = bundle["models"]
    out: dict[float, Any] = {}
    for key, model in raw.items():
        out[float(key)] = model
    return out


def _predict_quantiles(bundle: dict[str, Any], features: dict[str, float]) -> tuple[float, float, float]:
    models = _model_map(bundle)
    cols = tuple(bundle["feature_columns"])
    x = pd.DataFrame([features], columns=list(cols))
    q10 = float(models[0.1].predict(x)[0])
    q50 = float(models[0.5].predict(x)[0])
    q90 = float(models[0.9].predict(x)[0])
    rul_max = bundle_rul_max(bundle)
    return enforce_quantile_order(
        float(np.clip(q10, 0, rul_max)),
        float(np.clip(q50, 0, rul_max)),
        float(np.clip(q90, 0, rul_max)),
    )


def _apply_conformal(
    q10: float,
    q50: float,
    q90: float,
    *,
    delta_80: float,
    delta_90: float,
    rul_max: int,
) -> tuple[float, float, float, float, float, float]:
    c80_lo = max(0.0, q10 - delta_80)
    c80_hi = min(float(rul_max), q90 + delta_80)
    c90_lo = max(0.0, q10 - delta_90)
    c90_hi = min(float(rul_max), q90 + delta_90)
    return c80_lo, c80_hi, c90_lo, c90_hi, q50, q50


def top_sensor_attributions(
    ranking: pd.DataFrame,
    *,
    learner: str = "p50",
    limit: int = 5,
) -> list[SensorAttribution]:
    work = ranking.loc[ranking["learner"] == learner].sort_values("rank")
    out: list[SensorAttribution] = []
    for _, row in work.head(limit).iterrows():
        out.append(
            SensorAttribution(
                sensor=str(row["sensor"]),
                learner=str(row["learner"]),
                mean_shap=float(row["mean_shap"]),
                mean_abs_shap=float(row["mean_abs_shap"]),
                rank=int(row["rank"]),
            )
        )
    return out


def predict_rul(artifacts: LoadedArtifacts, body: TwinRulRequest) -> TwinRulResponse:
    features = {col: getattr(body, col) for col in DAILY_FEATURE_COLUMNS}
    q10, q50, q90 = _predict_quantiles(artifacts.rul_bundle, features)
    rul_max = bundle_rul_max(artifacts.rul_bundle)
    c80_lo, c80_hi, c90_lo, c90_hi, _, _ = _apply_conformal(
        q10,
        q50,
        q90,
        delta_80=artifacts.conformal.delta_80,
        delta_90=artifacts.conformal.delta_90,
        rul_max=rul_max,
    )
    return TwinRulResponse(
        turbine_id=body.turbine_id,
        date=body.date,
        rul_p10=q10,
        rul_p50=q50,
        rul_p90=q90,
        cqr80_lo=c80_lo,
        cqr80_hi=c80_hi,
        cqr90_lo=c90_lo,
        cqr90_hi=c90_hi,
        attributions=top_sensor_attributions(artifacts.shap_ranking),
    )


def list_turbines(artifacts: LoadedArtifacts) -> list[str]:
    turbines = sorted(artifacts.timeline["turbine_id"].astype(str).unique().tolist())
    return turbines


def series_for_turbine(artifacts: LoadedArtifacts, turbine_id: str) -> pd.DataFrame:
    work = artifacts.timeline.loc[artifacts.timeline["turbine_id"] == turbine_id].copy()
    return work.sort_values("date")


def filter_series_to_date(frame: pd.DataFrame, selected_date: str) -> pd.DataFrame:
    """Return rows up to and including ``selected_date`` (for dashboard scrub)."""
    if frame.empty:
        return frame
    day = pd.to_datetime(selected_date, utc=True).floor("D")
    dates = pd.to_datetime(frame["date"], utc=True)
    return frame.loc[dates <= day].copy()
