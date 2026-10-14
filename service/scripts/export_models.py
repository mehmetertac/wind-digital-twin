#!/usr/bin/env python
"""Export synthetic smoke models under ``service/models/`` for CI and Docker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SERVICE_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from wind_digital_twin.anomaly.hybrid_residual_detector import (  # noqa: E402
    fit_hybrid_residual_detector,
)
from wind_digital_twin.config import DATA_RAW, GEARBOX_FAILURES, THERMAL_MIN_POWER_KW
from wind_digital_twin.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_digital_twin.data.load_edp import load_edp_dataset
from wind_digital_twin.data.synthetic_edp import generate_synthetic_edp_dataset
from wind_digital_twin.eval.twin_pass import (  # noqa: E402
    build_twin_layer_frame,
    threshold_from_healthy_holdout,
)
from wind_digital_twin.residual.fusion_features import build_fusion_feature_frame

from twin_service.constants import (  # noqa: E402
    DAILY_FEATURE_COLUMNS,
    FEATURE_TO_SENSOR,
    RUL_MAX_DAYS,
    SENSOR_ORDER,
)

MODEL_VERSION = "synthetic-smoke-v1"
LGBM_PARAMS = {
    "num_leaves": 7,
    "n_estimators": 120,
    "learning_rate": 0.08,
    "min_child_samples": 5,
    "random_state": 11,
    "verbosity": -1,
    "force_col_wise": True,
    "n_jobs": 1,
}


def _ensure_synthetic(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not list(raw_dir.glob("*.csv")):
        generate_synthetic_edp_dataset(raw_dir, seed=42, force=True)


def _failure_log_frame() -> pd.DataFrame:
    rows = []
    for entry in GEARBOX_FAILURES:
        rows.append(
            {
                "turbine_id": entry["turbine_id"],
                "timestamp": entry["timestamp"],
                "component": "GEARBOX",
                "remarks": entry["remarks"],
            }
        )
    return pd.DataFrame(rows)


def _scored_from_layer(
    clean: pd.DataFrame,
    layer: pd.DataFrame,
    turbine_id: str,
) -> pd.DataFrame:
    sub = clean.reindex(layer.index).dropna(how="any")
    common = sub.index.intersection(layer.index)
    layer = layer.loc[common]
    sub = sub.loc[common]
    fusion = build_fusion_feature_frame(sub)
    frame = pd.DataFrame(
        {
            "timestamp": layer.index,
            "turbine_id": turbine_id,
            "oil_residual": layer["oil_hybrid_residual"],
            "bear_residual": layer["bear_hybrid_residual"],
            "anomaly_score": layer["anomaly_score"],
            "theta_oil": fusion.reindex(layer.index)["theta_oil"],
            "theta_bear": fusion.reindex(layer.index)["theta_bear"],
            "delta_bear_oil": fusion.reindex(layer.index)["delta_bear_oil"],
        }
    )
    return frame.dropna(subset=["anomaly_score"])


def _daily_from_scored(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["oil_deg"] = -frame["oil_residual"].astype(float)
    frame["bear_deg"] = -frame["bear_residual"].astype(float)
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")
    daily = frame.groupby(["turbine_id", "date"], as_index=False).agg(
        oil_deg_mean=("oil_deg", "mean"),
        oil_deg_std=("oil_deg", "std"),
        bear_deg_mean=("bear_deg", "mean"),
        bear_deg_std=("bear_deg", "std"),
        anomaly_score_mean=("anomaly_score", "mean"),
        anomaly_score_max=("anomaly_score", "max"),
        theta_oil_mean=("theta_oil", "mean"),
        theta_bear_mean=("theta_bear", "mean"),
        delta_bear_oil_mean=("delta_bear_oil", "mean"),
    )
    for col in ("oil_deg_std", "bear_deg_std"):
        daily[col] = daily[col].fillna(0.0)
    daily = daily.sort_values(["turbine_id", "date"])
    for deg_col, slope_col in (
        ("oil_deg_mean", "oil_deg_slope_7d"),
        ("bear_deg_mean", "bear_deg_slope_7d"),
    ):
        slopes = []
        for _, group in daily.groupby("turbine_id", sort=False):
            values = group[deg_col].astype(float).values
            out = [0.0] * len(values)
            for i in range(len(values)):
                start = max(0, i - 6)
                window = values[start : i + 1]
                if len(window) >= 2:
                    out[i] = float((window[-1] - window[0]) / max(len(window) - 1, 1))
            slopes.extend(out)
        daily[slope_col] = slopes
    return daily


def _attach_synthetic_rul(daily: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    events = failures.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events = events.sort_values(["turbine_id", "timestamp"])
    labeled = daily.copy()
    labeled["rul"] = float(RUL_MAX_DAYS)
    labeled["censored"] = True
    labeled["days_to_event"] = np.nan
    for idx, row in labeled.iterrows():
        tid = row["turbine_id"]
        day = pd.Timestamp(row["date"])
        if day.tzinfo is None:
            day = day.tz_localize("UTC")
        else:
            day = day.tz_convert("UTC")
        future = events.loc[
            (events["turbine_id"] == tid) & (events["timestamp"].dt.floor("D") > day)
        ]
        if future.empty:
            continue
        next_ev = future.iloc[0]["timestamp"]
        days = (next_ev.floor("D") - day).days
        if days <= 0:
            continue
        labeled.at[idx, "days_to_event"] = days
        labeled.at[idx, "rul"] = float(min(RUL_MAX_DAYS, days))
        labeled.at[idx, "censored"] = days > RUL_MAX_DAYS
    return labeled


def _fit_rul_bundle(labeled: pd.DataFrame) -> dict:
    train = labeled.loc[~labeled["censored"].astype(bool) | (labeled["rul"] <= RUL_MAX_DAYS)]
    if len(train) < 30:
        train = labeled
    x = train.loc[:, DAILY_FEATURE_COLUMNS].fillna(0.0)
    y = train["rul"].astype(float)
    models: dict[float, LGBMRegressor] = {}
    for q in (0.1, 0.5, 0.9):
        model = LGBMRegressor(**LGBM_PARAMS, objective="quantile", alpha=q)
        model.fit(x, y)
        models[q] = model
    return {
        "models": models,
        "feature_columns": DAILY_FEATURE_COLUMNS,
        "monotone_applied": False,
        "rul_max": RUL_MAX_DAYS,
    }


def _conformal_deltas(bundle: dict, labeled: pd.DataFrame) -> dict[str, float]:
    cal = labeled.loc[~labeled["censored"].astype(bool)].tail(40)
    if len(cal) < 5:
        cal = labeled.tail(40)
    x = cal.loc[:, DAILY_FEATURE_COLUMNS].fillna(0.0)
    y = cal["rul"].astype(float).to_numpy()
    q10 = bundle["models"][0.1].predict(x)
    q90 = bundle["models"][0.9].predict(x)
    scores_80 = np.maximum(q10 - y, y - q90)
    delta_80 = float(np.quantile(scores_80, 0.80, method="higher"))
    q05 = bundle["models"][0.1].predict(x)
    q95 = bundle["models"][0.9].predict(x)
    scores_90 = np.maximum(q05 - y, y - q95)
    delta_90 = float(np.quantile(scores_90, 0.90, method="higher"))
    return {"delta_80": max(delta_80, 0.5), "delta_90": max(delta_90, 1.0)}


def _sensor_shap_ranking(bundle: dict, labeled: pd.DataFrame) -> pd.DataFrame:
    import shap

    x = labeled.loc[:, DAILY_FEATURE_COLUMNS].fillna(0.0)
    rows: list[dict] = []
    for q, learner in ((0.1, "p10"), (0.5, "p50")):
        explainer = shap.TreeExplainer(bundle["models"][q])
        shap_vals = explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_by_feature = np.mean(shap_vals, axis=0)
        sensor_sums: dict[str, list[float]] = {s: [] for s in SENSOR_ORDER}
        for col, val in zip(DAILY_FEATURE_COLUMNS, mean_by_feature, strict=True):
            sensor_sums[FEATURE_TO_SENSOR[col]].append(float(val))
        ranked: list[tuple[str, float, float]] = []
        for sensor in SENSOR_ORDER:
            vals = sensor_sums[sensor]
            ranked.append((sensor, float(np.sum(vals)), float(np.mean(np.abs(vals)))))
        ranked.sort(key=lambda item: item[2], reverse=True)
        for rank, (sensor, mean_shap, mean_abs) in enumerate(ranked, start=1):
            feats = [c for c in DAILY_FEATURE_COLUMNS if FEATURE_TO_SENSOR[c] == sensor]
            rows.append(
                {
                    "sensor": sensor,
                    "learner": learner,
                    "mean_shap": mean_shap,
                    "mean_abs_shap": mean_abs,
                    "rank": rank,
                    "n_rows": int(len(x)),
                    "features": ";".join(feats),
                }
            )
    return pd.DataFrame(rows)


def _predict_rul_columns(bundle: dict, daily: pd.DataFrame, deltas: dict[str, float]) -> pd.DataFrame:
    x = daily.loc[:, DAILY_FEATURE_COLUMNS].fillna(0.0)
    p10 = bundle["models"][0.1].predict(x)
    p50 = bundle["models"][0.5].predict(x)
    p90 = bundle["models"][0.9].predict(x)
    out = daily.copy()
    out["rul_p10"] = np.clip(p10, 0, RUL_MAX_DAYS)
    out["rul_p50"] = np.clip(p50, 0, RUL_MAX_DAYS)
    out["rul_p90"] = np.clip(p90, 0, RUL_MAX_DAYS)
    d80 = deltas["delta_80"]
    d90 = deltas["delta_90"]
    out["cqr80_lo"] = np.clip(out["rul_p10"] - d80, 0, RUL_MAX_DAYS)
    out["cqr80_hi"] = np.clip(out["rul_p90"] + d80, 0, RUL_MAX_DAYS)
    out["cqr90_lo"] = np.clip(out["rul_p10"] - d90, 0, RUL_MAX_DAYS)
    out["cqr90_hi"] = np.clip(out["rul_p90"] + d90, 0, RUL_MAX_DAYS)
    return out


def _build_timeline(
    turbines: dict[str, pd.DataFrame],
    failures: pd.DataFrame,
    pipeline,
    threshold: float,
    bundle: dict,
    deltas: dict[str, float],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for tid, df in turbines.items():
        if tid not in {"T01", "T06"}:
            continue
        clean = clean_turbine_df(df, min_power_kw=THERMAL_MIN_POWER_KW)
        layer = build_twin_layer_frame(clean, pipeline)
        layer = layer.dropna(subset=["anomaly_score"])
        if layer.empty:
            continue
        layer_ts = layer.copy()
        layer_ts["date"] = pd.to_datetime(layer_ts.index, utc=True).floor("D")
        daily_phys = layer_ts.groupby("date", as_index=False).agg(
            oil_temp_actual=("oil_actual", "mean"),
            oil_temp_physics=("oil_physics_pred", "mean"),
            oil_residual=("oil_hybrid_residual", "mean"),
            anomaly_score=("anomaly_score", "mean"),
        )
        daily_phys["date"] = pd.to_datetime(daily_phys["date"], utc=True)
        daily_phys["turbine_id"] = tid
        scored = _scored_from_layer(clean, layer, tid)
        daily_feats = _daily_from_scored(scored)
        daily_feats = _attach_synthetic_rul(daily_feats, failures)
        daily_feats = _predict_rul_columns(bundle, daily_feats, deltas)
        merge_cols = ["date", *list(DAILY_FEATURE_COLUMNS)]
        merge_cols.extend(
            [
                "rul_p10",
                "rul_p50",
                "rul_p90",
                "cqr80_lo",
                "cqr80_hi",
                "cqr90_lo",
                "cqr90_hi",
            ]
        )
        merged = daily_phys.merge(
            daily_feats.loc[:, merge_cols],
            on="date",
            how="left",
        )
        merged["anomaly_threshold"] = threshold
        event_rows = failures.loc[failures["turbine_id"] == tid]
        if not event_rows.empty:
            fail_day = pd.to_datetime(event_rows.iloc[-1]["timestamp"], utc=True).floor("D")
            window_start = fail_day - pd.Timedelta(days=RUL_MAX_DAYS)
            merged = merged.loc[
                (merged["date"] >= window_start) & (merged["date"] <= fail_day)
            ]
        parts.append(merged)
    if not parts:
        raise RuntimeError("No timeline rows exported")
    return pd.concat(parts, ignore_index=True)


def export_models(*, raw_dir: Path = DATA_RAW, out_dir: Path = MODELS_DIR) -> None:
    _ensure_synthetic(raw_dir)
    turbines, gearbox_failures = load_edp_dataset(raw_dir)
    failures = _failure_log_frame()

    primary_turbine = "T06"
    if primary_turbine not in turbines:
        primary_turbine = next(iter(turbines))
    clean_primary = clean_turbine_df(
        turbines[primary_turbine],
        min_power_kw=THERMAL_MIN_POWER_KW,
    )
    failure_primary = get_failure_for_turbine(primary_turbine, gearbox_failures)
    healthy_mask = healthy_training_mask(clean_primary.index, failure_primary, buffer_days=90)
    healthy_df = clean_primary.loc[healthy_mask]
    if len(healthy_df) < 500:
        raise ValueError(f"Insufficient healthy rows ({len(healthy_df)})")

    pipeline = fit_hybrid_residual_detector(healthy_df)
    threshold, _ = threshold_from_healthy_holdout(pipeline, healthy_df)

    scored_parts: list[pd.DataFrame] = []
    for tid, df in turbines.items():
        clean = clean_turbine_df(df, min_power_kw=THERMAL_MIN_POWER_KW)
        layer = build_twin_layer_frame(clean, pipeline)
        scored_parts.append(_scored_from_layer(clean, layer, tid))
    scored_all = pd.concat(scored_parts, ignore_index=True)
    daily = _daily_from_scored(scored_all)
    labeled = _attach_synthetic_rul(daily, failures)
    bundle = _fit_rul_bundle(labeled)
    deltas = _conformal_deltas(bundle, labeled)
    ranking = _sensor_shap_ranking(bundle, labeled)
    timeline = _build_timeline(turbines, failures, pipeline, threshold, bundle, deltas)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "hybrid_pipeline.joblib")
    (out_dir / "anomaly_threshold.json").write_text(
        json.dumps(
            {"threshold": threshold, "model_version": MODEL_VERSION},
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(bundle, out_dir / "rul_bundle.joblib")
    (out_dir / "conformal_offsets.json").write_text(
        json.dumps(deltas, indent=2),
        encoding="utf-8",
    )
    ranking.to_csv(out_dir / "shap_sensor_ranking.csv", index=False)
    timeline.to_parquet(out_dir / "timeline_demo.parquet", index=False)
    print(f"Wrote serving artifacts to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    export_models(raw_dir=args.raw_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
