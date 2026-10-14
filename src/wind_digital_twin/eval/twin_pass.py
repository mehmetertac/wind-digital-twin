"""End-to-end digital twin pass: SCADA → physics → hybrid → anomaly score."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wind_digital_twin.anomaly.hybrid_residual_detector import (
    HybridResidualPipeline,
    fit_hybrid_residual_detector,
)
from wind_digital_twin.anomaly.scoring import threshold_from_training
from wind_digital_twin.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    POWER_COLUMN,
    PROJECT_ROOT,
    RESULTS_DIR,
    THERMAL_MIN_POWER_KW,
    THERMAL_TARGET_COLUMNS,
)
from wind_digital_twin.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_digital_twin.data.load_edp import load_edp_dataset
from wind_digital_twin.data.synthetic_edp import is_synthetic_dataset
from wind_digital_twin.eval.plots import (
    plot_hybrid_vs_physics_rmse,
    plot_twin_timeline,
)
from wind_digital_twin.eval.protocol import (
    AlarmEpisode,
    detect_alarm_episodes,
    evaluate_turbine,
)
from wind_digital_twin.physics.gearbox_thermal import healthy_train_validate_split
from wind_digital_twin.residual.fusion_features import build_fusion_feature_frame
from wind_digital_twin.residual.learned_residual import fit_hybrid_thermal_corrector

TWIN_PASS_RESULTS_DIR = RESULTS_DIR / "twin_pass"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


@dataclass
class TwinPassResult:
    """Artifacts from one turbine end-to-end twin pass."""

    turbine_id: str
    synthetic: bool
    failure_remarks: str | None
    layer_frame: pd.DataFrame
    threshold: float
    corrector_metrics: dict[str, Any]
    eval_summary: dict[str, Any]
    episodes: list[AlarmEpisode]
    timeline_figure: Path
    rmse_figure: Path
    frame_path: Path


def build_twin_layer_frame(
    df: pd.DataFrame,
    pipeline: HybridResidualPipeline,
) -> pd.DataFrame:
    """
    Align SCADA, physics expectation, hybrid expectation, residuals, and scores.

    Rows are operating timestamps (power/RPM filter shared with corrector).
    Anomaly scores are NaN where rolling detector features are incomplete.
    """
    hybrid = pipeline.hybrid
    oil_col, bear_col = THERMAL_TARGET_COLUMNS
    operating = build_fusion_feature_frame(df).index

    oil_phys = hybrid.oil_model.predict(df).reindex(operating)
    bear_phys = hybrid.bear_model.predict(df).reindex(operating)
    oil_hyb = hybrid.hybrid_predict(df, oil_col).reindex(operating)
    bear_hyb = hybrid.hybrid_predict(df, bear_col).reindex(operating)
    oil_act = df[oil_col].reindex(operating)
    bear_act = df[bear_col].reindex(operating)
    scores = pipeline.score(df).reindex(operating)

    return pd.DataFrame(
        {
            "oil_actual": oil_act,
            "bear_actual": bear_act,
            "oil_physics_pred": oil_phys,
            "bear_physics_pred": bear_phys,
            "oil_hybrid_pred": oil_hyb,
            "bear_hybrid_pred": bear_hyb,
            "oil_physics_residual": oil_phys - oil_act,
            "bear_physics_residual": bear_phys - bear_act,
            "oil_hybrid_residual": oil_hyb - oil_act,
            "bear_hybrid_residual": bear_hyb - bear_act,
            "anomaly_score": scores,
            POWER_COLUMN: df[POWER_COLUMN].reindex(operating),
        },
        index=operating,
    )


def threshold_from_healthy_holdout(
    pipeline: HybridResidualPipeline,
    healthy_df: pd.DataFrame,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
) -> tuple[float, pd.DataFrame]:
    """Threshold from time-ordered healthy validation tail (not pre-failure buffer)."""
    _train, val_df = healthy_train_validate_split(healthy_df)
    if val_df.empty:
        val_df = healthy_df.iloc[max(0, len(healthy_df) - 500) :]
    holdout_scores = pipeline.score(val_df)
    if holdout_scores.empty:
        holdout_scores = pipeline.score(healthy_df)
    return threshold_from_training(holdout_scores, percentile=percentile), val_df


def run_twin_pass(
    raw_dir: Path = DATA_RAW,
    turbine_id: str = "T06",
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    lookback_days: int = 120,
    figures_dir: Path | None = None,
    results_dir: Path | None = None,
) -> TwinPassResult:
    """Fit hybrid detector on healthy data; score full history; export figures."""
    turbines, failures = load_edp_dataset(raw_dir)
    if turbine_id not in turbines:
        raise KeyError(f"{turbine_id} not in dataset")

    df = clean_turbine_df(turbines[turbine_id], min_power_kw=THERMAL_MIN_POWER_KW)
    failure = get_failure_for_turbine(turbine_id, failures)
    healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
    healthy_df = df.loc[healthy_mask]

    if len(healthy_df) < 500:
        raise ValueError(
            f"Insufficient healthy rows for {turbine_id} ({len(healthy_df)})"
        )

    _corrector, corrector_metrics = fit_hybrid_thermal_corrector(healthy_df)
    pipeline = fit_hybrid_residual_detector(healthy_df)
    threshold, _val_holdout = threshold_from_healthy_holdout(pipeline, healthy_df)

    layer = build_twin_layer_frame(df, pipeline)
    scores = layer["anomaly_score"].dropna()
    eval_result = evaluate_turbine(
        turbine_id, scores, threshold, failure=failure
    )
    episodes = detect_alarm_episodes(scores, threshold=threshold)

    figures_dir = figures_dir or DOCS_FIGURES_DIR
    results_dir = results_dir or TWIN_PASS_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    failure_ts = eval_result.failure_time
    timeline_path = plot_twin_timeline(
        turbine_id=turbine_id,
        layer_frame=layer,
        failure_time=failure_ts,
        threshold=threshold,
        episodes=episodes,
        buffer_days=buffer_days,
        lookback_days=lookback_days,
        synthetic=is_synthetic_dataset(raw_dir),
        failure_remarks=failure.remarks if failure else None,
        out_path=figures_dir / f"twin_timeline_{turbine_id}.png",
    )
    rmse_path = plot_hybrid_vs_physics_rmse(
        corrector_metrics,
        synthetic=is_synthetic_dataset(raw_dir),
        out_path=figures_dir / "hybrid_vs_physics_rmse.png",
    )

    frame_path = results_dir / f"{turbine_id}_layers.parquet"
    layer.to_parquet(frame_path)

    summary_path = results_dir / f"{turbine_id}_summary.json"
    summary = {
        "turbine_id": turbine_id,
        "synthetic": is_synthetic_dataset(raw_dir),
        "threshold": threshold,
        "corrector_metrics": corrector_metrics,
        "eval": {k: v for k, v in asdict(eval_result).items()},
        "failure_remarks": failure.remarks if failure else None,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    return TwinPassResult(
        turbine_id=turbine_id,
        synthetic=is_synthetic_dataset(raw_dir),
        failure_remarks=failure.remarks if failure else None,
        layer_frame=layer,
        threshold=threshold,
        corrector_metrics=corrector_metrics,
        eval_summary=asdict(eval_result),
        episodes=episodes,
        timeline_figure=timeline_path,
        rmse_figure=rmse_path,
        frame_path=frame_path,
    )


__all__ = [
    "TwinPassResult",
    "build_twin_layer_frame",
    "run_twin_pass",
    "threshold_from_healthy_holdout",
]
