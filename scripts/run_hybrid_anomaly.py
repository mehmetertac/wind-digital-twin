#!/usr/bin/env python3
"""Spot-check T06: hybrid-residual detector vs Week 6 physics-only hybrid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wind_digital_twin.anomaly.hybrid_residual_detector import (  # noqa: E402
    fit_hybrid_residual_detector,
)
from wind_digital_twin.anomaly.physics_hybrid import fit_physics_hybrid  # noqa: E402
from wind_digital_twin.anomaly.scoring import threshold_from_training  # noqa: E402
from wind_digital_twin.config import (  # noqa: E402
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    RESIDUAL_RESULTS_DIR,
    THERMAL_MIN_POWER_KW,
)
from wind_digital_twin.data.clean import (  # noqa: E402
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_digital_twin.data.load_edp import load_edp_dataset  # noqa: E402
from wind_digital_twin.data.synthetic_edp import is_synthetic_dataset  # noqa: E402
from wind_digital_twin.eval.protocol import evaluate_turbine  # noqa: E402
from wind_digital_twin.residual.residual_features import build_residual_feature_frame  # noqa: E402


def _mean_bear_deg(scores_index: pd.DatetimeIndex, df: pd.DataFrame, pipeline) -> float:
    """Pre-failure mean bear degradation from physics or hybrid feature builder."""
    if hasattr(pipeline, "oil_model"):
        feats = build_residual_feature_frame(df, pipeline.oil_model, pipeline.bear_model)
    else:
        from wind_digital_twin.anomaly.hybrid_residual_detector import (
            build_hybrid_detector_feature_frame,
        )

        feats = build_hybrid_detector_feature_frame(df, pipeline.hybrid)
    common = feats.index.intersection(scores_index)
    if common.empty:
        return float("nan")
    return float(feats.loc[common, "bear_deg"].mean())


def run_t06_spotcheck(
    raw_dir: Path = DATA_RAW,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    turbine_id: str = "T06",
) -> dict:
    turbines, failures = load_edp_dataset(raw_dir)
    if turbine_id not in turbines:
        raise KeyError(f"{turbine_id} not in dataset")

    df = clean_turbine_df(turbines[turbine_id], min_power_kw=THERMAL_MIN_POWER_KW)
    failure = get_failure_for_turbine(turbine_id, failures)
    healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
    healthy_df = df.loc[healthy_mask]
    n_train = min(len(healthy_df), max(2000, int(len(healthy_df) * 0.6)))
    train = healthy_df.iloc[:n_train]
    score_df = df.iloc[n_train:]

    physics_pipe = fit_physics_hybrid(train)
    hybrid_pipe = fit_hybrid_residual_detector(train)

    train_phys = physics_pipe.score(train.iloc[n_train // 2 :])
    train_hybrid = hybrid_pipe.score(train.iloc[n_train // 2 :])
    thresh_phys = threshold_from_training(
        train_phys, percentile=DEFAULT_THRESHOLD_PERCENTILE
    )
    thresh_hybrid = threshold_from_training(
        train_hybrid, percentile=DEFAULT_THRESHOLD_PERCENTILE
    )

    phys_scores = physics_pipe.score(score_df)
    hybrid_scores = hybrid_pipe.score(score_df)

    eval_phys = evaluate_turbine(
        turbine_id, phys_scores, thresh_phys, failure=failure
    )
    eval_hybrid = evaluate_turbine(
        turbine_id, hybrid_scores, thresh_hybrid, failure=failure
    )

    if failure is not None:
        pre_fail = score_df.index < pd.Timestamp(failure.timestamp)
        ramp = score_df.loc[pre_fail]
        phys_ramp = physics_pipe.score(ramp)
        hybrid_ramp = hybrid_pipe.score(ramp)
    else:
        phys_ramp = phys_scores
        hybrid_ramp = hybrid_scores

    report = {
        "turbine_id": turbine_id,
        "synthetic": is_synthetic_dataset(raw_dir),
        "physics_only": {
            "lead_time_days": eval_phys.lead_time_days,
            "false_alarm_episodes": eval_phys.false_alarm_episodes,
            "mean_bear_deg_scored": _mean_bear_deg(phys_scores.index, score_df, physics_pipe),
            "mean_score_ramp": float(phys_ramp.mean()) if len(phys_ramp) else None,
        },
        "hybrid_residual": {
            "lead_time_days": eval_hybrid.lead_time_days,
            "false_alarm_episodes": eval_hybrid.false_alarm_episodes,
            "mean_bear_deg_scored": _mean_bear_deg(hybrid_scores.index, score_df, hybrid_pipe),
            "mean_score_ramp": float(hybrid_ramp.mean()) if len(hybrid_ramp) else None,
        },
        "pass_hybrid_as_good_or_better": (
            (eval_hybrid.lead_time_days or 0) >= (eval_phys.lead_time_days or 0)
            or (
                hybrid_ramp.mean() > phys_ramp.mean()
                if len(hybrid_ramp) and len(phys_ramp)
                else False
            )
        )
        and eval_hybrid.false_alarm_episodes <= eval_phys.false_alarm_episodes + 2,
    }

    out_dir = RESIDUAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{turbine_id}_detector_spotcheck.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="T06 hybrid vs physics detector spot-check")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--turbine", default="T06")
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    args = parser.parse_args()
    run_t06_spotcheck(
        raw_dir=args.data_dir, buffer_days=args.buffer_days, turbine_id=args.turbine
    )


if __name__ == "__main__":
    main()
