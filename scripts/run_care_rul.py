#!/usr/bin/env python3
"""Leave-one-failure-out quantile RUL on CARE Wind Farm A drivetrain events.

Uses grid-gnn-faults ``evaluate_quantile_model`` / ``evaluate_conformal_rul``.
Lead times are integers or ``never crossed on event <id>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNN_SRC = PROJECT_ROOT.parent / "grid-gnn-faults" / "src"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(GNN_SRC))

from wind_digital_twin.config import CARE_RESULTS_DIR, DATA_RAW_CARE_WFA  # noqa: E402
from wind_digital_twin.data.care_channels import load_frozen_channel_map  # noqa: E402
from wind_digital_twin.data.load_care import load_care_dataset, load_event_info  # noqa: E402
from wind_digital_twin.eval.care_rul import (  # noqa: E402
    care_id_from_remarks,
    failure_log_frame,
    independent_onsets,
    scored_frame_for_asset,
    select_failure_events,
)
from wind_digital_twin.eval.identity_calibration import format_crossed_lead  # noqa: E402


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _coverage_value(row: pd.Series, column: str):
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def evaluate_mode(mode: str, farm_dir: Path, out_dir: Path) -> dict:
    from rul.conformal import evaluate_conformal_rul
    from rul.labels import attach_rul_labels, daily_from_scored_frame, normalize_failure_log
    from rul.quantile import evaluate_quantile_model, warning_lead_time_days

    catalog = load_event_info(farm_dir)
    selected = select_failure_events(catalog, mode)
    kept, dropped = independent_onsets(selected)
    failures = failure_log_frame(kept)
    channel_map = load_frozen_channel_map()
    assets = {str(event.asset_id) for event in kept}
    pieces: list[pd.DataFrame] = []
    for asset in sorted(assets):
        asset_events = [event for event in catalog if str(event.asset_id) == asset]
        print(f"{mode}: scoring asset {asset} ({len(asset_events)} windows)", flush=True)
        datasets = [load_care_dataset(farm_dir, event.event_id) for event in asset_events]
        asset_failures = [event for event in kept if str(event.asset_id) == asset]
        pieces.append(scored_frame_for_asset(datasets, channel_map, asset_failures))
    scored = pd.concat(pieces, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scored_10min.csv"
    failures_path = out_dir / "failures.csv"
    scored.to_csv(scored_path, index=False)
    failures.to_csv(failures_path, index=False)

    daily = daily_from_scored_frame(scored)
    labeled = attach_rul_labels(daily, failures)
    labeled.to_csv(out_dir / "labeled_daily.csv", index=False)
    metrics, _coverage, _warnings, fold_results = evaluate_quantile_model(labeled)
    conformal, _width, _folds = evaluate_conformal_rul(labeled)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    conformal.to_csv(out_dir / "conformal_coverage.csv", index=False)

    normalized = normalize_failure_log(failures)
    care_by_fold = {
        str(row.event_id): care_id_from_remarks(row.remarks) for row in normalized.itertuples(index=False)
    }
    leads_by_fold: dict[str, dict[str, str]] = {}
    for scored_fold in fold_results:
        care_id = care_by_fold.get(scored_fold.fold.event_id, scored_fold.fold.event_id)
        leads_by_fold[scored_fold.fold.event_id] = {
            "p10_lead": format_crossed_lead(
                warning_lead_time_days(scored_fold.fold.test, scored_fold.q10),
                care_id,
            ),
            "p50_lead": format_crossed_lead(
                warning_lead_time_days(scored_fold.fold.test, scored_fold.q50),
                care_id,
            ),
            "p90_lead": format_crossed_lead(
                warning_lead_time_days(scored_fold.fold.test, scored_fold.q90),
                care_id,
            ),
        }

    metric_by_id = {str(row.event_id): row for _, row in metrics.iterrows()} if not metrics.empty else {}
    conformal_by_id = (
        {str(row.event_id): row for _, row in conformal.iterrows()} if not conformal.empty else {}
    )
    event_rows = []
    for row in normalized.itertuples(index=False):
        fold_id = str(row.event_id)
        care_id = care_id_from_remarks(row.remarks)
        metric = metric_by_id.get(fold_id)
        band = conformal_by_id.get(fold_id)
        leads = leads_by_fold.get(fold_id) or {
            "p10_lead": format_crossed_lead(None, care_id),
            "p50_lead": format_crossed_lead(None, care_id),
            "p90_lead": format_crossed_lead(None, care_id),
        }
        event_rows.append(
            {
                "care_event_id": care_id,
                "turbine_id": str(row.turbine_id),
                "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                "remarks": str(row.remarks),
                "pinball_0.1": None if metric is None else _coverage_value(metric, "pinball_0.1"),
                "pinball_0.5": None if metric is None else _coverage_value(metric, "pinball_0.5"),
                "pinball_0.9": None if metric is None else _coverage_value(metric, "pinball_0.9"),
                "mae_p50": None if metric is None else _coverage_value(metric, "mae"),
                "raw_p10_p90_coverage": None if band is None else _coverage_value(band, "raw80_coverage"),
                "cqr80_coverage": None if band is None else _coverage_value(band, "cqr80_coverage"),
                "cqr90_coverage": None if band is None else _coverage_value(band, "cqr90_coverage"),
                "calibration_small": None if band is None else bool(band["calibration_small"]),
                "n_cal": None if band is None else _coverage_value(band, "n_cal"),
                **leads,
            }
        )

    pooled = conformal_by_id.get("pooled")
    return {
        "mode": mode,
        "failure_time": "CARE event_start (annotated onset)",
        "calibration": "conformal quantiles from turbines other than the held-out asset",
        "n_events": len(event_rows),
        "dropped_overlaps": dropped,
        "events": event_rows,
        "pooled_raw_p10_p90_coverage": None if pooled is None else _coverage_value(pooled, "raw80_coverage"),
        "pooled_cqr80_coverage": None if pooled is None else _coverage_value(pooled, "cqr80_coverage"),
        "pooled_cqr90_coverage": None if pooled is None else _coverage_value(pooled, "cqr90_coverage"),
        "scored_csv": str(scored_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument("--out-dir", type=Path, default=CARE_RESULTS_DIR.parent / "rul" / "care_wfa")
    parser.add_argument("--mode", choices=("gearbox", "drivetrain", "both"), default="both")
    args = parser.parse_args()

    if not GNN_SRC.exists():
        print(f"grid-gnn-faults not found at {GNN_SRC}")
        return 1
    if not (args.farm_dir / "event_info.csv").exists():
        print(f"CARE data missing in {args.farm_dir}")
        return 1

    modes = ["gearbox", "drivetrain"] if args.mode == "both" else [args.mode]
    payload = {}
    for mode in modes:
        payload[mode] = evaluate_mode(mode, args.farm_dir, args.out_dir / mode)
        dest = args.out_dir / f"{mode}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload[mode], indent=2, default=_json_default), encoding="utf-8")
        print(json.dumps(payload[mode], indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
