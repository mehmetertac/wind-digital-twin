#!/usr/bin/env python3
"""CARE ablations: gearbox-only, P95, persistence, minimal 86-dim AE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_digital_twin.config import (  # noqa: E402
    CARE_RESULTS_DIR,
    DATA_RAW_CARE_WFA,
    DEFAULT_PERSISTENCE_SAMPLES,
)
from wind_digital_twin.data.care_channels import load_frozen_channel_map  # noqa: E402
from wind_digital_twin.data.load_care import iter_care_events  # noqa: E402
from wind_digital_twin.eval.care_hybrid_eval import (  # noqa: E402
    CareStrategy,
    predict_hybrid_alarms,
    predict_minimal_ae_alarms,
    run_care_strategy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument("--out", type=Path, default=CARE_RESULTS_DIR / "wfa_ablations.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run subset of strategy names (default: all)",
    )
    args = parser.parse_args()

    if not (args.farm_dir / "event_info.csv").exists() and not (args.farm_dir / "datasets" / "event_info.csv").exists():
        print(f"CARE data missing in {args.farm_dir}; run scripts/download_care.py")
        return 1

    channel_map = load_frozen_channel_map()
    events = list(iter_care_events(args.farm_dir))

    strategies = [
        (
            CareStrategy(
                name="hybrid_p99_baseline",
                description="Hybrid-residual, P99, pointwise (replicates first run)",
                gearbox_only=False,
            ),
            lambda ds: predict_hybrid_alarms(ds, channel_map, percentile=99.0, persistence_samples=0),
        ),
        (
            CareStrategy(
                name="hybrid_gearbox_only",
                description="Hybrid-residual P99; anomaly events with gearbox in description only (3 events)",
                gearbox_only=True,
            ),
            lambda ds: predict_hybrid_alarms(ds, channel_map, percentile=99.0, persistence_samples=0),
        ),
        (
            CareStrategy(
                name="hybrid_p95",
                description="Hybrid-residual with P95 threshold",
                gearbox_only=False,
            ),
            lambda ds: predict_hybrid_alarms(ds, channel_map, percentile=95.0, persistence_samples=0),
        ),
        (
            CareStrategy(
                name="hybrid_persist_6",
                description=f"Hybrid-residual P99 + {DEFAULT_PERSISTENCE_SAMPLES}-sample persistence episodes",
                gearbox_only=False,
            ),
            lambda ds: predict_hybrid_alarms(
                ds,
                channel_map,
                percentile=99.0,
                persistence_samples=DEFAULT_PERSISTENCE_SAMPLES,
            ),
        ),
        (
            CareStrategy(
                name="minimal_ae_86",
                description="Sklearn MLP autoencoder on all numeric features, P99 recon error",
                gearbox_only=False,
            ),
            lambda ds: predict_minimal_ae_alarms(ds, percentile=99.0, persistence_samples=0, seed=args.seed),
        ),
        (
            CareStrategy(
                name="minimal_ae_86_gearbox_only",
                description="MLP AE P99; gearbox-labelled anomaly events only",
                gearbox_only=True,
            ),
            lambda ds: predict_minimal_ae_alarms(ds, percentile=99.0, persistence_samples=0, seed=args.seed),
        ),
    ]

    selected_names = set(args.only) if args.only else {s.name for s, _ in strategies}

    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        results = [r for r in prior.get("strategies", []) if r["summary"]["name"] not in selected_names]
    else:
        results = []

    for strategy, predict_fn in strategies:
        if strategy.name not in selected_names:
            continue
        print(f"Running {strategy.name}...", flush=True)
        results.append(run_care_strategy(events, strategy, predict_fn))
        payload = {"farm_dir": str(args.farm_dir), "strategies": results}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    payload = {"farm_dir": str(args.farm_dir), "strategies": results}
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())