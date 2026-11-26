#!/usr/bin/env python3
"""CARE score for the hybrid-residual detector on Wind Farm A event windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_digital_twin.config import CARE_RESULTS_DIR, DATA_RAW_CARE_WFA  # noqa: E402
from wind_digital_twin.data.care_channels import load_frozen_channel_map  # noqa: E402
from wind_digital_twin.data.load_care import iter_care_events  # noqa: E402
from wind_digital_twin.eval.care_hybrid_eval import (  # noqa: E402
    CareStrategy,
    predict_hybrid_alarms,
    run_care_strategy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument(
        "--out",
        type=Path,
        default=CARE_RESULTS_DIR / "wfa_hybrid_residual.json",
    )
    args = parser.parse_args()

    if not (args.farm_dir / "event_info.csv").exists() and not (args.farm_dir / "datasets" / "event_info.csv").exists():
        print(f"CARE data missing in {args.farm_dir}; run scripts/download_care.py")
        return 1

    channel_map = load_frozen_channel_map()
    strategy = CareStrategy(
        name="hybrid_residual",
        description="Hybrid-residual P99 pointwise",
        gearbox_only=False,
    )
    events = list(iter_care_events(args.farm_dir))
    result = run_care_strategy(
        events,
        strategy,
        lambda ds: predict_hybrid_alarms(ds, channel_map),
    )
    payload = {
        "farm_dir": str(args.farm_dir),
        "detector": "hybrid_residual (per-event fit on CARE train normal, THERMAL_MIN_POWER_KW)",
        "summary": result["summary"],
        "events": result["events"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())