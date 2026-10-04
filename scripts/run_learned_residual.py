#!/usr/bin/env python3
"""Fit learned hybrid thermal correctors and write RMSE comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_digital_twin.config import (  # noqa: E402
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
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
from wind_digital_twin.residual.learned_residual import fit_hybrid_thermal_corrector  # noqa: E402


def run_learned_residual(
    raw_dir: Path = DATA_RAW,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> dict:
    turbines, failures = load_edp_dataset(raw_dir)
    out_dir = RESIDUAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "synthetic": is_synthetic_dataset(raw_dir),
        "turbines": {},
    }

    for turbine_id, raw_df in sorted(turbines.items()):
        df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
        failure = get_failure_for_turbine(turbine_id, failures)
        healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
        healthy_df = df.loc[healthy_mask]
        if len(healthy_df) < 200:
            print(f"  SKIP {turbine_id}: insufficient healthy rows ({len(healthy_df)})")
            continue
        print(f"  {turbine_id}: fitting hybrid corrector...", flush=True)
        _hybrid, metrics = fit_hybrid_thermal_corrector(healthy_df)
        summary["turbines"][turbine_id] = metrics

    out_path = out_dir / "rmse_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned residual RMSE comparison")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    args = parser.parse_args()
    run_learned_residual(raw_dir=args.data_dir, buffer_days=args.buffer_days)


if __name__ == "__main__":
    main()
