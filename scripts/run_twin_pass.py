#!/usr/bin/env python3
"""End-to-end twin pass: SCADA → physics → hybrid residual → anomaly score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_digital_twin.config import DATA_RAW, DEFAULT_BUFFER_DAYS  # noqa: E402
from wind_digital_twin.eval.twin_pass import run_twin_pass  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end hybrid twin pass")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--turbine", default="T06")
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Default: docs/figures (committed headline PNGs)",
    )
    args = parser.parse_args()

    result = run_twin_pass(
        raw_dir=args.data_dir,
        turbine_id=args.turbine,
        buffer_days=args.buffer_days,
        lookback_days=args.lookback_days,
        figures_dir=args.figures_dir,
    )
    print(
        json.dumps(
            {
                "turbine_id": result.turbine_id,
                "synthetic": result.synthetic,
                "threshold": result.threshold,
                "lead_time_days": result.eval_summary.get("lead_time_days"),
                "corrector_metrics": result.corrector_metrics,
                "timeline_figure": str(result.timeline_figure),
                "rmse_figure": str(result.rmse_figure),
                "frame_path": str(result.frame_path),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
