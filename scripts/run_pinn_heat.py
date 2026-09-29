#!/usr/bin/env python3
"""Train 1-D heat PINN and physics-off ablation; write comparison figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_digital_twin.config import RESULTS_DIR  # noqa: E402
from wind_digital_twin.pinn.heat1d import (  # noqa: E402
    make_model,
    relative_l2,
    sample_training_data,
    train,
)
from wind_digital_twin.pinn.plots import (  # noqa: E402
    plot_ablation_slices,
    plot_error_heatmaps,
    plot_loss_curves,
)


def run_pinn_heat(
    *,
    seed: int = 42,
    n_ic: int = 25,
    n_bc: int = 25,
    n_data: int = 0,
    n_f: int = 2048,
    adam_steps: int = 5000,
    lbfgs_max_iter: int = 500,
    quick: bool = False,
) -> dict:
    if quick:
        adam_steps = 800
        lbfgs_max_iter = 50
        n_f = 500

    data = sample_training_data(
        n_ic=n_ic,
        n_bc=n_bc,
        n_data=n_data,
        n_f=n_f,
        seed=seed,
    )

    model_pinn = make_model(seed=seed)
    hist_pinn = train(
        model_pinn,
        data,
        w_f=1.0,
        adam_steps=adam_steps,
        lbfgs_max_iter=lbfgs_max_iter,
    )
    l2_pinn = relative_l2(model_pinn)

    model_data = make_model(seed=seed)
    hist_data = train(
        model_data,
        data,
        w_f=0.0,
        adam_steps=adam_steps,
        lbfgs_max_iter=lbfgs_max_iter,
    )
    l2_data = relative_l2(model_data)

    results_dir = RESULTS_DIR / "pinn"
    docs_fig = ROOT / "docs" / "figures" / "pinn_ablation.png"
    plot_ablation_slices(
        model_pinn,
        model_data,
        data,
        l2_pinn=l2_pinn,
        l2_data=l2_data,
        out_path=docs_fig,
    )
    plot_ablation_slices(
        model_pinn,
        model_data,
        data,
        l2_pinn=l2_pinn,
        l2_data=l2_data,
        out_path=results_dir / "pinn_ablation.png",
    )
    plot_error_heatmaps(
        model_pinn,
        model_data,
        out_path=results_dir / "pinn_error_heatmaps.png",
    )
    plot_loss_curves(
        hist_pinn,
        hist_data,
        out_path=results_dir / "pinn_loss_curves.png",
    )

    summary = {
        "relative_l2_pinn": l2_pinn,
        "relative_l2_data_only": l2_data,
        "adam_steps": adam_steps,
        "lbfgs_max_iter": lbfgs_max_iter,
        "n_ic": n_ic,
        "n_bc": n_bc,
        "n_data": n_data,
        "n_f": n_f,
        "seed": seed,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("Relative L2 vs analytical solution:")
    print(f"  PINN (w_f=1):     {l2_pinn:.4e}")
    print(f"  Data-only (w_f=0): {l2_data:.4e}")
    print(f"Figures: {docs_fig}, {results_dir}/")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ic", type=int, default=25)
    parser.add_argument("--n-bc", type=int, default=25)
    parser.add_argument(
        "--n-data",
        type=int,
        default=0,
        help="Noisy interior sensors (0 = IC/BC only)",
    )
    parser.add_argument("--n-f", type=int, default=2048)
    parser.add_argument("--adam-steps", type=int, default=5000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter training for smoke tests",
    )
    args = parser.parse_args()
    run_pinn_heat(
        seed=args.seed,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        n_data=args.n_data,
        n_f=args.n_f,
        adam_steps=args.adam_steps,
        lbfgs_max_iter=args.lbfgs_max_iter,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
