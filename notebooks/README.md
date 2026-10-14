# Notebooks

Notebooks are **thin drivers** for exploration — keep heavy logic in `src/wind_digital_twin/`.

- [`pinn_heat_toy.ipynb`](pinn_heat_toy.ipynb) — Week 10 heat-equation PINN ablation (requires `pip install -e ".[pinn]"`)
- [`thermal_surrogate.ipynb`](thermal_surrogate.ipynb) — lumped ODE vs NN surrogate speed/accuracy demo (requires `.[pinn]`)
- EDA on EDP / synthetic SCADA (optional)

Do not commit large outputs; write artifacts to `results/` (gitignored). Committed figures under `docs/figures/` (PINN ablation, T06 twin timeline, hybrid vs physics RMSE) are produced by `scripts/run_pinn_heat.py` and `scripts/run_twin_pass.py`.
