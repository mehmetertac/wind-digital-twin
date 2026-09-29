# Notebooks

Notebooks are **thin drivers** for exploration — keep heavy logic in `src/wind_digital_twin/`.

- [`pinn_heat_toy.ipynb`](pinn_heat_toy.ipynb) — Week 10 heat-equation PINN ablation (requires `pip install -e ".[pinn]"`)
- EDA on EDP / synthetic SCADA (optional)

Do not commit large outputs; write artifacts to `results/` (gitignored). The headline ablation figure is committed under `docs/figures/`.
