# handover.md

**Last updated:** 2026-10-01

Track B capstone: hybrid drivetrain digital twin (lumped ODE thermal + residual + anomaly). Week 10 — ODE backbone shipped; PINN toy done; [WEEK_10_REFLECTION.md](WEEK_10_REFLECTION.md) written.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — taxonomy, fidelity, pipeline diagram, scope | Done |
| Package `wind_digital_twin` — data, physics, residual, anomaly, eval | Done |
| **Lumped thermal ODE backbone** — [`lumped_ode.py`](src/wind_digital_twin/physics/lumped_ode.py), optional [`lumped_ode_correction.py`](src/wind_digital_twin/physics/lumped_ode_correction.py) | Done |
| Scripts: synthetic EDP, download check, gearbox thermal run (`--nn-correction`), PINN heat run | Done |
| Tests + pre-commit (file size + pytest) | Done |
| [AGENT.md](AGENT.md) agent rules | Done |
| [docs/reading_notes.md](docs/reading_notes.md) — Raissi + Pujana deep reads, abstract skims, PINN toy spec | Done |
| [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md) — EDP channel inventory + θ/ratio feature spec for learned residual | Done |
| PINN heat-equation toy + ablation figure | Done ([`pinn/`](src/wind_digital_twin/pinn/), [`run_pinn_heat.py`](scripts/run_pinn_heat.py), [`pinn_heat_toy.ipynb`](notebooks/pinn_heat_toy.ipynb)) |
| [WEEK_10_REFLECTION.md](WEEK_10_REFLECTION.md) | Done |
| Learned residual upgrade | Not started |
| Week 11 RUL + conformal + deployment | Not started |

---

## Repo layout

```
wind-digital-twin/
├── AGENT.md
├── handover.md
├── WEEK_10_REFLECTION.md
├── README.md
├── docs/ARCHITECTURE.md
├── docs/reading_notes.md
├── docs/SENSOR_FUSION.md
├── docs/figures/pinn_ablation.png
├── data/raw/edp/          # EDP CSVs or synthetic (--force generator)
├── src/wind_digital_twin/
│   ├── config.py
│   ├── data/
│   ├── physics/gearbox_thermal.py, lumped_ode.py, lumped_ode_correction.py
│   ├── pinn/heat1d.py, plots.py
│   ├── residual/residual_features.py
│   ├── anomaly/           # physics_hybrid, isolation_forest, scoring
│   └── eval/              # protocol, plots
├── scripts/
├── tests/                 # incl. thermal_fixtures.py, test_lumped_ode.py
├── notebooks/pinn_heat_toy.ipynb
└── results/               # gitignored outputs (incl. results/physics_thermal/)
```

---

## Port provenance (wind-turbine-anomaly → here)

| Source | Destination |
|--------|-------------|
| `models/gearbox_thermal.py` | `physics/gearbox_thermal.py` (**upgraded to ODE**) |
| `models/residual_features.py` | `residual/residual_features.py` |
| `models/physics_hybrid.py` | `anomaly/physics_hybrid.py` |
| `models/isolation_forest.py`, `scoring.py` | `anomaly/` |
| `data/load_edp.py`, `clean.py`, `synthetic_edp.py` | `data/` |
| `eval/protocol.py` | `eval/protocol.py` |
| `eval/plots.py` (gearbox residual plot only) | `eval/plots.py` |
| ML baselines, SHAP, threshold sweep | **Stay in anomaly repo** |

---

## Core module API

| Symbol | Module | Purpose |
|--------|--------|---------|
| `load_edp_dataset` | `data.load_edp` | SCADA dict + gearbox failures |
| `clean_turbine_df`, `healthy_training_mask` | `data.clean` | Operating filter + healthy mask |
| `generate_synthetic_edp_dataset` | `data.synthetic_edp` | Local/CI data |
| `fit_lumped_thermal`, `simulate_temperature`, `LumpedThermalParams` | `physics.lumped_ode` | ODE fit + open-loop predict |
| `fit_gearbox_thermal_with_selection`, `GearboxThermalModel`, `validate_thermal_model` | `physics.gearbox_thermal` | Expected temperature + residuals + τ checks |
| `build_residual_feature_frame` | `residual.residual_features` | EWMA + rolling degradation features |
| `fit_physics_hybrid`, `PhysicsHybridPipeline` | `anomaly.physics_hybrid` | Thermal + IF detector |
| `evaluate_turbine`, `detect_alarm_episodes` | `eval.protocol` | Lead time / false alarms |
| `train`, `sample_training_data`, `relative_l2`, `MLP` | `pinn.heat1d` | 1-D heat PINN + metrics |
| `plot_ablation_slices`, `plot_error_heatmaps` | `pinn.plots` | Ablation figures |

**CLI**

```powershell
python scripts/generate_synthetic_edp.py --force
python scripts/download_edp.py --check
python scripts/run_gearbox_thermal.py
python scripts/run_gearbox_thermal.py --nn-correction   # optional; needs torch
pip install -e ".[pinn]"
python scripts/run_pinn_heat.py
pytest tests/ -q
```

---

## Suggested next step

1. Implement fusion columns + `build_fusion_feature_frame()` per [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md); then **learned residual model**.
2. Re-fit ODE on **real** EDP when downloaded; compare `ode_val_rmse` vs `linear_val_rmse` in `results/physics_thermal/`.
3. Week 11: RUL, conformal intervals, deployment sketch.

---

## Key commit

*(Update on next push with ODE backbone commit hash.)*
