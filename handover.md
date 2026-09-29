# handover.md

**Last updated:** 2026-09-27

Track B capstone scaffold: hybrid drivetrain digital twin (physics thermal + residual + anomaly). Week 10 — PINN heat-equation toy implemented with physics-off ablation.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — taxonomy, fidelity, pipeline diagram, scope | Done |
| Package `wind_digital_twin` — data, physics, residual, anomaly, eval | Done (ported baseline) |
| Scripts: synthetic EDP, download check, gearbox thermal run, **PINN heat run** | Done |
| Tests + pre-commit (file size + pytest) | Done |
| [AGENT.md](AGENT.md) agent rules | Done |
| [docs/reading_notes.md](docs/reading_notes.md) — Raissi 2019 PINN notes + drivetrain transfer + toy spec | Done |
| PINN heat-equation toy + ablation figure | Done ([`pinn/`](src/wind_digital_twin/pinn/), [`run_pinn_heat.py`](scripts/run_pinn_heat.py), [`pinn_heat_toy.ipynb`](notebooks/pinn_heat_toy.ipynb)) |
| Learned residual upgrade | Not started |
| Week 11 RUL + conformal + deployment | Not started |
| [WEEK_10_REFLECTION.md](WEEK_10_REFLECTION.md) | Not started (Friday) |

---

## Repo layout

```
wind-digital-twin/
├── AGENT.md
├── handover.md
├── README.md
├── docs/ARCHITECTURE.md
├── docs/reading_notes.md
├── docs/figures/pinn_ablation.png
├── data/raw/edp/          # EDP CSVs or synthetic (--force generator)
├── src/wind_digital_twin/
│   ├── config.py
│   ├── data/
│   ├── physics/gearbox_thermal.py
│   ├── pinn/heat1d.py, plots.py
│   ├── residual/residual_features.py
│   ├── anomaly/           # physics_hybrid, isolation_forest, scoring
│   └── eval/              # protocol, plots
├── scripts/
├── tests/fixtures/        # tiny EDP-shaped CSVs
├── notebooks/pinn_heat_toy.ipynb
└── results/               # gitignored outputs (incl. results/pinn/)
```

---

## Port provenance (wind-turbine-anomaly → here)

| Source | Destination |
|--------|-------------|
| `models/gearbox_thermal.py` | `physics/gearbox_thermal.py` |
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
| `fit_gearbox_thermal_with_selection`, `GearboxThermalModel` | `physics.gearbox_thermal` | Expected temperature + residuals |
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
pip install -e ".[pinn]"
python scripts/run_pinn_heat.py
pytest tests/ -q
```

---

## Suggested next step

1. Optional inverse variant: recover \(\alpha\) from noisy interior data ([docs/reading_notes.md](docs/reading_notes.md)).
2. Replace or augment linear physics error with a **learned residual model**.
3. Draft `WEEK_10_REFLECTION.md` (uncertainty, leakage, maintenance terms).

---

## Key commit

Pending push — PINN heat-equation module, physics-off ablation, README figure (`docs/figures/pinn_ablation.png`), `[pinn]` extra in `pyproject.toml`.
