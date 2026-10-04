# handover.md

**Last updated:** 2026-10-02

Track B capstone: hybrid drivetrain digital twin (lumped ODE thermal + learned residual + anomaly). Week 10 capstone build: fusion features, GBM residual corrector, hybrid-residual detector, ODE surrogate notebook.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — taxonomy, fidelity, pipeline diagram, scope | Done |
| Package `wind_digital_twin` — data, physics, residual, anomaly, eval, surrogate | Done |
| **Lumped thermal ODE backbone** — [`lumped_ode.py`](src/wind_digital_twin/physics/lumped_ode.py), optional [`lumped_ode_correction.py`](src/wind_digital_twin/physics/lumped_ode_correction.py) | Done |
| **Fusion features** — [`fusion_features.py`](src/wind_digital_twin/residual/fusion_features.py), optional EDP columns in [`clean.py`](src/wind_digital_twin/data/clean.py) | Done |
| **Learned residual corrector** — [`learned_residual.py`](src/wind_digital_twin/residual/learned_residual.py) (HGBM default; optional LightGBM via `WIND_DT_USE_LIGHTGBM=1`) | Done |
| **Hybrid-residual detector** — [`hybrid_residual_detector.py`](src/wind_digital_twin/anomaly/hybrid_residual_detector.py); Week 6 baseline unchanged in [`physics_hybrid.py`](src/wind_digital_twin/anomaly/physics_hybrid.py) | Done |
| **ODE surrogate demo** — [`surrogate/thermal_surrogate.py`](src/wind_digital_twin/surrogate/thermal_surrogate.py), [`notebooks/thermal_surrogate.ipynb`](notebooks/thermal_surrogate.ipynb) | Done |
| Scripts: synthetic EDP, download check, gearbox thermal, learned residual RMSE, T06 detector spot-check | Done |
| Tests + pre-commit (file size + pytest) | Done |
| [AGENT.md](AGENT.md) agent rules | Done |
| [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md) — inventory + shipped fusion/corrector split | Done |
| PINN heat-equation toy + ablation figure | Done |
| [WEEK_10_REFLECTION.md](WEEK_10_REFLECTION.md) | Done |
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
├── docs/SENSOR_FUSION.md
├── data/raw/edp/
├── src/wind_digital_twin/
│   ├── data/
│   ├── physics/
│   ├── pinn/
│   ├── residual/          # residual_features, fusion_features, learned_residual
│   ├── anomaly/           # physics_hybrid, hybrid_residual_detector, IF
│   ├── surrogate/thermal_surrogate.py
│   └── eval/
├── scripts/
│   run_learned_residual.py, run_hybrid_anomaly.py, ...
├── tests/
├── notebooks/pinn_heat_toy.ipynb, thermal_surrogate.ipynb
└── results/               # physics_thermal/, residual/
```

---

## Core module API

| Symbol | Module | Purpose |
|--------|--------|---------|
| `resolve_feature_columns`, `clean_turbine_df` | `data.clean` | Core + optional fusion columns |
| `build_fusion_feature_frame`, `build_corrector_matrix` | `residual.fusion_features` | θ ratios, detector context, corrector inputs |
| `fit_hybrid_thermal_corrector`, `HybridThermalCorrector` | `residual.learned_residual` | ODE + GBM on healthy physics error |
| `fit_physics_hybrid` | `anomaly.physics_hybrid` | Week 6 IF on physics residuals |
| `fit_hybrid_residual_detector` | `anomaly.hybrid_residual_detector` | IF on hybrid residuals + fusion context |
| `build_surrogate_dataset`, `train_surrogate`, `benchmark_surrogate_speed` | `surrogate.thermal_surrogate` | ODE surrogate demo |

**CLI**

```powershell
python scripts/generate_synthetic_edp.py --force
python scripts/run_gearbox_thermal.py
python scripts/run_learned_residual.py
python scripts/run_hybrid_anomaly.py --turbine T06
pip install -e ".[pinn]"
python notebooks/thermal_surrogate.ipynb   # or run cells locally
pytest tests/ -q
```

---

## Suggested next step

1. Re-fit on **real** EDP when downloaded; compare hybrid vs physics RMSE in `results/residual/rmse_comparison.json`.
2. Week 11: RUL head, conformal intervals on scores, deployment sketch.

---

## Key commit

`ad2fb4e` — Learned residual corrector, hybrid-residual detector, ODE surrogate notebook, fusion features, CLIs, tests, docs.
