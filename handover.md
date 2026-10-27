# handover.md

**Last updated:** 2026-10-27

Track B capstone **v1.0.0**: hybrid drivetrain digital twin with Week 11 **FastAPI + Streamlit** serving. Gearbox RUL, CQR, and SHAP live in [grid-gnn-faults-2](https://github.com/mehmetertac/grid-gnn-faults-2). IEEE 39-bus GNN demo lives in [grid-gnn-faults-2](https://github.com/mehmetertac/grid-gnn-faults-2) only.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full stack through RUL, conformal, SHAP, service | Done |
| End-to-end twin pass + headline figures | Done (synthetic EDP) |
| **`service/`** — FastAPI, Streamlit, Docker, CI; P10 SHAP attributions in API/UI | Done |
| [docs/figures/twin_dashboard.gif](docs/figures/twin_dashboard.gif) — scrub animation with residual + SHAP | Done |
| README v1.0.0 — hybrid RMSE table, RUL placeholder table, SHAP figure, limitations | Done |
| [docs/figures/shap_sensor_ranking.png](docs/figures/shap_sensor_ranking.png) | Done (synthetic export) |
| Real EDP re-export of serving models + filled RUL metrics table | Pending when data downloaded |
| Release tag **`v1.0.0`** | Tagged locally; push pending |

---

## Suggested next step

1. Re-run `export_models.py` after real EDP download (do not refit ODE in grid-gnn-faults-2).
2. Export scored CSV and fill LOFO RUL / lead-time table in README from grid-gnn-faults-2 CLI.

---

## Key commit

`c1d99a1` — P10 attributions, dashboard figure/GIF refresh (prior: `1b38e11` README v1.0.0).
