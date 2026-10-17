# handover.md

**Last updated:** 2026-10-17

Track B capstone **v1.0.0**: hybrid drivetrain digital twin with Week 11 **FastAPI + Streamlit** serving. Gearbox RUL, CQR, and SHAP live in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults). IEEE 39-bus GNN demo lives in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) only.

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full stack through RUL, conformal, SHAP, service | Done |
| End-to-end twin pass + headline figures | Done (synthetic EDP) |
| **`service/`** — FastAPI, Streamlit, Docker, CI | Done |
| README v1.0.0 — hybrid RMSE table, RUL placeholder table, SHAP figure, limitations | Done |
| [docs/figures/shap_sensor_ranking.png](docs/figures/shap_sensor_ranking.png) | Done (synthetic export) |
| Real EDP re-export of serving models + filled RUL metrics table | Pending when data downloaded |
| Release tag **`v1.0.0`** | Pending push |

---

## Suggested next step

1. Re-run `export_models.py` after real EDP download (do not refit ODE in grid-gnn-faults).
2. Export scored CSV and fill LOFO RUL / lead-time table in README from grid-gnn-faults CLI.

---

## Key commit

(Pending capstone README v1.0.0 polish — prior: `3930389` FastAPI/Streamlit serving.)
