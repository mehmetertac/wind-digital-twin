# handover.md

**Last updated:** 2026-10-13

Track B capstone: hybrid drivetrain digital twin (lumped ODE thermal + learned residual + anomaly) with Week 11 **FastAPI + Streamlit** serving. Gearbox RUL, CQR, and SHAP live in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults).

---

## What is done

| Item | Status |
|------|--------|
| Git remote `origin` → [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) | Done |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — taxonomy, fidelity, pipeline diagram | Done |
| Package `wind_digital_twin` — data, physics, residual, anomaly, eval, surrogate | Done |
| End-to-end twin pass — [`eval/twin_pass.py`](src/wind_digital_twin/eval/twin_pass.py), [`scripts/run_twin_pass.py`](scripts/run_twin_pass.py) | Done |
| Headline figures — twin timeline, hybrid RMSE, PINN ablation | Done (synthetic EDP) |
| **`service/`** — `twin_service` FastAPI, Streamlit dashboard, Docker, CI | Done |
| **`service/models/`** — synthetic smoke pipeline, RUL bundle, CQR offsets, SHAP CSV, timeline parquet | Done |
| Dashboard README assets — [`docs/figures/twin_dashboard.png`](docs/figures/twin_dashboard.png), [`.gif`](docs/figures/twin_dashboard.gif) | Done |
| Tests — root + `service/tests/` (API smokes, dashboard client); 64 passed locally 2026-10-13 | Done |
| Real EDP re-export of serving models | Pending when data downloaded |

---

## Repo layout

```
wind-digital-twin/
├── AGENT.md
├── handover.md
├── README.md
├── .github/workflows/ci.yml
├── docs/figures/
├── service/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── models/              # committed smoke artifacts
│   ├── scripts/export_models.py
│   ├── dashboard/
│   └── src/twin_service/
├── src/wind_digital_twin/
├── scripts/
└── tests/
```

---

## Core module API

| Symbol | Module | Purpose |
|--------|--------|---------|
| `run_twin_pass`, `build_twin_layer_frame` | `eval.twin_pass` | SCADA → layers + figures |
| `fit_hybrid_residual_detector` | `anomaly.hybrid_residual_detector` | IF on hybrid residuals |
| `create_app`, `load_artifacts` | `twin_service.api.app`, `twin_service.artifacts` | FastAPI + model load |
| `score_window`, `predict_rul` | `twin_service.inference` | `/twin/score`, `/twin/rul` |

**CLI**

```powershell
pip install -e ".[dev,serve,export]"
python scripts/generate_synthetic_edp.py --force
python service/scripts/export_models.py
pytest tests/ service/tests/ -q
uvicorn twin_service.api.app:app --port 8000
```

---

## Suggested next step

1. Re-run `export_models.py` after real EDP download (do not refit ODE in grid-gnn-faults).
2. IEEE 39-bus GNN demo remains in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) only.

---

## Key commit

*(Update after push with Week 11 deployment commit hash.)*
