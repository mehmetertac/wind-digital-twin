# handover.md

**Last updated:** 2026-11-26

Track B capstone **v1.0.0**: hybrid drivetrain digital twin with Week 11 **FastAPI + Streamlit** serving. Gearbox RUL, CQR, and SHAP live in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults). IEEE 39-bus GNN demo lives in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) only.

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
| Week 12 README polish — recruiter opener, badges, grid-gnn-faults links, PINN/surrogate sourcing | Done |
| [docs/figures/shap_sensor_ranking.png](docs/figures/shap_sensor_ranking.png) | Done (synthetic export) |
| Real EDP re-export of serving models + filled RUL metrics table | Pending when data downloaded |
| **`CHANGELOG.md`** + CI pandas pin (`pandas<3` for parquet export) | Done |
| **Reproducibility** — `requires-python` `>=3.11,<3.13`, `requirements.lock`, wheels-only README, CI **repro** job (ubuntu + windows) | Done |
| **CARE-to-Compare** — scorer, loader, downloader, hybrid + ablation scripts, tests, `care_wfa_channel_map.csv`; WF-A scores in `results/care/` | Done (local data not committed) |
| Release tag **`v1.0.1`** (portfolio polish; **`v1.0.0`** unchanged on remote) | Pending push |

---

## Suggested next step

1. Improve WF-A CARE (e.g. tuned AE or hybrid P95) vs ablation table in `results/care/wfa_ablations.json` (best so far ~0.58 AE gearbox-only).
2. Add Mendeley v2 per-turbine importer (`zjxjnjp3xs/2`).
3. Re-run `export_models.py` after real EDP download for serving.

---

## Key commit

CARE groundwork — vendored MIT CARE scorer, Zenodo v6 WF-A range download, EDP channel map (oil/bear/power/RPM), loader + baseline script, Mendeley v2 docs.