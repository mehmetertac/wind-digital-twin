# handover.md

**Last updated:** 2026-11-28

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
| Real EDP re-export of serving models + filled RUL metrics table | RUL table filled from CARE WF-A (local EDP SCADA is still synthetic) |
| **`CHANGELOG.md`** + CI pandas pin (`pandas<3` for parquet export) | Done |
| **Reproducibility** — `requires-python` `>=3.11,<3.13`, `requirements.lock`, wheels-only README, CI **repro** job (ubuntu + windows) | Done |
| **CARE-to-Compare** — scorer, loader, downloader, hybrid + ablation scripts, tests, `care_wfa_channel_map.csv` | Done (local data not committed) |
| **WF-A identity-disjoint comparison** — hybrid, repo Isolation Forest, dense AE, LSTM AE; CARE and four sub-scores in the README | Done (hybrid CARE 0.460; dense AE 0.557; below the published 95-event AE of 0.66 on earliness and reliability) |
| **Hybrid CARE ablations** — uncapped-season rise model, one-sided z, multi-target temperatures, max-z and alarm-OR with the dense AE | Done, then superseded by the spline and min-p rows below |
| **Power-spline rise, per-channel Bonferroni p-values, min-p fusion, EWMA rows, 200-draw 90% intervals** | Done (fifteen-channel CARE 0.517, interval 0.000–0.577; dense AE 0.557, interval 0.478–0.612; alarm OR 0.599; gearbox-only CARE 0; event 71 bias 0.06 °C / 0.13 °C) |
| **CARE RUL** — `scripts/run_care_rul.py` leave-one-failure-out on gearbox labels and on generator-bearing + gearbox; P10/P50/P90 lead text, pinball, conformal coverage | Done |
| [docs/EVALUATION.md](docs/EVALUATION.md) — why thresholds and conformal quantiles exclude the test turbine | Done |
| Release tag **`v1.0.1`** (portfolio polish; **`v1.0.0`** unchanged on remote) | Pending push |

---

## Repo layout (evaluation additions)

| Path | Role |
|------|------|
| [src/wind_digital_twin/eval/identity_calibration.py](src/wind_digital_twin/eval/identity_calibration.py) | Chronological fit/cal split, conformal rank, and `conformal_pvalues` |
| [src/wind_digital_twin/eval/care_nbm.py](src/wind_digital_twin/eval/care_nbm.py) | Hybrid, Isolation Forest, dense AE, LSTM AE z-scores, CARE aggregation, event bootstrap |
| [src/wind_digital_twin/eval/care_rul.py](src/wind_digital_twin/eval/care_rul.py) | CARE failure selection, overlap merge, scored-frame export |
| [scripts/run_care_compare.py](scripts/run_care_compare.py) | WF-A comparison CLI, including rise and fusion ablations |
| [src/wind_digital_twin/eval/care_rise_score.py](src/wind_digital_twin/eval/care_rise_score.py) | Power-spline rise, per-channel z, Bonferroni min-p |
| [scripts/run_care_rul.py](scripts/run_care_rul.py) | Quantile RUL CLI (imports grid-gnn-faults) |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Identity-disjoint calibration paragraph |

---

## Suggested next step

1. Add the Mendeley v2 per-turbine importer (`zjxjnjp3xs/2`) so the named T01/T06/T09 gearbox log can be scored on full-farm SCADA rather than CARE windows. On Wind Farm A the fifteen-channel rise still leaves gearbox-only CARE at 0: events 10 and 72 have no alarm inside the window, and event 51 has 11, short of criticality 72. A longer SCADA history is what can test whether those failures have a thermal precursor at all.
2. Re-run `export_models.py` after a real EDP download for serving.

---

## Key commit

Power-spline rise with per-channel Bonferroni p-values: fifteen-channel CARE 0.517 (90% interval 0.000–0.577) against the dense autoencoder at 0.557 (0.478–0.612). Alarm OR reaches 0.599. Gearbox-only CARE stays 0.
