# wind-digital-twin

Hybrid **digital twin** capstone for a wind-turbine **gearbox/drivetrain thermal** path: EDP SCADA → lumped thermal ODE → **learned residual corrector** (healthy-only GBM on fused exogenous features) → **hybrid-residual anomaly detector** → maintenance alarms. **Week 11 (this milestone):** end-to-end twin pass, headline figures, and maintenance narrative. **Next:** RUL head, conformal intervals, deployment sketch.

Full taxonomy, fidelity levels, and pipeline diagram: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install

# Optional: PINN heat toy (PyTorch; CI stays torch-free by default)
pip install -e ".[pinn]"
python scripts/run_pinn_heat.py

python scripts/generate_synthetic_edp.py --force
python scripts/download_edp.py --check
python scripts/run_gearbox_thermal.py
python scripts/run_learned_residual.py
python scripts/run_hybrid_anomaly.py --turbine T06
python scripts/run_twin_pass.py --turbine T06
pytest tests/ -q
```

Real EDP data: see [data/README.md](data/README.md) and `python scripts/download_edp.py --instructions`. Re-run `run_twin_pass.py --data-dir <path>` after download; figures label **real EDP** automatically.

## Architecture (shipped pipeline)

```mermaid
flowchart LR
    scada["EDP SCADA 10-min"] --> clean["clean_turbine_df"]
    clean --> ode["Lumped thermal ODE"]
    ode --> gbm["GBM residual corrector"]
    gbm --> det["Hybrid-residual Isolation Forest"]
    det --> alarm["Sustained alarm and lead time"]
```

One command runs the full path and writes the headline timeline: [`scripts/run_twin_pass.py`](scripts/run_twin_pass.py) → [`eval/twin_pass.py`](src/wind_digital_twin/eval/twin_pass.py).

## End-to-end twin timeline (T06)

**Synthetic EDP** (default local data): four aligned panels around the logged gearbox-bearing failure — SCADA temperatures, physics vs hybrid **expected** bearing temperature (same ODE backbone), physics vs hybrid **residuals**, and hybrid-residual anomaly score with threshold and alarm spans. Gray band: 90-day training holdout before failure.

![T06 hybrid twin timeline around failure](docs/figures/twin_timeline_T06.png)

## Why physics-informed? (PINN ablation)

Week 10 guardrail: a 1-D heat equation \(u_t = \alpha u_{xx}\) on \([0,1]^2\) with sparse **initial and boundary** labels only (no interior sensors). A plain MLP fits those points; with **physics loss off** (`w_f=0`) it does not recover the true decay in the interior (relative L2 **~2.7×10⁻¹** vs analytical). The same architecture with a **PDE residual** at collocation points (`w_f=1`) matches the analytical solution (**~2.7×10⁻³** relative L2).

![PINN vs data-only vs analytical heat solution](docs/figures/pinn_ablation.png)

That pattern is what we want on SCADA: sparse temperature observations plus a **thermal ODE/physics consistency** term, not interpolation between sensors. Code: [`src/wind_digital_twin/pinn/`](src/wind_digital_twin/pinn/), [`scripts/run_pinn_heat.py`](scripts/run_pinn_heat.py), notebook [`notebooks/pinn_heat_toy.ipynb`](notebooks/pinn_heat_toy.ipynb). Paper notes: [docs/reading_notes.md](docs/reading_notes.md).

## Hybrid vs physics-only (healthy validation)

On **healthy** rows only (time-ordered 80/20 split), the GBM corrector fits systematic ODE error from exogenous regime features — not gear temperatures. Typical synthetic T06 holdout: oil RMSE **~2.26 °C → ~0.40 °C** (~82% reduction); bear **~2.37 °C → ~0.44 °C** (~81% reduction). Reproduce: `python scripts/run_twin_pass.py`.

![Physics ODE vs hybrid validation RMSE](docs/figures/hybrid_vs_physics_rmse.png)

Pre-failure **degradation** shows in the timeline residuals and anomaly score, not in this healthy RMSE chart.

## ODE surrogate speedup (demo)

Small NN surrogate for batched lumped-ODE trajectories ([`thermal_surrogate.py`](src/wind_digital_twin/surrogate/thermal_surrogate.py), notebook [`notebooks/thermal_surrogate.ipynb`](notebooks/thermal_surrogate.ipynb)). **Not** the SCADA scoring path — illustration only. Requires `pip install -e ".[pinn]"`.

| Metric | Value (local benchmark) |
|--------|-------------------------|
| ODE seconds / trajectory | ~1.46×10⁻³ |
| NN seconds / trajectory | ~2.43×10⁻⁶ |
| Speedup | **~600×** |

## What this tells an O&M planner (vs a SCADA alarm list)

A **SCADA alarm list** fires on fixed or fleet thresholds: “bearing temperature high.” That conflates **normal hot operation** at high load with **extra heat the operating point does not explain**.

This twin answers three questions:

1. **Expected behavior:** Given power, RPM, and nacelle temperature, what should oil/bearing temperature be (lumped thermal ODE)?
2. **Systematic model gap:** On past healthy operation, where does physics systematically miss (GBM on exogenous features only)?
3. **Degradation signal:** Is the **hybrid residual** persistently worse than on healthy history, with an Isolation Forest score above a threshold fit on a **healthy validation tail** (not the 90 days before failure)?

Outcome: a **sustained alarm with lead time in days** before logged gearbox failure — advisory for inspection planning, not turbine control.

## Deployment dashboard (Week 11)

Probabilistic gearbox RUL and the Week 10 scoring path are served from [`service/`](service/): FastAPI on **:8000**, Streamlit on **:8501**. Models in [`service/models/`](service/models/) are **synthetic smoke** weights (regenerate with `python service/scripts/export_models.py`).

```powershell
pip install -e ".[serve]"
pip install -r service/dashboard/requirements.txt
python service/scripts/export_models.py   # if models/ missing
$env:TWIN_MODELS_DIR = "service/models"
uvicorn twin_service.api.app:app --reload --port 8000
# separate terminal:
$env:API_URL = "http://127.0.0.1:8000"
streamlit run service/dashboard/app.py
```

Or `docker compose -f service/docker-compose.yml up --build`.

| Endpoint | Role |
| --- | --- |
| `POST /twin/score` | SCADA window → physics-expected oil temp, hybrid residual, anomaly score |
| `POST /twin/rul` | Daily features → P10/P50/P90, CQR band, top-five sensor attributions |
| `GET /twin/series` | Precomputed failure-window timeline for the dashboard scrubber |

Gearbox RUL training, LOFO conformal evaluation, and TreeSHAP live in [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults). Week 11 reflection: [WEEK_11_REFLECTION.md](https://github.com/mehmetertac/grid-gnn-faults/blob/main/WEEK_11_REFLECTION.md).

![Twin dashboard — T06 failure window scrub](docs/figures/twin_dashboard.png)

## Limitations (honest)

- **No vibration / CMS** in EDP — thermal SCADA only; envelope and band energy often lead temperature by days to weeks on real gearboxes.
- **Single thermal failure mode** — one score on oil and HSS bearing temperature; not a classifier for pump vs bearing vs noise (T01/T06/T09 are different root causes in the log).
- **10-minute SCADA** — sub-hour transients are averaged; τ may pile at the floor when series are quasi-steady.
- **Open-loop ODE** — state resets on long gaps; sustained hot spells remain visible in residuals by design.
- **Figures in this README** use **synthetic EDP** unless you re-run `run_twin_pass.py` on a downloaded real dataset.

## Layer summary

- **Physics layer:** Lumped thermal ODE with time constant τ on healthy data.
- **Learned residual:** HGBM (optional LightGBM via `WIND_DT_USE_LIGHTGBM=1`) on healthy physics error.
- **Detector:** Isolation Forest on hybrid residual windows plus θ / bear–oil context; [`physics_hybrid`](src/wind_digital_twin/anomaly/physics_hybrid.py) remains the Week 6 physics-only baseline.
- **Evaluation:** Training excludes the **90 days before failure**; metrics are **time-ordered**.

Agent and contribution rules: [AGENT.md](AGENT.md). Current status: [handover.md](handover.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
