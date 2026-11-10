# Week 12 / Day-90 reflection

**Date:** 2026-11-06

---

## What I built in twelve weeks — nine public repos (honest state)

These nine match the study plan’s “Day 90” list. Each shipped as a public GitHub artifact with at least a README and runnable code; polish and “recruiter clickability” vary.

1. **energy-ts-fundamentals** — Week 1: ENTSO-E load forecasting, classical baselines (seasonal naive, ETS, SARIMA), rolling-origin backtest table. Narrow, foundational; not a headline pin.

2. **energy-feature-pipeline** — Week 2: NWP/ERA5 + OPSD feature joins, weather-vs-model lift notebooks, nested package layout. Real data plumbing; not polished as a one-click portfolio hero.

3. **wind-quantile-forecast** — Week 3: quantile LightGBM/XGB/CatBoost on German wind, pinball, calibration, SHAP. Short June 2019 sample; illustrative, not a multi-year production study.

4. **wind-turbine-anomaly** — Week 6: pure-ML vs physics-residual hybrid benchmark, trade-off curves, T06 case study, blog. **Synthetic EDP** for pipeline dev; the numbers in the README are reproducible, not a claim on live site SLAs. Kept as a **separate pin** (article #1 evidence), not one of the three “polish” repos, because the story overlaps the twin.

5. **battery-spo-rl** — Week 7: MAE vs SPO+ on the same LP battery arbitrage loop, dollar regret vs perfect foresight, pandapower OPF add-on. **MAE 9.38 vs SPO 9.40 EUR/MWh** but **EUR 9,198** cumulative regret win for SPO; PPO is in the repo with a fair **not** day-ahead information-set caveat. **v1.0.0** class release with article seed doc.

6. **energy-forecasting-service** — Week 9: TFT (from the hierarchy sibling) through **MLflow, Docker, FastAPI, CI, Streamlit** — the “repo recruiters click first.” MinT reconciliation stays in **multi-site-solar-hierarchy**; the service ships global plant TFT serving.

7. **wind-digital-twin** — Weeks 10–11: capstone **v1.0.0** — lumped thermal ODE, GBM healthy residual correction, hybrid-residual Isolation Forest, twin pass + figures, **FastAPI + Streamlit** Week 11 layer. Healthy RMSE table filled on **synthetic EDP**; RUL LOFO metrics committed from `run_twin.py --fleet` + `evaluate_quantile_rul.py` — **P10 first below 14 d** is **`none`** on all four events (no P10 &lt; 14 d in LOFO windows on that run).

8. **grid-gnn-faults** — Week 11: gearbox RUL labels, quantiles, CQR, TreeSHAP, export contract for the twin; **IEEE 39-bus** fault localization GCN vs MLP — the GCN **underperforms** the MLP on the published table (accuracy **0.662** vs **0.956**; mean hop error **0.85** vs **0.06**), which is an honest “intro GNN, not SOTA” result. **v0.1.0** tagged.

9. **scada-rag-copilot** — Week 12: grounded SCADA alarm → retrieve → **Ollama** structured causes with **citation gate**; twelve synthetic scenarios, **83%** top-1 / top-3 / grounding in the README eval. **v0.1.0** — **no agent tools beyond retrieval** (handover closes that sub-track).

### Also built, not in the “nine” count

Weeks **4**, **5**, and **8** live in sibling repos so the arc is not erased: **dl-vs-gbm-energy** (LightGBM beat DeepAR / N-HiTS / N-BEATS on the published head-to-head table), **multi-site-solar-hierarchy** (TFT + MinT coherence), **energy-causal-conformal** (synthetic control + conformal forecast notebooks, plus a compressed Bayesian GP appendix). They are part of the narrative; they are not the nine names the plan listed for Day 90.

---

## What is still fuzzy

**From the plan’s known gaps**

- **Bayesian depth** — intentionally thin: one GPyTorch solar GP notebook as contrast to conformal intervals (“production desks need coverage frequency, not just posterior width”). No thesis-level GP or variational Bayes.
- **“DL from scratch”** — not a goal: all deep models are **library** models (PyTorch / PyTorch Geometric / darts / etc.). Week 4’s own table says **LightGBM-quantile** won the head-to-head on the published run.

**Mine**

- **RUL on few failures** — four gearbox log lines on three turbines; LOFO and CQR are demonstration contracts, not fleet guarantees; README **P10 first below 14 d** is **`none`** on the synthetic fleet export (not integer lead times).
- **GNN dynamics** — the grid repo is **static** bus classification on simulated snapshots, not a dynamic stability or cascading-outage model; the GCN lost to an MLP on the shipped metrics.
- **Real EDP in the scoring path** — figures default to **synthetic EDP** until downloads and re-runs land; anyone comparing to Hack-the-Wind numbers must rerun on real CSVs.

---

## Which weeks slipped, and what the plan said to drop

- **Week 7** — the plan’s escape hatch was: if dense, **drop RL**, keep LP + SPO + pandapower; **never drop SPO**. RL **did** ship (PPO with honest caveats); SPO stayed the decision-focused headline.
- **Week 10** — PINN stayed a **small 1-D heat toy** (relative L2 **0.274** with physics loss off vs **0.00270** with PDE residual) — read two papers max, build the demo, move on. That matches the plan.
- **Week 12** — LLM/RAG was **optional** if capstone polish was behind; the twin was already **v1.0.0**, so **scada-rag-copilot** shipped and then **stopped** (no tool-use extension per handover).

The real slip is **field data**, not a skipped week: synthetic fleet LOFO is in the README (**P10 lead `none`**); **real** EDP re-run and field-grade calibration are what might move integer lead times — not new architecture.

---

## Three artifacts I would defend hardest in an interview

1. **wind-digital-twin** — Because it states the **EE question** (“should it be this hot?”) in code: ODE + learned residual + detector + **served** API and dashboard — and the README says **what is not** in the box (vibration, CMS, fleet SLAs) before an interviewer asks.

2. **battery-spo-rl** — Because **MAE went the wrong way** and **regret went the right way** — EUR **9,198** — on the same features and battery LP; that is the “optimize the decision, not the leaderboard” story in one table.

3. **energy-forecasting-service** — Because **probabilistic** forecasts survive **Docker, MLflow registry, batch job, and API** — most notebooks never make that trip.

**wind-turbine-anomaly** is the **evidence graph** behind article #1 (threshold curve, case study); I would point to it for charts, not as a substitute for MLOps proof.

---

## First thing I would build in Month 4

**No new repo.** Re-run the fleet path on **real** EDP when downloaded so **P10 first below 14 d** might show integer warnings; synthetic export is already in the README with **`none`** where P10 never crossed 14 days. Month 4 is real data + honest integers, not tuning for a prettier cell.

---

## Related

- Article #1 draft: [docs/from-the-turbine-floor.md](docs/from-the-turbine-floor.md)
- Twin status: [handover.md](handover.md)
