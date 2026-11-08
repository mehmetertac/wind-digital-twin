# From the turbine floor to the model: physics-informed anomaly detection with field intuition

Figures and numbers come from the [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) and [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) READMEs.

---

## 1. What the alarm list does not tell you

Picture the SCADA banner: “bearing temperature high.” On the EDP open-data wind farm, turbine **T06** logs **gearbox bearing damage** on **17 October 2017**. The alarm never answers the question an electrical engineer asks first: is this bearing **as hot as it ought to be** for the power, speed, and nacelle temperature it is carrying?

Absolute bearing temperature rises with load and with a hot enclosure. It does not separate a windy summer afternoon from extra friction in the HSS bearing. What the list lacks is **load-conditioned expectation**. Field intuition is knowing weather-and-work heat from wear-and-friction heat. The work below encodes that as *should it be this hot right now?* for the same 10‑minute SCADA the alarm server already sees.

---

## 2. Why pure multivariate ML on SCADA misfires

The Week 6 benchmark compares Isolation Forest, dense and LSTM autoencoders, and a **physics-residual hybrid** at the **99th-percentile** threshold: lead time to logged gearbox failure versus false alarms per turbine-year.

**Isolation Forest** reaches **18** days on T01 pump damage and **48** on T06 bearing, at **106** false alarms per turbine-year. A **dense autoencoder** hits **73** days at **90** false alarms. The best pure-ML row, **LSTM autoencoder**, delivers **80** days at **31** false alarms. The **physics-residual hybrid** lands at **65** days at **8.5** false alarms. Per event, the hybrid trades lead time for operability: T01 **53** vs LSTM **88** days; T06 **77** vs **72**. The headline is lower false-alarm rate at a still-usable horizon.

Multivariate detectors learn training joint patterns. When high power and a hot nacelle imply high bearing temperature, raw novelty flags **healthy peak production**. The benchmark uses **synthetic EDP**; re-run on real CSVs before external comparison. The structural lesson stands: **raw-SCADA novelty is a poor fault proxy** when operating point dominates variance.

![Week 6 trade-off: lead time vs false-alarm rate as threshold varies; hybrid sits in the lower-left operable region.](https://github.com/mehmetertac/wind-turbine-anomaly/raw/main/docs/assets/threshold_tradeoff.png)

*The hybrid occupies the lower-left operable region — fewer false alarms at comparable sensitivity — with the 99th-percentile point marked.*

---

## 3. The physics-residual idea in plain language

The hybrid asks whether **measured temperature** matches **expected temperature** given what the turbine is doing. Expected oil and bearing temperatures are functions of power, speed, and nacelle temperature. The **residual** is **predicted minus actual**; sustained extra heat drives residual drift. On T06 the annotated case shows roughly **−4 °C** over **90** days before failure — visible degradation that raw temperature blurs.

Alarm on the residual stream so a high-load day is not a fault by default. **Subtract weather and work first; the systematic gap is mechanical signal.** This is not a finite-element gearbox model; it is the right coordinate system for SCADA-only thermal surveillance.

---

## 4. From Week 6 charts to the Week 10–11 twin

Week 6 ships the T06 four-panel case study — prediction, residual drift, score, and **77** days from first alarm to logged failure.

![T06 annotated case study from Week 6: power and temperatures, prediction, residual drift, score, and timeline to failure.](https://github.com/mehmetertac/wind-turbine-anomaly/raw/main/docs/assets/case_study_T06.png)

*Residual drift and hybrid score tie to first sustained alarm and **77** days of warning before the logged bearing event.*

The [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) capstone adds a **lumped thermal ODE**, a **GBM residual corrector** on **healthy** rows (exogenous features only), and **Isolation Forest** on hybrid residual windows. On **synthetic EDP** healthy holdout, **T06** oil RMSE falls **2.26 → 0.40 °C** (**82%**); bearing **2.37 → 0.44 °C** (**81%**). Pre-failure degradation appears on the **timeline**, not in the healthy RMSE table.

![End-to-end twin timeline on synthetic EDP around T06 failure: expected temps, residuals, anomaly score, and alarms.](docs/figures/twin_timeline_T06.png)

*Hybrid expectations, residual separation from physics-only, and anomaly scoring through the failure window.*

![Oil and bearing RMSE: physics-only ODE versus hybrid on healthy validation rows.](docs/figures/hybrid_vs_physics_rmse.png)

*Hybrid RMSE stays far below physics-only on healthy rows — ODE bias learned without using gear temps as inputs.*

---

## 5. What it buys an O&M planner

Planners optimize dispatch, not leaderboard rank. Week 6 economics ( **€5k** per inspection, **four-turbine** fleet ): LSTM at **31** false alarms per turbine-year implies **€155k** annual inspection-like spend if every alarm is acted on; hybrid at **8.5** implies **€43k**. One **€100k+** gearbox avoidance pays for ~**20** false inspections; **65** days median warning turns crane calls into planned stops.

Week 11 adds **probabilistic RUL** — LightGBM **P10/P50/P90** with **MAPIE CQR**, leave-one-failure-out on **four** gearbox events across **three** turbines. LOFO metrics and **P10 first below 14 d** come from [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) on a synthetic fleet export; current P10 lead cells read **`none`** — pessimistic P10 never crossed 14 days in held-out windows. Serving is real: FastAPI **/twin/score** and **/twin/rul**, Streamlit scrubber.

![Week 11 Streamlit dashboard scrub through the T06 failure window.](docs/figures/twin_dashboard.png)

*Residual view, thresholding, and RUL band on the same failure window an operator opens after an alarm.*

---

## 6. Honest limits

**Four** gearbox log lines on **three** turbines — demonstration-grade RUL, not fleet SLAs. **Marginal CQR** with tiny calibration tails. **No vibration/CMS** on EDP; envelope energy often leads 10‑minute thermal SCADA. One thermal score does **not** separate pump, bearing, and noise-led failures. **Open-loop ODE**, 10‑minute averaging, **synthetic EDP** figures unless you re-run on downloaded data. Vibration and more events would layer on load-conditioned residuals, not replace them.

---

## 7. Why field-shaped questions are the moat

Employers can buy libraries. They cannot buy the habit of reading an alarm as power, speed, ambient, then **what is left over**. Pure ML asks whether the SCADA row is rare. EE-shaped work asks whether the machine is **appropriately hot for the work it is doing**. Transparent thermal backbone, learning on the **residual**, benchmark trade-offs toward **enough lead time with a false-alarm rate someone will staff** — that is physics-informed anomaly detection with field intuition. The repos are there to be rerun; the question survives the turbine floor.

---

## Source repos

- [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) — Week 6 benchmarks, trade-off figure, T06 case study.
- [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) — hybrid twin, RMSE table, timeline, dashboard, limitations.
- [grid-gnn-faults](https://github.com/mehmetertac/grid-gnn-faults) — conformal RUL, LOFO evaluation export.
