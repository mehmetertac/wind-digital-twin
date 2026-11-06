# From the turbine floor to the model: physics-informed anomaly detection with field intuition

**Draft — not yet edited for Saturday publish.** Figures and numbers below come only from the [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) and [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) READMEs.

---

## 1. What the alarm list does not tell you

Picture the alarm banner on a wind farm’s SCADA: “bearing temperature high.” That single line sits at the top of a list that scrolls with yaw errors, converter messages, and communications. For turbine **T06** on the EDP open-data wind farm, the maintenance log eventually records **gearbox bearing damage** on **17 October 2017**. Between routine checks and that log line, the alarm screen never answers the question an electrical engineer asks first: is this bearing **as hot as it ought to be** for the power and speed it is carrying and for the temperature of the nacelle around it?

Gearbox temperature is not like a fuse rating stamped on a nameplate. It rises when the machine works hard and when the enclosure is warm. A hot bearing trace on its own does not separate “we are in a windy afternoon in summer” from “something in the lubrication path or the HSS bearing is generating heat we did not budget.” The alarm list flags the first; it does not encode the second. What it lacks is **load-conditioned expectation** — the thermodynamic map from operating point to steady thermal state. Without that map, every hot spell looks like a candidate emergency, and every quiet shift after a noisy week looks like recovery. Field intuition is knowing which fires are weather and work, and which ones are friction and wear. The model story below is an attempt to put that question — *should it be this hot right now?* — into software that an O&M planner can use alongside the same 10‑minute SCADA the alarm server already sees.

---

## 2. Why pure multivariate ML on SCADA misfires

A natural data-science baseline is to treat SCADA as a fat table of correlated sensors and ask whether the current row is “unusual.” The Week 6 benchmark does exactly that: Isolation Forest on raw features, a dense autoencoder, and an LSTM autoencoder are compared against a **physics-residual hybrid** on EDP-shaped data, at a **99th-percentile** threshold, with lead time measured in days to logged gearbox failure and false alarms counted per turbine-year.

The published table is unambiguous. **Isolation Forest** reaches **18** days lead on T01’s pump-damage case and **48** days on T06’s bearing case, but it racks up **106** false alarms per turbine-year. A **dense autoencoder** pushes median lead time to **73** days at **90** false alarms per turbine-year. The strongest pure-ML competitor in the table, the **LSTM autoencoder**, delivers **80** days median lead time at **31** false alarms per turbine-year. The **physics-residual hybrid** lands at **65** days median lead time at **8.5** false alarms per turbine-year. Per-event, the hybrid is not uniformly ahead on early warning: on T01 it reports **53** days to the pump-damage log line versus **88** for the LSTM-AE; on T06 it reports **77** days versus **72** for the LSTM-AE. The README’s own headline is that the hybrid **trades some lead time, especially on T1, for a much lower false-alarm rate** compared with the best pure-ML row.

That trade-off is the point. Multivariate detectors learn whatever joint pattern held in training. When power, RPM, and nacelle temperature jointly imply a “high” bearing temperature, a model that does not subtract that expectation first will sometimes score **healthy peak production** as anomalous. The benchmark is run on **synthetic EDP** for pipeline development; the README is explicit that you should obtain real EDP CSVs and re-run before comparing to external literature. Even so, the relative lesson is structural: **raw-SCADA novelty is a poor proxy for mechanical fault** when the dominant variance is operating point.

![Week 6 trade-off: lead time vs false-alarm rate as threshold varies; hybrid sits in the lower-left operable region.](https://github.com/mehmetertac/wind-turbine-anomaly/raw/main/docs/assets/threshold_tradeoff.png)

*At this sweep, the physics-residual hybrid occupies the lower-left “operable” region — fewer false alarms at comparable sensitivity — with the default 99th-percentile point marked on the curve.*

---

## 3. The physics-residual idea in plain language

The hybrid path does not ask whether the current SCADA vector is rare. It asks whether **measured temperature** matches **expected temperature** given what the turbine is doing. The Week 6 pipeline fits a normal-behavior thermal model so that expected oil and bearing temperatures are functions of drivers such as power, speed, and nacelle temperature. The **residual** is **predicted minus actual**. Where physics expects a cooler bearing than measured, the residual goes negative over time if extra heat accumulates; the README’s annotated T06 case cites roughly **−4 °C** of residual drift over the **90** days before failure as the visible degradation signal compared with raw temperature alone.

Detection then moves to the residual stream — with the same broad alarm mechanics as the pure-ML baselines in the benchmark — so that a high-load, hot-nacelle day does not automatically look like a fault simply because absolute temperature is high. In one sentence: **subtract the weather and the work first; whatever systematic gap remains is closer to “something wrong with the machine.”** That is the entire physics-informed trick. It is not a claim that the thermal fit is a full finite-element gearbox model; it is a claim that **the residual is the right coordinate system** for SCADA-only thermal surveillance when alarms confuse load with loss.

---

## 4. From Week 6 charts to the Week 10–11 twin

The Week 6 repo ships a head-to-head story: thresholds, lead times, and a four-panel case study for T6 bearing wear-out — raw traces, predicted versus actual bearing temperature, residual drift, anomaly score with first alarm, and timeline from first alarm to logged failure with **77** days of lead time.

![T06 annotated case study from Week 6: power and temperatures, prediction, residual drift, score, and timeline to failure.](https://github.com/mehmetertac/wind-turbine-anomaly/raw/main/docs/assets/case_study_T06.png)

*The case study ties residual drift and hybrid anomaly score to first sustained alarm and **77** days of warning to the logged bearing event.*

The digital-twin capstone carries the same backbone forward: a **lumped thermal ODE** for expected oil and bearing temperature, a **GBM residual corrector** fit on **healthy** rows only from exogenous regime features (not gear temperatures), and an **Isolation Forest** on hybrid residual windows plus bearing–oil context. On **healthy** holdout data, the README reports **synthetic EDP** RMSE in °C. For **T06**, oil temperature RMSE falls from **2.26** (physics-only) to **0.40** (hybrid), an **82** percent relative improvement; bearing RMSE falls from **2.37** to **0.44**, an **81** percent improvement. **T01**, **T07**, and **T11** show the same pattern — large reductions from physics-only to hybrid on the healthy split. The README cautions explicitly that **pre-failure degradation** shows up in the **timeline** residuals and anomaly score, **not** in that healthy RMSE table, which is a goodness-of-fit check on normal operation rather than a failure detector headline.

The Week 10 figure below is the four-aligned-panel **twin timeline** around T06’s logged gearbox-bearing failure: SCADA temperatures, physics versus hybrid **expected** bearing temperature on the **same** ODE backbone, physics versus hybrid **residuals**, and hybrid-residual anomaly score with threshold and alarm spans, including the gray band for the **90**‑day training holdout before failure.

![End-to-end twin timeline on synthetic EDP around T06 failure: expected temps, residuals, anomaly score, and alarms.](docs/figures/twin_timeline_T06.png)

*The timeline connects hybrid expectations, residual separation from physics-only, and sustained anomaly scoring through the failure window on **synthetic EDP**.*

The healthy-validation RMSE comparison figure drives home where the GBM layer earns its keep on **normal** segments.

![Oil and bearing RMSE: physics-only ODE versus hybrid on healthy validation rows.](docs/figures/hybrid_vs_physics_rmse.png)

*On healthy data, hybrid RMSE is far below physics-only across turbines — systematic ODE error learned out of band without using gear temps as inputs.*

---

## 5. What it buys an O&M planner

Maintenance organizations do not optimize Kaggle leaderboards. They optimize **when** to send a technician, when to order parts, and how often they can tolerate a false dispatch before the site manager stops answering the phone. The Week 6 README restates economics with round numbers: assume about **€5k** per inspection on a **four-turbine** fleet. At **31** false alarms per turbine-year, the LSTM autoencoder implies about **€155k** per year in inspection-like spend if every alarm is acted on. At **8.5** false alarms per turbine-year, the physics-residual hybrid implies about **€43k** per year under the same assumption. The same section notes that one avoided catastrophic gearbox replacement on the order of **€100k+** pays for roughly **20** false-alarm inspections, and that **65** days of warning is the median headline horizon at the 99th-percentile point — enough to turn an unplanned crane mobilization into a **planned** intervention.

The Week 11 layer in the twin README adds **probabilistic RUL** — LightGBM **P10 / P50 / P90** with **MAPIE CQR** widening, leave-one-failure-out framing on **four** catalogued gearbox events across **three** turbines, and an explicit table for operational metrics including **P10 first below 14 d (days before failure)**. In the committed README, the numeric cells for that table are **em dashes** until `python scripts/run_twin.py` in grid-gnn-faults writes a fleet **scored CSV** from the twin without refitting the ODE. **This draft does not invent “X days” for those cells.** The evaluation contract is written; the fleet export is the missing step. The **serving** story is still real: FastAPI **POST /twin/score** and **POST /twin/rul**, Streamlit dashboard, precomputed failure-window series for a scrubber — the interface an O&M planner would actually open after the SCADA alarm fires.

![Week 11 Streamlit dashboard scrub through the T06 failure window.](docs/figures/twin_dashboard.png)

*The dashboard binds residual view, anomaly thresholding, and RUL band display to the same failure window an operator would interrogate after an alarm.*

---

## 6. Honest limits

The twin README lists the constraints that belong in every interview answer. **Failure labels** are thin: **four** gearbox log lines on **three** turbines, so RUL and conformal coverage are **demonstration-grade**, not fleet SLAs. **CQR coverage** is **marginal** under the plan’s assumptions; with tiny calibration tails and non-exchangeable run-to-failure windows, gaps versus nominal 80% / 90% are expected rather than something to “tune away.” **EDP has no vibration or CMS**; on real gearboxes, accelerometer envelope and band energy often lead **10**‑minute thermal SCADA by days to weeks. A **single thermal score** on oil and HSS bearing temperature does **not** separate pump damage from bearing wear from noise-led events — T01, T06, and T09 are different root causes in the log. **10**‑minute SCADA averages sub-hour transients. The **open-loop ODE** resets state across long gaps by design so sustained hot spells stay visible in residuals, but it is not a closed-loop OEM twin. **Figures** in the README use **synthetic EDP** unless you re-run the pass scripts on a downloaded real dataset.

More failure exemplars and vibration would change the story: they would tighten class separation, shorten the path from anomaly to mode hypothesis, and give independent confirmation before anyone books a borescope. None of that negates the SCADA lesson — it **layers** on top of load-conditioned residuals.

---

## 7. Why field-shaped questions are the moat

Algorithms come and go; employers can buy libraries. What they cannot buy from a generic resume is the habit of translating a flashing alarm into a **governing relationship** — power, speed, ambient envelope, then **what is left over**. Pure ML on raw SCADA asks whether the current point is rare in the training distribution. An EE-shaped workflow asks whether the machine is **appropriately hot for the work it is doing**. Encode that second question with a transparent thermal backbone, put learning on the **residual** where degradation actually lives, and the benchmark table shifts from “more lead time at any cost” to **enough lead time with a false-alarm rate someone will staff**. That shift — from novelty in sensor space to physical consistency in operating space — is what “physics-informed anomaly detection with field intuition” was meant to name. The repos are there to be rerun, critiqued, and improved; the question they standardize is the part that survives contact with the turbine floor.

---

## Source repos

- [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) — Week 6 benchmarks, trade-off figure, T06 case study.
- [wind-digital-twin](https://github.com/mehmetertac/wind-digital-twin) — hybrid twin, RMSE table, timeline, dashboard, limitations.
