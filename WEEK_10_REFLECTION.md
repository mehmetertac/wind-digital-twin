# Week 10 reflection

**Date:** 2026-10-01

## What I built

This week the capstone **physics backbone** moved from the Week 6 steady-state Ridge/GBM map in [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) to a **first-order lumped thermal ODE** in [`src/wind_digital_twin/physics/lumped_ode.py`](src/wind_digital_twin/physics/lumped_ode.py), wired through [`gearbox_thermal.py`](src/wind_digital_twin/physics/gearbox_thermal.py). The identifiable form is:

\[
\tau \frac{dT}{dt} = T_{\mathrm{eq}} - T, \qquad
T_{\mathrm{eq}} = T_{\mathrm{nac}} + \beta_0 + \beta_P P + \beta_\omega\,\mathrm{RPM}.
\]

Parameters are fit with `scipy.optimize.least_squares` on **healthy** EDP rows only; held-out validation still checks residuals (small, unbiased, structureless) plus **τ in minutes** against oil/bearing plausibility bands in [`config.py`](src/wind_digital_twin/config.py). The old linear model remains as a **RMSE baseline** in `fit_gearbox_thermal_with_selection` (`linear_val_rmse` vs `ode_val_rmse`).

The **1-D heat PINN toy** ([`pinn/heat1d.py`](src/wind_digital_twin/pinn/heat1d.py)) stays the guardrail for “physics loss matters when labels are sparse”—it is **not** the SCADA backbone. Optional `--nn-correction` adds a **frozen-physics**, capped (\(\pm 3\,^\circ\mathrm{C}\)) MLP tweak to \(T_{\mathrm{eq}}\) only ([`lumped_ode_correction.py`](src/wind_digital_twin/physics/lumped_ode_correction.py)); joint PINN-style training of \(\tau\) and the network was deliberately **not** shipped.

Downstream contract unchanged: `GearboxThermalModel.predict` / `residual` → [`residual_features`](src/wind_digital_twin/residual/residual_features.py) → [`physics_hybrid`](src/wind_digital_twin/anomaly/physics_hybrid.py).

## Validation snapshot (synthetic EDP)

On locally generated quasi-steady SCADA (`python scripts/generate_synthetic_edp.py --force`), oil τ values landed in a **plausible** range (roughly 30–50 min on T01/T06) but **validation “passed” flags often failed** because ODE held-out RMSE was higher than the Ridge baseline (~0.25 °C vs ~1.8–2.8 °C). That is expected here: the generator is **instantaneous** in load and temperature (no true lag), while the linear baseline still uses **\(P^2\)** features the ODE does not. On real 10-min EDP I expect τ to pile against the **1 min floor** when the series is quasi-steady; that is not a physics failure—it means “use the equilibrium gains; lag is unresolved at this sampling.”

## What is still fuzzy

**PINN / NN training instability.** I did not run a joint data + ODE-residual loss on SCADA. The optional NN only fits after \(\{\tau, \beta\}\) are fixed; a single end-to-end PINN would likely trade identifiability for fit and need careful loss weighting (same pain as the spatial PINN toy when `w_f` is wrong).

**ODE parameter identifiability.** \(\tau\) and the equilibrium gains are separable when drivers move; **\(C\)** and **\(hA\)** are not identifiable from temperature alone. **\(\beta_P\)** and **\(\beta_\omega\)** are partially collinear because RPM tracks power on this farm. **\(T_{\mathrm{nac}}\)** is a proxy for ambient/cooling air, not a first-principles boundary condition.

**10-min SCADA limits.** Averaging hides sub-hour bearing transients; open-loop prediction (no feedback from measured gear temp except gap resets) is intentional so a sustained hot spell stays visible in residuals— but it is harder to fit than one-step “innovation” models.

**What vibration would unlock.** EDP has **no CMS**. Gearbox **noise** (T09) and **bearing damage** (T06) are exactly the cases where accelerometer envelope / band energy leads **days to weeks** before 10-min oil/bearing temperature drifts. Oil **pressure/flow** (T01 pump) would front-run sump temperature. See [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md).

## How this twin compares to CMS I have seen on turbines

| Typical field CMS / OEM monitoring | This capstone twin |
|-----------------------------------|-------------------|
| Fixed thresholds on raw oil/bearing temp | **Load-conditioned** expectation via \(T_{\mathrm{eq}}(P,\mathrm{RPM},T_{\mathrm{nac}})\) |
| Vibration KPIs (ISO bands, demod, etc.) | **Not in repo** — thermal SCADA path only |
| Oil debris / particle counters | Not in EDP |
| Fleet dashboards (shadow/descriptive) | **Diagnostic/predictive** hybrid: physics residual stream + IF + lead-time eval |
| Black-box ML on all SCADA | **Hybrid**: interpretable ODE core + (planned) learned residual on structured errors |

This is closer to a **component-level predictive advisory twin** (open-loop alarms for maintenance planning) than to a full OEM aero-elastic or converter twin (cf. Pujana et al. notes in [docs/reading_notes.md](docs/reading_notes.md)). It is **more explicit about physics** than many “AI CMS” demos, but **less sensitive early** than a proper vibration CMS on the same gearbox.

## Leakage and uncertainty (unchanged commitments)

- Train on **healthy** rows only; **90-day** buffer before logged failure.
- **Time-ordered** train/validation split for thermal fit.
- Week 11: **conformal** intervals on scores/RUL—not done yet; this week’s honest uncertainty is validation RMSE, residual spread, and τ plausibility flags.

## Next

1. Fusion columns + `build_fusion_feature_frame()` per [SENSOR_FUSION.md](docs/SENSOR_FUSION.md).
2. Learned residual model on top of ODE (and linear baseline comparison).
3. Re-run `python scripts/run_gearbox_thermal.py` on **real** EDP download when available; record τ and whether ODE beats linear on held-out healthy data.
