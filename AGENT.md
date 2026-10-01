# AGENT.md — rules for AI agents

Guidance for agents working in **wind-digital-twin**. Read this file first, then follow the linked docs.

---

## Documentation map

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | Entry point: setup, architecture link, maintenance interpretation |
| [handover.md](handover.md) | Current status, repo layout, module API, week roadmap |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Digital twin taxonomy, fidelity levels, capstone pipeline diagram |
| [docs/reading_notes.md](docs/reading_notes.md) | Deep reads (Raissi 2019, Pujana 2023), abstract skims, PINN toy spec |
| [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md) | EDP SCADA signature inventory + fused residual-model feature spec |
| [src/wind_digital_twin/pinn/](src/wind_digital_twin/pinn/) | 1-D heat PINN + ablation plots |
| [scripts/run_pinn_heat.py](scripts/run_pinn_heat.py) | Train PINN vs data-only; writes `docs/figures/pinn_ablation.png` |
| [data/README.md](data/README.md) | EDP provenance, synthetic fallback |
| [notebooks/README.md](notebooks/README.md) | Notebooks are thin drivers only (PINN toy, EDA) |
| [src/wind_digital_twin/data/](src/wind_digital_twin/data/) | Load, clean, synthetic EDP |
| [src/wind_digital_twin/physics/](src/wind_digital_twin/physics/) | Lumped ODE gearbox thermal model (`lumped_ode.py`, `gearbox_thermal.py`) |
| [src/wind_digital_twin/residual/](src/wind_digital_twin/residual/) | Residual feature engineering |
| [src/wind_digital_twin/anomaly/](src/wind_digital_twin/anomaly/) | Physics-hybrid anomaly detector |
| [src/wind_digital_twin/eval/](src/wind_digital_twin/eval/) | Lead-time protocol, plots |
| [scripts/run_gearbox_thermal.py](scripts/run_gearbox_thermal.py) | Fit physics layer CLI |
| [tests/](tests/) | Unit + integration tests (fixtures + synthetic; no EDP download in CI) |
| [wind-turbine-anomaly](https://github.com/mehmetertac/wind-turbine-anomaly) | Source repo for ported physics-residual baseline |

Headline contract: **honest uncertainty on every model output** (residual spread now; conformal in Week 11), **time-ordered leakage-free evaluation**, and a **maintenance-meaning** section in user-facing docs.

---

## Rules

### 1. File size limit

- **No file should exceed 1,000 lines.**
- If a file approaches or exceeds that limit, **stop and suggest a refactor** before adding more code.
- Pre-commit runs [`scripts/check_file_size.py`](scripts/check_file_size.py).

### 2. Documentation before every push

- **Update documentation before every push** to the repository.
- At minimum, check [README.md](README.md) and [handover.md](handover.md).

### 2a. Update handover.md on every push (required)

- Refresh **Last updated**, **What is done**, **Repo layout**, **Core module API**, **Suggested next step**, and **Key commit** on each push.
- If nothing functional changed, still bump **Last updated** and note “no functional change.”

### 3. Tests — always, at least minimal

- **Always create at least minimal unit tests**, even for small changes.
- Add **integration** tests when wiring modules (data → physics → residual → anomaly).
- Add **functional** tests when the project supports them (CLI smoke).
- Pattern: [tests/](tests/) uses fixtures and synthetic data — CI must not depend on EDP portal downloads.

### 4. Run tests before commit or push

```powershell
pytest tests/ -q
python scripts/check_file_size.py
```

**Git hooks:** After creating the venv:

```powershell
pip install -e ".[dev]"
pre-commit install
```

Hooks run file-size check + pytest (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)). If hooks do not exist yet, create them.

### 5. Keep reading in-repo docs

- Do not guess API or roadmap from memory — use [handover.md](handover.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick checklist (before push)

- [ ] No file > 1,000 lines (or refactor proposed)
- [ ] [README.md](README.md) updated if behavior or layout changed
- [ ] [handover.md](handover.md) updated (required on every push)
- [ ] New/changed logic has tests in [tests/](tests/)
- [ ] `pytest tests/ -q` passes
- [ ] `python scripts/check_file_size.py` passes
