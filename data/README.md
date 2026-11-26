# Data

## EDP Wind Farm 1 (primary)

Open SCADA and failure logs from EDP / Hack the Wind (10-minute averages).

**Layout:** place CSVs under `data/raw/edp/`.

Accepted filenames (aliases supported — see `wind_digital_twin.config`):

- `wind-farm-1-signals-2016.csv`, `wind-farm-1-signals-2017.csv`
- `htw-failures-2016.csv`, `htw-failures-2017.csv`

```powershell
python scripts/download_edp.py --instructions
python scripts/download_edp.py --check
```

Mendeley XLSX fallback (v1 portal layout only): `python scripts/download_edp.py --from-mendeley`

### Mendeley filtered EDP (second real-data route)

[Kijanowski et al. v2](https://data.mendeley.com/datasets/zjxjnjp3xs/2) (Aug 2026): filtered SCADA for **T01, T06, T07, T11** (83 columns each; 95% and 99.9% confidence variants), met-mast (41 columns), event log (~256k rows), failure log (28 rows). Filtering method: [doi:10.3390/en18225954](https://doi.org/10.3390/en18225954). The v2 bundle uses per-turbine files, not the v1 XLSX names in `MENDELEY_XLSX_MAP` — a dedicated v2 converter is still pending.

## CARE-to-Compare (Wind Farm A = EDP Portugal)

Benchmark data from [Zenodo v6](https://zenodo.org/records/15846963) (paper [arXiv:2404.10320](https://arxiv.org/abs/2404.10320)): 22 sub-datasets, 86 features, real EDP onshore farm.

```powershell
python scripts/download_care.py --instructions
python scripts/download_care.py --wind-farm A
python scripts/download_care.py --check
```

Extracted layout: `data/raw/care/wind_farm_a/` (`datasets/*.csv`, `event_info.csv`, `feature_description.csv`). Channel map for the twin: `src/wind_digital_twin/data/care_wfa_channel_map.csv`.

Sanity baselines (random ≈ 0.5 CARE on full 95-event benchmark; WF-A only):

```powershell
python scripts/run_care_baselines.py
python scripts/run_care_hybrid.py
python scripts/run_care_ablations.py
```

Ablation scores (hybrid P95/AE/persistence): `results/care/wfa_ablations.json`.

## Synthetic fallback (CI / local)

When the portal is unavailable:

```powershell
python scripts/generate_synthetic_edp.py --force
```

Marker file: `data/raw/edp/.synthetic_edp`. Synthetic metrics are for **development only**, not publication.

## Gearbox scope columns

Used by the thermal path: `Gear_Oil_Temp_Avg`, `Gear_Bear_Temp_Avg`, `Grd_Prod_Pwr_Avg`, `Rtr_RPM_Avg`, `Nac_Temp_Avg` (plus wind columns in the panel).

See also [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) and the source anomaly project [PHYSICS_THERMAL](https://github.com/mehmetertac/wind-turbine-anomaly/blob/main/docs/PHYSICS_THERMAL.md).
