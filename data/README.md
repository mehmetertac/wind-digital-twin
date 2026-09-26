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

Mendeley XLSX fallback: `python scripts/download_edp.py --from-mendeley`

## Synthetic fallback (CI / local)

When the portal is unavailable:

```powershell
python scripts/generate_synthetic_edp.py --force
```

Marker file: `data/raw/edp/.synthetic_edp`. Synthetic metrics are for **development only**, not publication.

## Gearbox scope columns

Used by the thermal path: `Gear_Oil_Temp_Avg`, `Gear_Bear_Temp_Avg`, `Grd_Prod_Pwr_Avg`, `Rtr_RPM_Avg`, `Nac_Temp_Avg` (plus wind columns in the panel).

See also [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) and the source anomaly project [PHYSICS_THERMAL](https://github.com/mehmetertac/wind-turbine-anomaly/blob/main/docs/PHYSICS_THERMAL.md).
