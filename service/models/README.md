# Serving model artifacts

These files are **synthetic smoke models** fit on the repo’s generated EDP CSVs. They exist so CI, Docker, and the Streamlit dashboard start without downloading real SCADA or refitting the thermal ODE on every push.

Regenerate:

```powershell
pip install -e ".[export]"
python service/scripts/export_models.py
```

| File | Role |
| --- | --- |
| `hybrid_pipeline.joblib` | Week 10 hybrid-residual detector (ODE + GBM + IF) |
| `anomaly_threshold.json` | Healthy holdout alarm threshold |
| `rul_bundle.joblib` | LightGBM P10/P50/P90 on daily twin features |
| `conformal_offsets.json` | Split-CQR widening constants for 80%/90% bands |
| `shap_sensor_ranking.csv` | Global TreeSHAP sensor ranks (p10/p50) |
| `timeline_demo.parquet` | Precomputed T01/T06 failure-window daily series |

Do not treat these as fleet-calibrated production weights.
