"""Daily RUL feature contract (matches grid-gnn-faults ``rul.labels``)."""

from __future__ import annotations

RUL_MAX_DAYS = 90
MIN_SCADA_ROWS = 160

DAILY_FEATURE_COLUMNS: tuple[str, ...] = (
    "oil_deg_mean",
    "oil_deg_std",
    "bear_deg_mean",
    "bear_deg_std",
    "anomaly_score_mean",
    "anomaly_score_max",
    "oil_deg_slope_7d",
    "bear_deg_slope_7d",
    "theta_oil_mean",
    "theta_bear_mean",
    "delta_bear_oil_mean",
)

SCADA_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
    "Grd_Prod_Pwr_Avg",
    "Rtr_RPM_Avg",
    "Nac_Temp_Avg",
)

SENSOR_ORDER: tuple[str, ...] = (
    "Gearbox oil temperature",
    "Gearbox bearing temperature",
    "Oil temperature per unit power",
    "Bearing temperature per unit power",
    "Bearing-oil temperature difference",
    "Hybrid anomaly score",
)

FEATURE_TO_SENSOR: dict[str, str] = {
    "oil_deg_mean": "Gearbox oil temperature",
    "oil_deg_std": "Gearbox oil temperature",
    "oil_deg_slope_7d": "Gearbox oil temperature",
    "bear_deg_mean": "Gearbox bearing temperature",
    "bear_deg_std": "Gearbox bearing temperature",
    "bear_deg_slope_7d": "Gearbox bearing temperature",
    "theta_oil_mean": "Oil temperature per unit power",
    "theta_bear_mean": "Bearing temperature per unit power",
    "delta_bear_oil_mean": "Bearing-oil temperature difference",
    "anomaly_score_mean": "Hybrid anomaly score",
    "anomaly_score_max": "Hybrid anomaly score",
}
