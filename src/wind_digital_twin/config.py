"""Configuration for wind turbine digital twin (gearbox thermal path)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "edp"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_COLUMNS: list[str] = [
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
    "Grd_Prod_Pwr_Avg",
    "Rtr_RPM_Avg",
    "Amb_WindSpeed_Avg",
    "Amb_WindDir_Relative_Avg",
    "Nac_Temp_Avg",
]

POWER_COLUMN = "Grd_Prod_Pwr_Avg"
MIN_POWER_KW = 0.0

THERMAL_TARGET_COLUMNS: list[str] = [
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
]
THERMAL_DRIVER_COLUMNS: list[str] = [
    POWER_COLUMN,
    "Rtr_RPM_Avg",
    "Nac_Temp_Avg",
]
THERMAL_MIN_POWER_KW = 50.0
THERMAL_TRAIN_FRACTION = 0.8
THERMAL_GBM_RMSE_IMPROVEMENT = 0.10
THERMAL_SEASONAL_TERMS = False
THERMAL_RESULTS_DIR = RESULTS_DIR / "physics_thermal"

RESIDUAL_WINDOW_SIZE = 6
RESIDUAL_EWMA_SPAN = 36
RESIDUAL_ROLLING_WINDOWS: list[int] = [6, 36, 144]

SIGNAL_FILE_ALIASES: dict[str, list[str]] = {
    "signals_2016": [
        "wind-farm-1-signals-2016.csv",
        "Wind-Turbine-SCADA-signals-2016.csv",
    ],
    "signals_2017": [
        "wind-farm-1-signals-2017.csv",
        "Wind-Turbine-SCADA-signals-2017_0.csv",
        "Wind-Turbine-SCADA-signals-2017.csv",
    ],
}

FAILURE_FILE_ALIASES: dict[str, list[str]] = {
    "failures_2016": [
        "htw-failures-2016.csv",
        "Historical-Failure-Logbook-2016.csv",
    ],
    "failures_2017": [
        "htw-failures-2017.csv",
        "opendata-wind-failures-2017.csv",
    ],
}

MENDELEY_XLSX_MAP: dict[str, str] = {
    "Wind-Turbine-SCADA-signals-2016.xlsx": "wind-farm-1-signals-2016.csv",
    "Wind-Turbine-SCADA-signals-2017.xlsx": "wind-farm-1-signals-2017.csv",
    "Historical-Failure-Logbook-2016.xlsx": "htw-failures-2016.csv",
    "opendata-wind-failures-2017.xlsx": "htw-failures-2017.csv",
}

GEARBOX_FAILURES: list[dict[str, str]] = [
    {
        "turbine_id": "T01",
        "timestamp": "2016-07-18T02:10:00+00:00",
        "remarks": "Gearbox pump damaged",
    },
    {
        "turbine_id": "T06",
        "timestamp": "2017-10-17T08:38:00+00:00",
        "remarks": "Gearbox bearings damaged",
    },
    {
        "turbine_id": "T09",
        "timestamp": "2016-10-11T08:06:00+00:00",
        "remarks": "Gearbox repaired",
    },
    {
        "turbine_id": "T09",
        "timestamp": "2017-10-18T08:32:00+00:00",
        "remarks": "Gearbox noise",
    },
]

DEFAULT_BUFFER_DAYS = 90
DEFAULT_HORIZON_DAYS = 30
DEFAULT_CONTAMINATION = 0.01
DEFAULT_PERSISTENCE_SAMPLES = 6
DEFAULT_COOLDOWN_HOURS = 24
DEFAULT_THRESHOLD_PERCENTILE = 99.0
FAILURE_TURBINES: tuple[str, ...] = ("T01", "T06")
PHYSICS_HYBRID_DETECTOR = "physics_hybrid"


@dataclass(frozen=True)
class GearboxFailure:
    """Logged gearbox failure event."""

    turbine_id: str
    timestamp: datetime
    remarks: str
