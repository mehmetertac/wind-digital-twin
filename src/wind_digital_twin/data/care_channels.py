"""Map CARE Wind Farm A anonymized features to twin thermal/electrical/mechanical groups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wind_digital_twin.config import DATA_RAW_CARE_WFA

FROZEN_CHANNEL_MAP = Path(__file__).with_name("care_wfa_channel_map.csv")

# CARE avg column -> twin EDP canonical (verified against WF-A feature_description.csv)
TWIN_ROLE_BY_FEATURE: dict[str, str] = {
    "sensor_0_avg": "Amb_Temp_Avg",
    "sensor_1_avg": "Amb_WindDir_Relative_Avg",
    "sensor_11_avg": "Gear_Bear_Temp_Avg",
    "sensor_12_avg": "Gear_Oil_Temp_Avg",
    "sensor_13_avg": "Gen_Bear2_Temp_Avg",
    "sensor_14_avg": "Gen_Bear1_Temp_Avg",
    "sensor_15_avg": "Gen_Stator1_Temp_Avg",
    "sensor_16_avg": "Gen_Stator2_Temp_Avg",
    "sensor_17_avg": "Gen_Stator3_Temp_Avg",
    "sensor_21_avg": "IGBT_Grid_Temp_Avg",
    "sensor_35_avg": "IGBT_Rotor1_Temp_Avg",
    "sensor_36_avg": "IGBT_Rotor2_Temp_Avg",
    "sensor_37_avg": "IGBT_Rotor3_Temp_Avg",
    "sensor_38_avg": "HV_Trafo_L1_Temp_Avg",
    "sensor_39_avg": "HV_Trafo_L2_Temp_Avg",
    "sensor_40_avg": "HV_Trafo_L3_Temp_Avg",
    "sensor_41_avg": "Hyd_Oil_Temp_Avg",
    "sensor_43_avg": "Nac_Temp_Avg",
    "power_30_avg": "Grd_Prod_Pwr_Avg",
    "wind_speed_3_avg": "Amb_WindSpeed_Avg",
    "sensor_52_avg": "Rtr_RPM_Avg",
    "sensor_18_avg": "Gen_RPM_Avg",
    "sensor_5_avg": "Blds_PitchAngle_Avg",
    "sensor_23_avg": "Grd_Prod_CurPhse1_Avg",
    "sensor_24_avg": "Grd_Prod_CurPhse2_Avg",
    "sensor_25_avg": "Grd_Prod_CurPhse3_Avg",
    "sensor_32_avg": "Grd_Prod_VoltPhse1_Avg",
    "sensor_33_avg": "Grd_Prod_VoltPhse2_Avg",
    "sensor_34_avg": "Grd_Prod_VoltPhse3_Avg",
}

FeatureGroup = str


def classify_feature(description: str, unit: str) -> FeatureGroup:
    text = f"{description} {unit}".lower()
    if "°c" in unit.lower() or "temperature" in text or "temp" in text:
        return "thermal"
    if any(k in text for k in ("voltage", "current", "power", "frequency", "cos", "reactive")):
        return "electrical"
    if any(k in text for k in ("rpm", "pitch", "torque", "generator rpm", "rotor rpm")):
        if "wind speed" not in text and "windspeed" not in text:
            return "mechanical"
    if any(k in text for k in ("wind", "ambient", "direction", "nacelle direction")):
        return "environmental"
    return "other"


_STAT_MAP = {
    "average": "avg",
    "minimum": "min",
    "maximum": "max",
    "std_dev": "std",
    "standard deviation": "std",
}


def _feature_column_name(sensor_name: str, statistic_type: str) -> str:
    stat = statistic_type.strip().lower()
    suffix = _STAT_MAP.get(stat, stat.replace(" ", "_"))
    if suffix == "avg" and statistic_type.lower() == "average":
        return f"{sensor_name}_avg"
    return f"{sensor_name}_{suffix}"


def load_feature_description(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    stat_col = "statistics_type" if "statistics_type" in df.columns else "statistic_type"
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        sensor = str(row["sensor_name"])
        desc = str(row["description"])
        unit = str(row.get("unit", ""))
        group = classify_feature(desc, unit)
        stats = [s.strip() for s in str(row[stat_col]).split(",")]
        for stat in stats:
            stat_key = stat.lower().replace(" ", "_")
            if stat_key not in _STAT_MAP and stat_key not in ("avg", "min", "max", "std"):
                continue
            feature = _feature_column_name(sensor, stat)
            rows.append(
                {
                    "sensor": sensor,
                    "statistic_type": stat,
                    "feature": feature,
                    "description": desc,
                    "unit": unit,
                    "group": group,
                    "twin_role": TWIN_ROLE_BY_FEATURE.get(feature, ""),
                }
            )
    return pd.DataFrame(rows)


def build_channel_map(feature_description_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if feature_description_df is None:
        path = DATA_RAW_CARE_WFA / "feature_description.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return load_feature_description(path)
    return feature_description_df.copy()


def load_frozen_channel_map() -> pd.DataFrame:
    if FROZEN_CHANNEL_MAP.exists():
        return pd.read_csv(FROZEN_CHANNEL_MAP)
    return build_channel_map()


def role_rename_columns(channel_map: pd.DataFrame) -> dict[str, str]:
    mapped = channel_map[channel_map["twin_role"].astype(str).str.len() > 0]
    return dict(zip(mapped["feature"], mapped["twin_role"], strict=False))


def freeze_channel_map_from_data(out_path: Path | None = None) -> pd.DataFrame:
    """Regenerate committed map from downloaded feature_description.csv."""
    out_path = out_path or FROZEN_CHANNEL_MAP
    frame = build_channel_map()
    frame.to_csv(out_path, index=False)
    return frame