"""Cross-signature fusion features for residual corrector and detector context."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.config import (
    FUSION_MIN_RPM,
    POWER_COLUMN,
    THERMAL_MIN_POWER_KW,
    THERMAL_TARGET_COLUMNS,
)

OIL_COL = THERMAL_TARGET_COLUMNS[0]
BEAR_COL = THERMAL_TARGET_COLUMNS[1]
NAC_COL = "Nac_Temp_Avg"
RPM_COL = "Rtr_RPM_Avg"
WIND_COL = "Amb_WindSpeed_Avg"

# Exogenous inputs for the learned corrector (no gear temps or theta ratios).
CORRECTOR_BASE_COLUMNS: list[str] = [
    POWER_COLUMN,
    RPM_COL,
    NAC_COL,
    WIND_COL,
    "torque_proxy",
    "power_sq",
]

# Fusion context for anomaly detector (not corrector inputs).
DETECTOR_FUSION_CONTEXT_COLUMNS: list[str] = [
    "theta_oil",
    "theta_bear",
    "delta_bear_oil",
]


def _phase_imbalance(df: pd.DataFrame, prefix: str) -> pd.Series | None:
    """std/mean of phase columns when all three exist."""
    cols = [f"Grd_Prod_{prefix}Phse{i}_Avg" for i in (1, 2, 3)]
    if not all(c in df.columns for c in cols):
        return None
    values = df[cols].to_numpy(dtype=float)
    mean_abs = np.mean(np.abs(values), axis=1)
    std = np.std(values, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_abs > 1e-6, std / mean_abs, 0.0)
    return pd.Series(ratio, index=df.index, name=f"{prefix[0].lower()}_imbalance")


def build_fusion_feature_frame(
    df: pd.DataFrame,
    min_power_kw: float = THERMAL_MIN_POWER_KW,
    min_rpm: float = FUSION_MIN_RPM,
) -> pd.DataFrame:
    """
    Point fusion features on operating rows (P > min_power_kw).

    Includes theta ratios, delta_bear_oil, torque_proxy, and optional imbalance /
    pitch / gear_ratio when columns exist.
    """
    power = df[POWER_COLUMN]
    rpm = df[RPM_COL]
    nac = df[NAC_COL]
    mask = (power > min_power_kw) & (rpm > min_rpm)
    sub = df.loc[mask].copy()

    parts: dict[str, pd.Series] = {
        POWER_COLUMN: sub[POWER_COLUMN],
        RPM_COL: sub[RPM_COL],
        NAC_COL: sub[NAC_COL],
        "theta_oil": (sub[OIL_COL] - sub[NAC_COL]) / sub[POWER_COLUMN],
        "theta_bear": (sub[BEAR_COL] - sub[NAC_COL]) / sub[POWER_COLUMN],
        "delta_bear_oil": sub[BEAR_COL] - sub[OIL_COL],
        "torque_proxy": sub[POWER_COLUMN] / sub[RPM_COL],
        "power_sq": sub[POWER_COLUMN] ** 2,
    }
    if WIND_COL in sub.columns:
        parts[WIND_COL] = sub[WIND_COL]

    i_imb = _phase_imbalance(sub, "Cur")
    if i_imb is not None:
        parts["i_imbalance"] = i_imb
    v_imb = _phase_imbalance(sub, "Volt")
    if v_imb is not None:
        parts["v_imbalance"] = v_imb

    if "Blds_PitchAngle_Avg" in sub.columns:
        parts["pitch"] = sub["Blds_PitchAngle_Avg"]
    if "Gen_RPM_Avg" in sub.columns:
        parts["gear_ratio"] = sub["Gen_RPM_Avg"] / sub[RPM_COL]

    return pd.DataFrame(parts, index=sub.index)


def corrector_feature_columns(fusion: pd.DataFrame) -> list[str]:
    """Columns available for LightGBM residual corrector (leakage-safe)."""
    return [c for c in CORRECTOR_BASE_COLUMNS if c in fusion.columns]


def build_corrector_matrix(
    df: pd.DataFrame,
    min_power_kw: float = THERMAL_MIN_POWER_KW,
) -> pd.DataFrame:
    """Exogenous feature matrix aligned to operating rows for corrector training."""
    fusion = build_fusion_feature_frame(df, min_power_kw=min_power_kw)
    cols = corrector_feature_columns(fusion)
    return fusion[cols]


__all__ = [
    "CORRECTOR_BASE_COLUMNS",
    "DETECTOR_FUSION_CONTEXT_COLUMNS",
    "build_corrector_matrix",
    "build_fusion_feature_frame",
    "corrector_feature_columns",
]
