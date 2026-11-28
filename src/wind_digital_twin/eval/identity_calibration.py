"""Identity-disjoint alert thresholds for CARE-style evaluation.

The 2026 Wind Energy Science CARE v6 benchmark (wes-2026-145) fits a
normal-behaviour model on an earlier status-normal block, then sets the
alert rule without using the turbine that will be scored. This module
implements that split: location and scale come from the fit block, and the
finite-sample rank threshold comes from calibration scores of other assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Published Farm A holdout in wes-2026-145 (identity-only SHA-256 split).
FARM_A_LOCKED_TEST_ASSETS = frozenset({"13", "21"})

# Repo default P99, frozen before looking at Wind Farm A scores. Not tuned here.
DEFAULT_ALERT_ALPHA = 0.01

MAX_FIT_ROWS = 12_000
MAX_CAL_ROWS = 6_000
FIT_FRACTION = 0.8


def chronological_fit_cal_split(
    frame: pd.DataFrame,
    *,
    fit_fraction: float = FIT_FRACTION,
    max_fit_rows: int = MAX_FIT_ROWS,
    max_cal_rows: int = MAX_CAL_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier normal block for fitting, later disjoint block for calibration.

    ``frame`` must already be time-sorted. The boundary is the first
    ``fit_fraction`` of rows. Fitting uses the most recent ``max_fit_rows``
    before that boundary; calibration uses the following ``max_cal_rows``.
    """
    if frame.empty:
        return frame, frame.iloc[0:0]
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in [0.5, 1)")
    n = len(frame)
    if n < 2:
        return frame.iloc[0:0], frame.iloc[0:0]
    boundary = int(n * fit_fraction)
    boundary = min(max(boundary, 1), n - 1)
    fit_start = max(0, boundary - max_fit_rows)
    cal_end = min(n, boundary + max_cal_rows)
    fit = frame.iloc[fit_start:boundary]
    cal = frame.iloc[boundary:cal_end]
    return fit, cal


def chronological_fit_cal_split_uncapped(
    frame: pd.DataFrame,
    *,
    fit_fraction: float = FIT_FRACTION,
    max_cal_rows: int = MAX_CAL_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same chronological holdout as ``chronological_fit_cal_split``, without the fit cap.

    The 12,000-row cap keeps only the latest part of the fit block, which on
    Wind Farm A is often a single season. Rise models need the whole earlier
    block. Calibration stays the following disjoint rows.
    """
    if frame.empty:
        return frame, frame.iloc[0:0]
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in [0.5, 1)")
    n = len(frame)
    if n < 2:
        return frame.iloc[0:0], frame.iloc[0:0]
    boundary = int(n * fit_fraction)
    boundary = min(max(boundary, 1), n - 1)
    cal_end = min(n, boundary + max_cal_rows)
    return frame.iloc[:boundary], frame.iloc[boundary:cal_end]


def fit_location_scale(scores: np.ndarray) -> tuple[float, float]:
    """Robust center and scale from fit-block scores (model statistics, not a cutoff)."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return center, scale


def to_z(scores: np.ndarray | pd.Series, center: float, scale: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return (values - center) / scale


def conformal_pvalues(cal: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Upper-tail split conformal p-values of ``values`` against ``cal``.

    ``p = (1 + #{calibration >= value}) / (n + 1)``. Non-finite values stay
    NaN. An empty calibration pool returns 1 for every finite value, which
    does not alarm.
    """
    calibration = np.asarray(cal, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    observed = np.asarray(values, dtype=float)
    out = np.full(observed.shape, np.nan, dtype=float)
    finite = np.isfinite(observed)
    if not finite.any():
        return out
    n = int(calibration.size)
    if n == 0:
        out[finite] = 1.0
        return out
    order = np.sort(calibration)
    counts = n - np.searchsorted(order, observed[finite], side="left")
    out[finite] = (1.0 + counts) / (n + 1.0)
    return out


def conformal_rank_threshold(calibration_z: np.ndarray, alpha: float = DEFAULT_ALERT_ALPHA) -> float:
    """Upper conformal rank on sorted calibration scores.

    Rank ``q = min(n, max(1, ceil((n+1)(1-alpha))))`` follows wes-2026-145.
    An empty pool falls back to the standard-normal quantile, which does not
    read the test turbine.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    values = np.sort(np.asarray(calibration_z, dtype=float))
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha))
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(n, max(1, rank))
    return float(values[rank - 1])


def threshold_excluding_assets(
    calibration_by_asset: dict[str, np.ndarray],
    excluded_assets: set[str] | frozenset[str],
    alpha: float = DEFAULT_ALERT_ALPHA,
) -> float:
    """Pool calibration z-scores from assets outside ``excluded_assets``."""
    excluded = {str(asset) for asset in excluded_assets}
    parts = [
        np.asarray(scores, dtype=float)
        for asset, scores in calibration_by_asset.items()
        if str(asset) not in excluded and len(scores)
    ]
    if not parts:
        return conformal_rank_threshold(np.array([]), alpha)
    return conformal_rank_threshold(np.concatenate(parts), alpha)


def format_crossed_lead(days: float | None, event_id: str) -> str:
    """Integer days, or an explicit miss. Never ``None`` and never the word none."""
    if days is None:
        return f"never crossed on event {event_id}"
    value = float(days)
    if not np.isfinite(value):
        return f"never crossed on event {event_id}"
    return str(int(np.rint(value)))
