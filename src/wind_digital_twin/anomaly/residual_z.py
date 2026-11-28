"""One-sided robust-z score on thermal rise residuals.

The Isolation Forest head treats a cold residual the same as a hot one and
caps the score. This score keeps only the hot side, after an EWMA, and takes
the strongest channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.config import RESIDUAL_EWMA_SPAN
from wind_digital_twin.eval.identity_calibration import fit_location_scale, to_z


def one_sided_channel_scores(
    fit_channels: dict[str, pd.Series],
    score_channels: dict[str, pd.Series],
    *,
    span: int = RESIDUAL_EWMA_SPAN,
    min_fit_rows: int = 30,
) -> dict[str, pd.Series]:
    """Positive EWMA robust-z for each channel.

    Location and scale come from ``fit_channels`` only. A cold residual is
    clipped to zero so it cannot raise the alarm.
    """
    parts: dict[str, pd.Series] = {}
    for name, fit_series in fit_channels.items():
        scored = score_channels.get(name)
        if scored is None or scored.empty:
            continue
        fit_values = np.asarray(fit_series.to_numpy(), dtype=float)
        fit_values = fit_values[np.isfinite(fit_values)]
        if fit_values.size < min_fit_rows:
            continue
        center, scale = fit_location_scale(fit_values)
        z = pd.Series(to_z(scored.to_numpy(), center, scale), index=scored.index)
        z = z.sort_index()
        z = z[~z.index.duplicated(keep="last")]
        smoothed = z.ewm(span=span, adjust=False, ignore_na=True).mean()
        parts[name] = smoothed.clip(lower=0.0).rename(name)
    return parts


def one_sided_max_score(
    fit_channels: dict[str, pd.Series],
    score_channels: dict[str, pd.Series],
    *,
    span: int = RESIDUAL_EWMA_SPAN,
    min_fit_rows: int = 30,
) -> pd.Series:
    """Max across channels of the positive EWMA robust-z.

    Location and scale come from ``fit_channels`` only. A channel that is
    missing on a row is ignored for that row.
    """
    parts = one_sided_channel_scores(
        fit_channels,
        score_channels,
        span=span,
        min_fit_rows=min_fit_rows,
    )
    if not parts:
        return pd.Series(dtype=float, name="rise_score")
    return pd.concat(parts, axis=1).max(axis=1, skipna=True).rename("rise_score")


__all__ = ["one_sided_channel_scores", "one_sided_max_score"]
