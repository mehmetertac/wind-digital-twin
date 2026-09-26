"""Diagnostic plots for gearbox thermal residuals."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wind_digital_twin.config import RESULTS_DIR
from wind_digital_twin.utils import to_utc


def plot_gearbox_residual_trajectories(
    turbine_id: str,
    target_label: str,
    failure_time: pd.Timestamp | None,
    residual_frame: pd.DataFrame,
    power_series: pd.Series | None = None,
    lookback_days: int = 90,
    out_path: Path | None = None,
) -> Path:
    """Plot actual vs predicted, residual trajectory, and residual vs power."""
    if failure_time is not None:
        failure_time = to_utc(failure_time)
        window_start = failure_time - timedelta(days=lookback_days)
        window = residual_frame[
            (residual_frame.index >= window_start)
            & (residual_frame.index <= failure_time)
        ]
    else:
        window = residual_frame
        failure_time = (
            window.index[-1] if not window.empty else pd.Timestamp.now(tz="UTC")
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    days_before = (window.index - failure_time).total_seconds() / 86400.0

    ax0 = axes[0]
    ax0.plot(days_before, window["actual"].values, lw=0.8, label="Actual", alpha=0.9)
    ax0.plot(days_before, window["predicted"].values, lw=0.8, label="Predicted", alpha=0.9)
    ax0.set_ylabel("Temperature (°C)")
    ax0.set_title(f"{turbine_id} — {target_label}: actual vs predicted")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(
        days_before,
        window["residual"].values,
        lw=0.8,
        color="tab:orange",
        label="Residual",
    )
    ax1.axhline(0.0, color="black", ls="--", lw=0.8)
    if failure_time is not None:
        ax1.axvline(0.0, color="red", ls="-", lw=1.0, label="Failure")
    ax1.set_ylabel("Residual (°C)")
    ax1.set_title("Predicted − actual (negative = hotter than expected)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    if power_series is not None:
        power_window = power_series.reindex(window.index)
        sc = ax2.scatter(
            power_window.values,
            window["residual"].values,
            c=days_before,
            cmap="viridis",
            s=8,
            alpha=0.6,
        )
        plt.colorbar(sc, ax=ax2, label="Days before failure")
        ax2.set_xlabel("Power (kW)")
    else:
        ax2.scatter(range(len(window)), window["residual"].values, s=8, alpha=0.6)
        ax2.set_xlabel("Sample index")
    ax2.axhline(0.0, color="black", ls="--", lw=0.8)
    ax2.set_ylabel("Residual (°C)")
    ax2.set_title("Residual vs power")
    ax2.grid(True, alpha=0.3)

    axes[1].set_xlabel("Days before failure")
    title_suffix = f"({lookback_days}d window)" if failure_time is not None else ""
    fig.suptitle(f"{turbine_id} — {target_label} thermal residuals {title_suffix}")
    fig.tight_layout()

    if out_path is None:
        slug = target_label.replace(" ", "_").lower()
        out_path = RESULTS_DIR / "plots" / f"residual_{turbine_id}_{slug}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
