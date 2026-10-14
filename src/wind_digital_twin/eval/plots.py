"""Diagnostic plots for gearbox thermal residuals and twin pass timelines."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_digital_twin.config import RESULTS_DIR
from wind_digital_twin.eval.protocol import AlarmEpisode
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


def plot_twin_timeline(
    turbine_id: str,
    layer_frame: pd.DataFrame,
    failure_time: pd.Timestamp | None,
    threshold: float,
    episodes: list[AlarmEpisode],
    *,
    buffer_days: int = 90,
    lookback_days: int = 120,
    synthetic: bool = True,
    failure_remarks: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """Headline figure: all twin layers around failure (days before failure on x)."""
    if failure_time is not None:
        failure_time = to_utc(failure_time)
        window_start = failure_time - timedelta(days=lookback_days)
        window = layer_frame.loc[
            (layer_frame.index >= window_start)
            & (layer_frame.index <= failure_time)
        ]
        buffer_start = failure_time - timedelta(days=buffer_days)
    else:
        window = layer_frame.iloc[-lookback_days * 144 :]
        failure_time = window.index[-1] if not window.empty else pd.Timestamp.now(tz="UTC")
        buffer_start = None

    if window.empty:
        raise ValueError("Empty timeline window for twin pass plot")

    days_before = (window.index - failure_time).total_seconds() / 86400.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax0 = axes[0]
    ax0.plot(days_before, window["bear_actual"], lw=0.7, label="Bear actual", alpha=0.9)
    ax0.plot(days_before, window["oil_actual"], lw=0.7, label="Oil actual", alpha=0.7)
    ax0.set_ylabel("Temperature (°C)")
    ax0.set_title("SCADA gearbox temperatures")
    ax0.legend(loc="upper left", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(
        days_before,
        window["bear_physics_pred"],
        lw=0.7,
        ls="--",
        label="Physics expected (bear)",
    )
    ax1.plot(
        days_before,
        window["bear_hybrid_pred"],
        lw=0.7,
        label="Hybrid expected (bear)",
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Expected bearing temperature (same ODE backbone)")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    ax2.plot(
        days_before,
        window["bear_physics_residual"],
        lw=0.7,
        label="Physics residual",
        alpha=0.85,
    )
    ax2.plot(
        days_before,
        window["bear_hybrid_residual"],
        lw=0.7,
        label="Hybrid residual",
        alpha=0.85,
    )
    ax2.axhline(0.0, color="black", ls="--", lw=0.8)
    ax2.set_ylabel("Residual (°C)")
    ax2.set_title("Predicted − actual (negative = hotter than expected)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[3]
    score_vals = window["anomaly_score"].values
    ax3.plot(days_before, score_vals, lw=0.7, color="tab:red", label="Anomaly score")
    ax3.axhline(threshold, color="gray", ls=":", lw=1.0, label=f"Threshold ({threshold:.3f})")
    for ep in episodes:
        if ep.end < window.index[0] or ep.start > window.index[-1]:
            continue
        ep_start = (ep.start - failure_time).total_seconds() / 86400.0
        ep_end = (ep.end - failure_time).total_seconds() / 86400.0
        ax3.axvspan(ep_start, ep_end, color="red", alpha=0.12)
    ax3.set_ylabel("Score")
    ax3.set_xlabel("Days before failure")
    ax3.set_title("Hybrid-residual Isolation Forest score")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)

    if buffer_start is not None and buffer_start >= window.index[0]:
        buf_days = (buffer_start - failure_time).total_seconds() / 86400.0
        for ax in axes[:3]:
            ax.axvspan(buf_days, 0.0, color="gray", alpha=0.08, zorder=0)
        axes[0].text(
            buf_days * 0.5,
            ax0.get_ylim()[1],
            "90d train holdout",
            fontsize=7,
            va="top",
            ha="center",
            alpha=0.7,
        )

    for ax in axes:
        ax.axvline(0.0, color="red", ls="-", lw=1.0, alpha=0.8)

    data_label = "synthetic EDP" if synthetic else "real EDP"
    remark = f" — {failure_remarks}" if failure_remarks else ""
    fig.suptitle(
        f"{turbine_id} hybrid digital twin ({data_label}){remark}\n"
        f"{lookback_days}d window ending at logged failure",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = RESULTS_DIR / "plots" / f"twin_timeline_{turbine_id}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hybrid_vs_physics_rmse(
    corrector_metrics: dict[str, Any],
    *,
    synthetic: bool = True,
    out_path: Path | None = None,
) -> Path:
    """Grouped bar chart: healthy validation RMSE — physics ODE vs hybrid corrector."""
    targets = corrector_metrics.get("targets", {})
    labels = ["Oil", "Bear"]
    slugs = ["oil", "bear"]
    phys = [targets.get(s, {}).get("physics_val_rmse", float("nan")) for s in slugs]
    hybrid = [targets.get(s, {}).get("hybrid_val_rmse", float("nan")) for s in slugs]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, phys, width, label="Physics ODE", color="C0")
    ax.bar(x + width / 2, hybrid, width, label="Hybrid (ODE + GBM)", color="C2")
    ax.set_ylabel("Validation RMSE (°C)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    data_label = "synthetic healthy holdout" if synthetic else "real healthy holdout"
    ax.set_title(f"Thermal expectation fit — {data_label}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (p, h) in enumerate(zip(phys, hybrid)):
        if np.isfinite(p) and np.isfinite(h) and p > 0:
            pct = 100.0 * (p - h) / p
            ax.text(i, max(p, h) + 0.05, f"{pct:.0f}% ↓", ha="center", fontsize=8)

    fig.tight_layout()
    if out_path is None:
        out_path = RESULTS_DIR / "plots" / "hybrid_vs_physics_rmse.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
