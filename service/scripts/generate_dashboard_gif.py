"""Generate docs/figures/twin_dashboard.png and twin_dashboard.gif from timeline export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(__file__).resolve().parents[1] / "models"
TIMELINE = MODELS / "timeline_demo.parquet"
SHAP_CSV = MODELS / "shap_sensor_ranking.csv"
OUT_PNG = ROOT / "docs" / "figures" / "twin_dashboard.png"
OUT_GIF = ROOT / "docs" / "figures" / "twin_dashboard.gif"


def _p10_attributions() -> pd.DataFrame:
    ranking = pd.read_csv(SHAP_CSV)
    work = ranking.loc[ranking["learner"] == "p10"].sort_values("rank").head(5)
    return work.iloc[::-1]


def _frame(
    df: pd.DataFrame,
    *,
    day_idx: int,
    attrs: pd.DataFrame,
    dpi: int = 100,
) -> Image.Image:
    view = df.iloc[: day_idx + 1]
    if view.empty:
        view = df.iloc[:1]
    dates = np.arange(len(view))
    threshold = float(view["anomaly_threshold"].iloc[0])

    fig = plt.figure(figsize=(10, 9), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 0.9, 0.9, 0.85], width_ratios=[3, 1.2], hspace=0.35, wspace=0.35)
    fig.suptitle(
        f"{view['turbine_id'].iloc[0]} — failure window scrub (day {day_idx + 1}/{len(df)})",
        fontsize=12,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(dates, view["oil_temp_actual"], color="black", label="Actual oil")
    ax0.plot(dates, view["oil_temp_physics"], color="#E45756", label="Physics expected")
    ax0.set_ylabel("°C")
    ax0.legend(loc="upper left", fontsize=7)
    ax0.grid(alpha=0.25)
    if "oil_residual" in view.columns:
        ax0r = ax0.twinx()
        ax0r.plot(dates, view["oil_residual"], color="#72B7B2", linestyle="--", label="Hybrid residual")
        ax0r.set_ylabel("Residual (°C)", fontsize=8)
        ax0r.legend(loc="upper right", fontsize=7)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(dates, view["anomaly_score"], color="#4C78A8", label="Anomaly score")
    ax1.axhline(threshold, color="#F58518", linestyle="--", linewidth=1, label="Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[2, 0])
    if "rul_p50" in view.columns:
        ax2.fill_between(
            dates,
            view["cqr80_lo"],
            view["cqr80_hi"],
            alpha=0.25,
            color="#4C78A8",
            label="CQR 80%",
        )
        ax2.plot(dates, view["rul_p50"], color="#4C78A8", linewidth=2, label="RUL P50")
    ax2.set_ylabel("Days")
    ax2.set_xlabel("Days before failure (index)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[0:3, 1])
    y_labels = [str(s)[:28] for s in attrs["sensor"]]
    ax3.barh(y_labels, attrs["mean_abs_shap"], color="#72B7B2")
    ax3.set_xlabel("Mean |SHAP| (P10)")
    ax3.set_title("Sensor attributions", fontsize=9)
    ax3.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf)


def main() -> None:
    if not TIMELINE.exists():
        raise SystemExit(f"Missing {TIMELINE}; run service/scripts/export_models.py first")
    if not SHAP_CSV.exists():
        raise SystemExit(f"Missing {SHAP_CSV}; run service/scripts/export_models.py first")
    df = pd.read_parquet(TIMELINE)
    attrs = _p10_attributions()
    t06 = df.loc[df["turbine_id"] == "T06"].sort_values("date").reset_index(drop=True)
    if t06.empty:
        t06 = df.sort_values("date").reset_index(drop=True)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    full = _frame(t06, day_idx=len(t06) - 1, attrs=attrs, dpi=120)
    full.save(OUT_PNG)

    indices = np.linspace(5, len(t06) - 1, num=min(8, len(t06)), dtype=int)
    frames = [_frame(t06, day_idx=int(i), attrs=attrs) for i in indices]
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_PNG} and {OUT_GIF}")


if __name__ == "__main__":
    main()
