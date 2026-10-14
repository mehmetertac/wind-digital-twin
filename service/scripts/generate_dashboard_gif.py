"""Generate docs/figures/twin_dashboard.png and twin_dashboard.gif from timeline export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TIMELINE = Path(__file__).resolve().parents[1] / "models" / "timeline_demo.parquet"
OUT_PNG = ROOT / "docs" / "figures" / "twin_dashboard.png"
OUT_GIF = ROOT / "docs" / "figures" / "twin_dashboard.gif"


def _frame(df: pd.DataFrame, *, day_idx: int, dpi: int = 100) -> Image.Image:
    view = df.iloc[: day_idx + 1]
    if view.empty:
        view = df.iloc[:1]
    dates = np.arange(len(view))
    threshold = float(view["anomaly_threshold"].iloc[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        f"{view['turbine_id'].iloc[0]} — failure window scrub (day {day_idx + 1}/{len(df)})",
        fontsize=12,
    )

    ax0, ax1, ax2 = axes
    ax0.plot(dates, view["oil_temp_actual"], color="black", label="Actual oil")
    ax0.plot(dates, view["oil_temp_physics"], color="#E45756", label="Physics expected")
    ax0.set_ylabel("°C")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(alpha=0.25)

    ax1.plot(dates, view["anomaly_score"], color="#4C78A8", label="Anomaly score")
    ax1.axhline(threshold, color="#F58518", linestyle="--", linewidth=1, label="Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)

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
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf)


def main() -> None:
    if not TIMELINE.exists():
        raise SystemExit(f"Missing {TIMELINE}; run service/scripts/export_models.py first")
    df = pd.read_parquet(TIMELINE)
    t06 = df.loc[df["turbine_id"] == "T06"].sort_values("date").reset_index(drop=True)
    if t06.empty:
        t06 = df.sort_values("date").reset_index(drop=True)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    full = _frame(t06, day_idx=len(t06) - 1, dpi=120)
    full.save(OUT_PNG)

    indices = np.linspace(5, len(t06) - 1, num=min(8, len(t06)), dtype=int)
    frames = [_frame(t06, day_idx=int(i)) for i in indices]
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
