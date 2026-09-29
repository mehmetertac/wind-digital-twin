"""Plots for 1-D heat PINN ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from wind_digital_twin.pinn.heat1d import (
    ALPHA,
    HeatTrainingData,
    MLP,
    analytical_solution,
)


def _grid(n: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, n)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    return x, t, xx, tt


@torch.no_grad()
def _predict_on_grid(model: MLP, n: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, t, xx, tt = _grid(n)
    xt = torch.tensor(
        np.column_stack([xx.ravel(), tt.ravel()]),
        dtype=torch.float32,
    )
    u = model(xt).numpy().reshape(n, n)
    return x, t, u


def plot_ablation_slices(
    model_pinn: MLP,
    model_data: MLP,
    data: HeatTrainingData,
    *,
    l2_pinn: float,
    l2_data: float,
    out_path: Path,
    slice_times: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> Path:
    """Headline figure: u(x) slices at several t — truth vs PINN vs data-only."""
    x, t, xx, tt = _grid()
    u_true_grid = analytical_solution(xx, tt, alpha=ALPHA)
    _, _, u_pinn = _predict_on_grid(model_pinn)
    _, _, u_data = _predict_on_grid(model_data)

    ic_x = data.ic_xt[:, 0].detach().numpy()
    ic_u = data.ic_u.detach().numpy().ravel()
    bc_x = data.bc_xt[:, 0].detach().numpy()
    bc_t = data.bc_xt[:, 1].detach().numpy()
    if data.data_xt.numel() > 0:
        sens_x = data.data_xt[:, 0].detach().numpy()
        sens_t = data.data_xt[:, 1].detach().numpy()
    else:
        sens_x = sens_t = np.array([])

    n_slices = len(slice_times)
    fig, axes = plt.subplots(1, n_slices, figsize=(3.2 * n_slices, 3.2), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, t_val in zip(axes, slice_times):
        j = int(np.argmin(np.abs(t - t_val)))
        u_true = u_true_grid[:, j]
        ax.plot(x, u_true, "k-", lw=2, label="Analytical")
        ax.plot(x, u_pinn[:, j], "--", color="C0", lw=1.5, label="PINN")
        ax.plot(x, u_data[:, j], ":", color="C3", lw=1.5, label="Data-only")

        mask_ic = np.abs(ic_x) > 1e-6  # t=0 points at interior x
        if t_val == 0.0 and mask_ic.any():
            ax.scatter(ic_x[mask_ic], ic_u[mask_ic], s=18, c="C2", zorder=5, label="IC data")
        for side in (0.0, 1.0):
            m = np.abs(bc_x - side) < 1e-6
            if m.any():
                idx = np.argmin(np.abs(bc_t[m] - t_val))
                xs = bc_x[m][idx : idx + 1]
                ax.scatter(xs, [0.0], s=18, c="C2", zorder=5)
        if sens_x.size:
            near = np.abs(sens_t - t_val) < 0.08
            if near.any():
                ax.scatter(
                    sens_x[near],
                    data.data_u.detach().numpy().ravel()[near],
                    s=22,
                    c="C2",
                    marker="s",
                    zorder=5,
                    label="Sensors",
                )

        ax.set_title(f"t = {t_val:g}")
        ax.set_xlabel("x")
        ax.set_xlim(0, 1)
        ax.axhline(0, color="0.85", lw=0.5)

    axes[0].set_ylabel("u(x, t)")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(
        f"Heat PINN ablation (rel. L2: PINN={l2_pinn:.2e}, data-only={l2_data:.2e})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_heatmaps(
    model_pinn: MLP,
    model_data: MLP,
    *,
    out_path: Path,
    n: int = 101,
) -> Path:
    """Absolute error |u_pred - u_true| for PINN and data-only."""
    x, t, xx, tt = _grid(n)
    u_true = analytical_solution(xx, tt, alpha=ALPHA)
    _, _, u_pinn = _predict_on_grid(model_pinn, n=n)
    _, _, u_data = _predict_on_grid(model_data, n=n)
    err_pinn = np.abs(u_pinn - u_true)
    err_data = np.abs(u_data - u_true)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, err, title in zip(
        axes,
        (err_pinn, err_data),
        ("PINN |error|", "Data-only |error|"),
    ):
        im = ax.imshow(
            err.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="magma",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loss_curves(
    history_pinn: dict[str, list[float]],
    history_data: dict[str, list[float]],
    *,
    out_path: Path,
) -> Path:
    """Total loss vs step for PINN and data-only."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(history_pinn["total"], label="PINN (w_f=1)", color="C0")
    ax.semilogy(history_data["total"], label="Data-only (w_f=0)", color="C3")
    ax.set_xlabel("Logged step")
    ax.set_ylabel("Total loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
