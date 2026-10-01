"""Optional bounded neural equilibrium correction (requires torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_digital_twin.config import POWER_COLUMN, THERMAL_NN_CORRECTION_CAP_C
from wind_digital_twin.physics.lumped_ode import (
    EquilibriumCorrectionFn,
    LumpedThermalParams,
    equilibrium_temperature,
    simulate_temperature,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Neural correction requires torch; pip install -e '.[pinn]'"
        ) from exc
    return torch, nn


@dataclass
class EquilibriumCorrectionMLP:
    """Tiny MLP; outputs raw g, scaled by cap * tanh(g) in correction."""

    state_dict_bytes: bytes
    input_mean: np.ndarray
    input_std: np.ndarray

    def correction_fn(self) -> EquilibriumCorrectionFn:
        torch, nn = _require_torch()
        import io

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(3, 8),
                    nn.Tanh(),
                    nn.Linear(8, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = _MLP()
        buf = io.BytesIO(self.state_dict_bytes)
        model.load_state_dict(torch.load(buf, weights_only=True))
        model.eval()
        cap = THERMAL_NN_CORRECTION_CAP_C
        mean = self.input_mean
        std = np.where(self.input_std > 1e-8, self.input_std, 1.0)

        def fn(power: np.ndarray, rpm: np.ndarray, nac: np.ndarray) -> np.ndarray:
            x = np.column_stack([power, rpm, nac]).astype(np.float32)
            x_norm = (x - mean) / std
            with torch.no_grad():
                g = model(torch.from_numpy(x_norm)).numpy()
            return cap * np.tanh(g)

        return fn


def fit_equilibrium_correction(
    train_df: pd.DataFrame,
    params: LumpedThermalParams,
    target_column: str,
    epochs: int = 80,
    lr: float = 1e-3,
) -> EquilibriumCorrectionMLP:
    """
    Fit MLP on frozen physics params to reduce train residuals on T_eq only.

    Physics ODE parameters are not updated during NN training.
    """
    torch, nn = _require_torch()

    power = train_df[POWER_COLUMN].to_numpy(dtype=float)
    rpm = train_df["Rtr_RPM_Avg"].to_numpy(dtype=float)
    nac = train_df["Nac_Temp_Avg"].to_numpy(dtype=float)
    y = train_df[target_column].to_numpy(dtype=float)

    t_eq_base = equilibrium_temperature(nac, power, rpm, params, correction=None)
    pred_physics = simulate_temperature(train_df, params, target_column, correction=None)
    target_delta = y - pred_physics

    features = np.column_stack([power, rpm, nac]).astype(np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    x_norm = (features - mean) / np.where(std > 1e-8, std, 1.0)

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 8),
                nn.Tanh(),
                nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cap = THERMAL_NN_CORRECTION_CAP_C
    x_t = torch.from_numpy(x_norm)
    target_t = torch.from_numpy(target_delta.astype(np.float32))

    for _ in range(epochs):
        opt.zero_grad()
        g = model(x_t)
        corr = cap * torch.tanh(g)
        loss = torch.mean((corr - target_t) ** 2)
        loss.backward()
        opt.step()

    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    del t_eq_base
    return EquilibriumCorrectionMLP(
        state_dict_bytes=buf.getvalue(),
        input_mean=mean,
        input_std=std,
    )


__all__ = ["EquilibriumCorrectionMLP", "fit_equilibrium_correction"]
