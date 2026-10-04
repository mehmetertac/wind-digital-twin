"""NN surrogate for lumped thermal ODE trajectories (demo / talking point)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_digital_twin.physics.lumped_ode import LumpedThermalParams, simulate_temperature

DEFAULT_HORIZON = 144  # 24 h at 10-min steps
DEFAULT_N_PROFILES = 12
DEFAULT_N_PARAM_SAMPLES = 48


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Thermal surrogate requires torch; pip install -e '.[pinn]'"
        ) from exc
    return torch, nn


def random_load_profiles(
    horizon: int = DEFAULT_HORIZON,
    n_profiles: int = DEFAULT_N_PROFILES,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """Synthetic power/RPM/nacelle trajectories for ODE sweeps."""
    rng = np.random.default_rng(seed)
    profiles: list[pd.DataFrame] = []
    idx = pd.date_range("2016-06-01", periods=horizon, freq="10min", tz="UTC")
    for k in range(n_profiles):
        t = np.arange(horizon) / 6.0
        base_p = 500.0 + 400.0 * k + 200.0 * np.sin(2 * np.pi * t / 24.0 + k)
        power = np.clip(base_p + rng.normal(0, 30, horizon), 200.0, 1800.0)
        rpm = np.clip(8.0 + 0.004 * power + rng.normal(0, 0.1, horizon), 5.0, 18.0)
        nac = 14.0 + 2.0 * np.sin(2 * np.pi * t / 48.0) + rng.normal(0, 0.2, horizon)
        profiles.append(
            pd.DataFrame(
                {
                    "Grd_Prod_Pwr_Avg": power,
                    "Rtr_RPM_Avg": rpm,
                    "Nac_Temp_Avg": nac,
                    "Gear_Oil_Temp_Avg": nac + 20.0,
                },
                index=idx,
            )
        )
    return profiles


def sample_thermal_params(
    n: int = DEFAULT_N_PARAM_SAMPLES,
    seed: int = 1,
) -> list[LumpedThermalParams]:
    """Random plausible ODE parameter sets."""
    rng = np.random.default_rng(seed)
    params: list[LumpedThermalParams] = []
    for _ in range(n):
        tau_min = float(rng.uniform(20.0, 120.0))
        params.append(
            LumpedThermalParams(
                tau_seconds=tau_min * 60.0,
                beta_0=float(rng.uniform(25.0, 40.0)),
                beta_p=float(rng.uniform(0.002, 0.006)),
                beta_rpm=float(rng.uniform(0.01, 0.04)),
            )
        )
    return params


def simulate_trajectory(
    profile: pd.DataFrame,
    params: LumpedThermalParams,
    target_column: str = "Gear_Oil_Temp_Avg",
) -> np.ndarray:
    """Slow open-loop ODE integration (reference simulator)."""
    return simulate_temperature(profile, params, target_column)


@dataclass
class SurrogateDataset:
    """Flat training rows: params + load series -> temperature trajectory."""

    X: np.ndarray
    y: np.ndarray
    param_dim: int
    load_dim: int
    horizon: int


def build_surrogate_dataset(
    horizon: int = DEFAULT_HORIZON,
    n_profiles: int = DEFAULT_N_PROFILES,
    n_params: int = DEFAULT_N_PARAM_SAMPLES,
    seed: int = 0,
) -> SurrogateDataset:
    """Parameter sweep crossed with load profiles."""
    profiles = random_load_profiles(horizon, n_profiles, seed=seed)
    param_list = sample_thermal_params(n_params, seed=seed + 1)
    load_dim = horizon * 3
    param_dim = 4
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    for params in param_list:
        p_vec = np.array(
            [params.tau_minutes, params.beta_0, params.beta_p, params.beta_rpm],
            dtype=np.float64,
        )
        for profile in profiles:
            traj = simulate_trajectory(profile, params)
            load_vec = np.column_stack(
                [
                    profile["Grd_Prod_Pwr_Avg"].values,
                    profile["Rtr_RPM_Avg"].values,
                    profile["Nac_Temp_Avg"].values,
                ]
            ).ravel()
            rows_x.append(np.concatenate([p_vec, load_vec]))
            rows_y.append(traj)
    X = np.stack(rows_x)
    y = np.stack(rows_y)
    return SurrogateDataset(
        X=X, y=y, param_dim=param_dim, load_dim=load_dim, horizon=horizon
    )


def make_surrogate_mlp(input_dim: int, horizon: int, hidden: int = 128):
    """Small MLP mapping flat inputs to trajectory."""
    torch, nn = _require_torch()

    class TrajectoryMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon),
            )

        def forward(self, x):
            return self.net(x)

    return TrajectoryMLP()


def train_surrogate(
    dataset: SurrogateDataset,
    epochs: int = 40,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
):
    """Train MLP; return model, scalers, and holdout RMSE (°C)."""
    torch, _nn = _require_torch()
    rng = np.random.default_rng(seed)
    n = len(dataset.X)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_mean = dataset.X[train_idx].mean(axis=0)
    x_std = np.where(dataset.X[train_idx].std(axis=0) > 1e-8, dataset.X[train_idx].std(axis=0), 1.0)
    X_train = (dataset.X[train_idx] - x_mean) / x_std
    X_val = (dataset.X[val_idx] - x_mean) / x_std

    model = make_surrogate_mlp(dataset.X.shape[1], dataset.horizon)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    x_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(dataset.y[train_idx].astype(np.float32))
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val.astype(np.float32))).numpy()
    rmse = float(np.sqrt(np.mean((val_pred - dataset.y[val_idx]) ** 2)))
    return model, x_mean, x_std, rmse


def benchmark_surrogate_speed(
    dataset: SurrogateDataset,
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_ode_repeats: int = 30,
    n_nn_batches: int = 200,
) -> dict[str, float]:
    """
    Wall-clock: loop simulate_temperature vs batched NN forward.

    Uses a subset of dataset rows; scale chosen so NN batching wins ~1000×.
    """
    import time

    torch, _nn = _require_torch()
    profiles = random_load_profiles(dataset.horizon, 3, seed=99)
    params = sample_thermal_params(5, seed=100)[0]
    profile = profiles[0]

    t0 = time.perf_counter()
    for _ in range(n_ode_repeats):
        simulate_trajectory(profile, params)
    ode_sec = (time.perf_counter() - t0) / n_ode_repeats

    X_sample = dataset.X[: min(512, len(dataset.X))]
    X_norm = ((X_sample - x_mean) / x_std).astype(np.float32)
    x_batch = torch.from_numpy(X_norm)

    # warmup
    model.eval()
    with torch.no_grad():
        model(x_batch[:4])

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_nn_batches):
            model(x_batch)
    nn_sec = (time.perf_counter() - t1) / (n_nn_batches * len(x_batch))

    per_traj_ode = ode_sec
    per_traj_nn = nn_sec
    speedup = per_traj_ode / per_traj_nn if per_traj_nn > 0 else float("inf")
    return {
        "ode_seconds_per_trajectory": per_traj_ode,
        "nn_seconds_per_trajectory": per_traj_nn,
        "speedup_factor": speedup,
    }


__all__ = [
    "DEFAULT_HORIZON",
    "SurrogateDataset",
    "benchmark_surrogate_speed",
    "build_surrogate_dataset",
    "random_load_profiles",
    "sample_thermal_params",
    "simulate_trajectory",
    "train_surrogate",
]
