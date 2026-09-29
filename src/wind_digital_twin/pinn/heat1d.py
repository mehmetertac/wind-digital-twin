"""1-D heat equation PINN: u_t = alpha * u_xx on [0,1] x [0,1]."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

ALPHA = 0.1


def _sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, d))
    n_sobol = 1 << (n - 1).bit_length()
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    return sampler.random(n=n_sobol)[:n]


def analytical_solution(
    x: torch.Tensor | np.ndarray,
    t: torch.Tensor | np.ndarray,
    alpha: float = ALPHA,
) -> torch.Tensor | np.ndarray:
    """u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)."""
    if isinstance(x, torch.Tensor):
        return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    return np.exp(-alpha * math.pi**2 * t_arr) * np.sin(math.pi * x_arr)


class MLP(nn.Module):
    """2 -> 4x32 tanh -> 1."""

    def __init__(self, hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


def make_model(seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    return MLP()


@dataclass
class HeatTrainingData:
    ic_xt: torch.Tensor
    ic_u: torch.Tensor
    bc_xt: torch.Tensor
    bc_u: torch.Tensor
    data_xt: torch.Tensor
    data_u: torch.Tensor
    f_xt: torch.Tensor


def sample_training_data(
    n_ic: int = 100,
    n_bc: int = 100,
    n_data: int = 10,
    n_f: int = 2000,
    noise: float = 0.01,
    seed: int = 0,
) -> HeatTrainingData:
    """Sample IC, BC, optional interior labels, and Sobol collocation points."""
    rng = np.random.default_rng(seed)

    x_ic = rng.uniform(0.0, 1.0, size=n_ic)
    t_ic = np.zeros(n_ic)
    u_ic = np.sin(math.pi * x_ic)

    t_bc = rng.uniform(0.0, 1.0, size=n_bc)
    x0 = np.zeros(n_bc)
    x1 = np.ones(n_bc)
    bc_xt = np.vstack(
        [
            np.column_stack([x0, t_bc]),
            np.column_stack([x1, t_bc]),
        ]
    )
    bc_u = np.zeros(2 * n_bc)

    if n_data > 0:
        data_raw = _sobol_unit(n_data, d=2, seed=seed)
        data_xt = qmc.scale(data_raw, [0.0, 0.0], [1.0, 1.0])
        u_clean = analytical_solution(data_xt[:, 0], data_xt[:, 1])
        u_data = u_clean + rng.normal(0.0, noise, size=n_data)
    else:
        data_xt = np.empty((0, 2))
        u_data = np.empty(0)

    f_raw = _sobol_unit(n_f, d=2, seed=seed + 1)
    f_xt = qmc.scale(f_raw, [0.0, 0.0], [1.0, 1.0])

    ic_xt = np.column_stack([x_ic, t_ic])

    def _to_xt(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    def _to_u(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).reshape(-1, 1)

    return HeatTrainingData(
        ic_xt=_to_xt(ic_xt),
        ic_u=_to_u(u_ic),
        bc_xt=_to_xt(bc_xt),
        bc_u=_to_u(bc_u),
        data_xt=_to_xt(data_xt),
        data_u=_to_u(u_data),
        f_xt=_to_xt(f_xt),
    )


def pde_residual(model: MLP, xt: torch.Tensor, alpha: float = ALPHA) -> torch.Tensor:
    """u_t - alpha * u_xx at collocation points."""
    u = model(xt)
    grad_u = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    grad_ux = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_xx = grad_ux[:, 0:1]
    return u_t - alpha * u_xx


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    return torch.mean((pred - target) ** 2)


def _loss_terms(
    model: MLP,
    data: HeatTrainingData,
    alpha: float,
    w_f: float,
    w_ic: float,
    w_bc: float,
    w_data: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse_ic = _mse(model(data.ic_xt), data.ic_u)
    mse_bc = _mse(model(data.bc_xt), data.bc_u)
    mse_data = _mse(model(data.data_xt), data.data_u)
    if w_f > 0:
        f = pde_residual(model, data.f_xt, alpha=alpha)
        mse_f = torch.mean(f**2)
    else:
        mse_f = torch.tensor(0.0, dtype=mse_ic.dtype)
    total = w_f * mse_f + w_ic * mse_ic + w_bc * mse_bc + w_data * mse_data
    terms = {
        "f": float(mse_f.detach()),
        "ic": float(mse_ic.detach()),
        "bc": float(mse_bc.detach()),
        "data": float(mse_data.detach()),
        "total": float(total.detach()),
    }
    return total, terms


def train(
    model: MLP,
    data: HeatTrainingData,
    *,
    alpha: float = ALPHA,
    w_f: float = 1.0,
    w_ic: float = 1.0,
    w_bc: float = 1.0,
    w_data: float = 1.0,
    adam_steps: int = 5000,
    adam_lr: float = 1e-3,
    lbfgs_max_iter: int = 500,
    log_every: int = 500,
) -> dict[str, list[float]]:
    """Adam pretrain then L-BFGS; returns per-term loss history."""
    history: dict[str, list[float]] = {
        "f": [],
        "ic": [],
        "bc": [],
        "data": [],
        "total": [],
    }
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    for step in range(1, adam_steps + 1):
        opt.zero_grad()
        loss, terms = _loss_terms(
            model, data, alpha, w_f, w_ic, w_bc, w_data
        )
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == adam_steps:
            for k in history:
                history[k].append(terms[k])

    if lbfgs_max_iter <= 0:
        return history

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, terms = _loss_terms(
            model, data, alpha, w_f, w_ic, w_bc, w_data
        )
        loss.backward()
        last_terms["update"] = terms
        return loss

    last_terms: dict[str, Any] = {}
    lbfgs.step(closure)
    if "update" in last_terms:
        for k in history:
            history[k].append(last_terms["update"][k])

    return history


@torch.no_grad()
def relative_l2(
    model: MLP,
    alpha: float = ALPHA,
    n: int = 101,
) -> float:
    """Relative L2 error vs analytical solution on an n x n grid."""
    x = torch.linspace(0.0, 1.0, n)
    t = torch.linspace(0.0, 1.0, n)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    u_pred = model(xt).reshape(n, n)
    u_true = analytical_solution(xx, tt, alpha=alpha)
    num = torch.linalg.norm(u_pred - u_true)
    den = torch.linalg.norm(u_true)
    return float(num / den)
