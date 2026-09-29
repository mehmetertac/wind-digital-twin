"""Physics-informed neural network toy models."""

from wind_digital_twin.pinn.heat1d import (
    ALPHA,
    MLP,
    HeatTrainingData,
    analytical_solution,
    make_model,
    pde_residual,
    relative_l2,
    sample_training_data,
    train,
)

__all__ = [
    "ALPHA",
    "MLP",
    "HeatTrainingData",
    "analytical_solution",
    "make_model",
    "pde_residual",
    "relative_l2",
    "sample_training_data",
    "train",
]
