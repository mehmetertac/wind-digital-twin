"""Fixtures for twin service API tests."""

from __future__ import annotations

from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_edp() -> None:
    from wind_digital_twin.config import DATA_RAW
    from wind_digital_twin.data.synthetic_edp import generate_synthetic_edp_dataset

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not list(DATA_RAW.glob("*.csv")):
        generate_synthetic_edp_dataset(DATA_RAW, seed=42, force=True)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    if not (MODELS_DIR / "hybrid_pipeline.joblib").exists():
        pytest.skip("Run service/scripts/export_models.py to create serving artifacts")
    return MODELS_DIR
