"""HTTP client for the wind digital twin FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

DEFAULT_API_URL = "http://127.0.0.1:8000"


def api_url() -> str:
    return os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")


def fetch_health() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/health")
        response.raise_for_status()
        return response.json()


def fetch_turbines() -> list[str]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/twin/turbines")
        response.raise_for_status()
        return response.json()["turbines"]


def fetch_attributions() -> list[dict[str, Any]]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{api_url()}/twin/attributions")
        response.raise_for_status()
        return response.json()["attributions"]


def fetch_series(*, turbine_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{api_url()}/twin/series", params={"turbine": turbine_id})
        response.raise_for_status()
        return response.json()


def filter_series_to_date(rows: list[dict[str, Any]] | pd.DataFrame, selected_date: str) -> pd.DataFrame:
    """Keep daily rows up to ``selected_date`` for timeline scrubbing."""
    if isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        df = rows.copy()
    if df.empty:
        return df
    day = pd.to_datetime(selected_date, utc=True).floor("D")
    dates = pd.to_datetime(df["date"], utc=True)
    return df.loc[dates <= day].copy().sort_values("date").reset_index(drop=True)
