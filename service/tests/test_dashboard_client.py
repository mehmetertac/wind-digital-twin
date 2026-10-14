"""Dashboard series filter helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_dashboard = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(_dashboard))

from client import filter_series_to_date  # noqa: E402


def test_filter_series_to_date_keeps_prefix():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2017-06-01", periods=5, freq="D", tz="UTC"),
            "rul_p50": [80.0, 70.0, 60.0, 50.0, 40.0],
        }
    )
    filtered = filter_series_to_date(frame, "2017-06-03T00:00:00Z")
    assert len(filtered) == 3
    assert float(filtered["rul_p50"].iloc[-1]) == 60.0
