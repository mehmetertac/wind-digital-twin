"""FastAPI twin scoring contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from twin_service.api.app import create_app
from twin_service.artifacts import load_artifacts
from twin_service.constants import DAILY_FEATURE_COLUMNS, MIN_SCADA_ROWS


def _load_client(models_dir: Path) -> TestClient:
    artifacts = load_artifacts(models_dir)
    app = create_app(load_fn=lambda: artifacts)
    return TestClient(app)


def _sample_scada_window(models_dir: Path) -> list[dict]:
    timeline = pd.read_parquet(models_dir / "timeline_demo.parquet")
    tid = str(timeline["turbine_id"].iloc[0])
    turbines, _ = __import__(
        "wind_digital_twin.data.load_edp",
        fromlist=["load_edp_dataset"],
    ).load_edp_dataset(
        __import__("wind_digital_twin.config", fromlist=["DATA_RAW"]).DATA_RAW
    )
    if tid not in turbines:
        pytest.skip("Synthetic turbine missing from EDP")
    from wind_digital_twin.data.clean import clean_turbine_df
    from wind_digital_twin.config import THERMAL_MIN_POWER_KW

    clean = clean_turbine_df(turbines[tid], min_power_kw=THERMAL_MIN_POWER_KW)
    tail = clean.tail(MIN_SCADA_ROWS + 20)
    rows = []
    for ts, row in tail.iterrows():
        payload = {
            "timestamp": pd.Timestamp(ts).isoformat().replace("+00:00", "Z"),
            "Gear_Oil_Temp_Avg": float(row["Gear_Oil_Temp_Avg"]),
            "Gear_Bear_Temp_Avg": float(row["Gear_Bear_Temp_Avg"]),
            "Grd_Prod_Pwr_Avg": float(row["Grd_Prod_Pwr_Avg"]),
            "Rtr_RPM_Avg": float(row["Rtr_RPM_Avg"]),
            "Nac_Temp_Avg": float(row["Nac_Temp_Avg"]),
        }
        for col in row.index:
            if col not in payload:
                payload[col] = float(row[col])
        rows.append(payload)
    return rows


def test_twin_score_endpoint(models_dir: Path):
    client = _load_client(models_dir)
    body = {"turbine_id": "T06", "rows": _sample_scada_window(models_dir)}
    response = client.post("/twin/score", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["anomaly_threshold"], float)
    assert len(payload["points"]) >= 1
    point = payload["points"][0]
    assert "expected_temp_c" in point
    assert "residual_c" in point
    assert "anomaly_score" in point


def test_twin_rul_endpoint(models_dir: Path):
    client = _load_client(models_dir)
    features = {col: 0.5 for col in DAILY_FEATURE_COLUMNS}
    features["anomaly_score_mean"] = 2.0
    features["anomaly_score_max"] = 3.0
    body = {"turbine_id": "T06", "date": "2017-07-01T00:00:00Z", **features}
    response = client.post("/twin/rul", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["rul_p10"] <= payload["rul_p50"] <= payload["rul_p90"]
    assert len(payload["attributions"]) == 5
    assert payload["cqr80_lo"] <= payload["cqr80_hi"]


def test_health_ok(models_dir: Path):
    client = _load_client(models_dir)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]
