"""Pydantic payloads for twin scoring and RUL endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from twin_service.constants import DAILY_FEATURE_COLUMNS, MIN_SCADA_ROWS, SCADA_COLUMNS


class ScadaRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str
    Gear_Oil_Temp_Avg: float
    Gear_Bear_Temp_Avg: float
    Grd_Prod_Pwr_Avg: float
    Rtr_RPM_Avg: float
    Nac_Temp_Avg: float


class TwinScoreRequest(BaseModel):
    """Ten-minute SCADA window for hybrid scoring."""

    turbine_id: str = Field(default="T06", description="Turbine identifier (metadata only).")
    rows: list[ScadaRow] = Field(..., min_length=MIN_SCADA_ROWS)

    @field_validator("turbine_id")
    @classmethod
    def strip_turbine(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("turbine_id must not be empty")
        return text


class ScoredPoint(BaseModel):
    timestamp: str
    expected_temp_c: float
    residual_c: float
    anomaly_score: float


class TwinScoreResponse(BaseModel):
    turbine_id: str
    anomaly_threshold: float
    points: list[ScoredPoint]


class TwinRulRequest(BaseModel):
    """One turbine-day feature vector for quantile RUL."""

    model_config = ConfigDict(extra="forbid")

    turbine_id: str = Field(default="T06")
    date: str | None = Field(default=None, description="Optional UTC calendar day label.")
    oil_deg_mean: float
    oil_deg_std: float = 0.0
    bear_deg_mean: float
    bear_deg_std: float = 0.0
    anomaly_score_mean: float
    anomaly_score_max: float
    oil_deg_slope_7d: float = 0.0
    bear_deg_slope_7d: float = 0.0
    theta_oil_mean: float
    theta_bear_mean: float
    delta_bear_oil_mean: float

    @field_validator("turbine_id")
    @classmethod
    def strip_turbine(cls, value: str) -> str:
        return value.strip() or "T06"


class SensorAttribution(BaseModel):
    sensor: str
    learner: str
    mean_shap: float
    mean_abs_shap: float
    rank: int


class TwinRulResponse(BaseModel):
    turbine_id: str
    date: str | None
    rul_p10: float
    rul_p50: float
    rul_p90: float
    cqr80_lo: float
    cqr80_hi: float
    cqr90_lo: float
    cqr90_hi: float
    attributions: list[SensorAttribution]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    hybrid_pipeline: str | None = None
    rul_bundle: str | None = None
    shap_ranking: str | None = None
    detail: str | None = None


class SeriesResponse(BaseModel):
    turbine_id: str
    anomaly_threshold: float
    rows: list[dict[str, float | str | int | None]]


class TurbinesResponse(BaseModel):
    turbines: list[str]


def enforce_quantile_order(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    ordered = sorted([q10, q50, q90])
    return ordered[0], ordered[1], ordered[2]


def daily_feature_dict(body: TwinRulRequest) -> dict[str, float]:
    data = body.model_dump()
    return {col: float(data[col]) for col in DAILY_FEATURE_COLUMNS}


def scada_column_names() -> tuple[str, ...]:
    return SCADA_COLUMNS
