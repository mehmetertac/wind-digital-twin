"""FastAPI serving layer — SCADA scoring and daily RUL."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from starlette.requests import Request

from twin_service.artifacts import LoadedArtifacts, load_artifacts, models_dir
from twin_service.inference import (
    list_turbines,
    predict_rul,
    score_window,
    series_for_turbine,
    top_sensor_attributions,
)
from twin_service.logging_config import log_request, new_request_id
from twin_service.schema import (
    HealthResponse,
    SeriesResponse,
    TwinRulRequest,
    TwinRulResponse,
    TwinScoreRequest,
    TwinScoreResponse,
    TurbinesResponse,
)

logger = logging.getLogger(__name__)

LoadArtifactsFn = Callable[[], LoadedArtifacts]


def _default_load() -> LoadedArtifacts:
    return load_artifacts()


def create_app(*, load_fn: LoadArtifactsFn | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    resolve_load = load_fn or _default_load
    artifacts_holder: dict[str, LoadedArtifacts | None] = {"value": None}

    def get_artifacts() -> LoadedArtifacts:
        if artifacts_holder["value"] is None:
            artifacts_holder["value"] = resolve_load()
        return artifacts_holder["value"]

    class RequestContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = request.headers.get("X-Request-ID") or new_request_id()
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            try:
                version = get_artifacts().model_version
            except Exception:
                version = None
            log_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                model_version=version,
            )
            return response

    app = FastAPI(
        title="wind-digital-twin-service",
        description="Hybrid thermal twin scoring and probabilistic gearbox RUL.",
        version="0.1.0",
    )
    app.add_middleware(RequestContextMiddleware)

    @app.on_event("startup")
    def _warm_models() -> None:
        try:
            get_artifacts()
        except Exception as exc:
            logger.warning("Model warm load failed (tests may inject load_fn): %s", exc)

    def _error_detail(*, code: str, message: str, request_id: str) -> dict[str, str]:
        return {"code": code, "message": message, "request_id": request_id}

    @app.get("/health", response_model=HealthResponse)
    def health(http_request: Request) -> HealthResponse | JSONResponse:
        request_id = http_request.state.request_id
        try:
            art = get_artifacts()
            root = models_dir()
            return HealthResponse(
                status="ok",
                model_version=art.model_version,
                hybrid_pipeline=str(root / "hybrid_pipeline.joblib"),
                rul_bundle=str(root / "rul_bundle.joblib"),
                shap_ranking=str(root / "shap_sensor_ranking.csv"),
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=503,
                content=_error_detail(
                    code="models_missing",
                    message=str(exc),
                    request_id=request_id,
                ),
            )
        except Exception as exc:
            return HealthResponse(status="unhealthy", model_version="unknown", detail=str(exc))

    @app.post("/twin/score", response_model=TwinScoreResponse)
    def twin_score(body: TwinScoreRequest, http_request: Request) -> TwinScoreResponse:
        request_id = http_request.state.request_id
        try:
            return score_window(get_artifacts(), body)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=_error_detail(code="invalid_window", message=str(exc), request_id=request_id),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=_error_detail(code="score_failed", message=str(exc), request_id=request_id),
            ) from exc

    @app.post("/twin/rul", response_model=TwinRulResponse)
    def twin_rul(body: TwinRulRequest, http_request: Request) -> TwinRulResponse:
        request_id = http_request.state.request_id
        try:
            return predict_rul(get_artifacts(), body)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=_error_detail(code="rul_failed", message=str(exc), request_id=request_id),
            ) from exc

    @app.get("/twin/turbines", response_model=TurbinesResponse)
    def twin_turbines() -> TurbinesResponse:
        return TurbinesResponse(turbines=list_turbines(get_artifacts()))

    @app.get("/twin/attributions")
    def twin_attributions() -> dict[str, list[dict[str, object]]]:
        attrs = top_sensor_attributions(get_artifacts().shap_ranking)
        return {"attributions": [item.model_dump() for item in attrs]}

    @app.get("/twin/series", response_model=SeriesResponse)
    def twin_series(
        turbine: str = Query(..., min_length=1, description="Turbine id, e.g. T06"),
    ) -> SeriesResponse:
        art = get_artifacts()
        frame = series_for_turbine(art, turbine.strip())
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No timeline for turbine={turbine!r}")
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            record: dict[str, Any] = {}
            for col in frame.columns:
                val = row[col]
                if isinstance(val, pd.Timestamp):
                    record[col] = val.isoformat().replace("+00:00", "Z")
                elif pd.isna(val):
                    record[col] = None
                else:
                    record[col] = val
            rows.append(record)
        return SeriesResponse(
            turbine_id=turbine.strip(),
            anomaly_threshold=art.anomaly_threshold,
            rows=rows,
        )

    return app


app = create_app()
