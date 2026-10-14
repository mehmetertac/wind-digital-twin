"""Load serving artifacts from ``service/models/`` at startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from twin_service.constants import RUL_MAX_DAYS

SERVICE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = SERVICE_ROOT / "models"
MODEL_VERSION = "synthetic-smoke-v1"


@dataclass
class ConformalOffsets:
    delta_80: float
    delta_90: float


@dataclass
class LoadedArtifacts:
    hybrid_pipeline: Any
    anomaly_threshold: float
    rul_bundle: dict[str, Any]
    conformal: ConformalOffsets
    shap_ranking: pd.DataFrame
    timeline: pd.DataFrame
    model_version: str = MODEL_VERSION


def models_dir() -> Path:
    override = __import__("os").environ.get("TWIN_MODELS_DIR")
    if override:
        return Path(override)
    return MODELS_DIR


def load_artifacts(models_path: Path | None = None) -> LoadedArtifacts:
    root = models_path or models_dir()
    pipeline_path = root / "hybrid_pipeline.joblib"
    threshold_path = root / "anomaly_threshold.json"
    bundle_path = root / "rul_bundle.joblib"
    offsets_path = root / "conformal_offsets.json"
    ranking_path = root / "shap_sensor_ranking.csv"
    timeline_path = root / "timeline_demo.parquet"

    for path in (
        pipeline_path,
        threshold_path,
        bundle_path,
        offsets_path,
        ranking_path,
        timeline_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing serving artifact: {path}")

    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    offsets_payload = json.loads(offsets_path.read_text(encoding="utf-8"))

    return LoadedArtifacts(
        hybrid_pipeline=joblib.load(pipeline_path),
        anomaly_threshold=float(threshold_payload["threshold"]),
        rul_bundle=joblib.load(bundle_path),
        conformal=ConformalOffsets(
            delta_80=float(offsets_payload["delta_80"]),
            delta_90=float(offsets_payload["delta_90"]),
        ),
        shap_ranking=pd.read_csv(ranking_path),
        timeline=pd.read_parquet(timeline_path),
        model_version=str(threshold_payload.get("model_version", MODEL_VERSION)),
    )


def bundle_rul_max(bundle: dict[str, Any]) -> int:
    return int(bundle.get("rul_max", RUL_MAX_DAYS))
