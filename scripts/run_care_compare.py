#!/usr/bin/env python3
"""WF-A CARE score: hybrid twin, own Isolation Forest, dense and LSTM autoencoders.

Models fit on status-normal training rows only. The alert threshold for an
asset is the conformal rank of calibration z-scores from the other assets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Torch's Windows DLL init fails if NumPy/MKL load first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    torch = None

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_digital_twin.config import CARE_RESULTS_DIR, DATA_RAW_CARE_WFA, RESIDUAL_EWMA_SPAN  # noqa: E402
from wind_digital_twin.data.care_channels import load_frozen_channel_map  # noqa: E402
from wind_digital_twin.data.load_care import iter_care_events  # noqa: E402
from wind_digital_twin.eval.care_nbm import (  # noqa: E402
    MODEL_NAMES,
    CachedEvent,
    attach_prediction_normal,
    bootstrap_care,
    bootstrap_care_difference,
    bootstrap_interval,
    care_from_alarm_union,
    care_from_cached,
    score_care_event,
)
from wind_digital_twin.eval.care_rise_score import (  # noqa: E402
    FUSED_MODEL_NAME,
    GEARBOX_TARGETS,
    RISE_MODEL_NAMES,
    SUBSYSTEM_TARGETS,
    apply_bonferroni_z,
    gearbox_fit_bin_bias_c,
    gearbox_rise_bias_c,
    score_rise_into_event,
)
from wind_digital_twin.eval.identity_calibration import (  # noqa: E402
    DEFAULT_ALERT_ALPHA,
    FARM_A_LOCKED_TEST_ASSETS,
)

BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 0

STORED_MODELS = MODEL_NAMES + RISE_MODEL_NAMES + (FUSED_MODEL_NAME,)
RISE_CHANNEL_COUNTS = {
    "hybrid_rise": len(GEARBOX_TARGETS),
    "hybrid_multi": len(SUBSYSTEM_TARGETS),
}


def _apply_rise_pvalues(events: list[CachedEvent], alpha: float, *, excluded: set[str]) -> None:
    """Bonferroni min-p. Empty ``excluded`` is leave-one-turbine-out."""
    per_event = not excluded
    for name, n_channels in RISE_CHANNEL_COUNTS.items():
        apply_bonferroni_z(
            events,
            name,
            alpha=alpha,
            n_channels=n_channels,
            excluded_assets=excluded,
            per_event_exclude=per_event,
        )
    apply_bonferroni_z(
        events,
        FUSED_MODEL_NAME,
        alpha=alpha,
        n_channels=len(SUBSYSTEM_TARGETS) + 1,
        source_model="hybrid_multi",
        extra_models=("dense_ae",),
        excluded_assets=excluded,
        per_event_exclude=per_event,
    )


def _with_interval(events: list[CachedEvent], name: str, **kwargs) -> dict:
    report = care_from_cached(events, name, **kwargs)
    boot = bootstrap_care(events, name, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, **kwargs)
    report["summary"]["care_ci90"] = boot["ci90"]
    report["summary"]["bootstrap_n"] = BOOTSTRAP_N
    report["summary"]["bootstrap_seed"] = BOOTSTRAP_SEED
    return report


def _union_with_interval(events: list[CachedEvent], models: tuple[str, ...], **kwargs) -> dict:
    report = care_from_alarm_union(events, models, include_frame=True, **kwargs)
    frame = report.pop("frame")
    report["summary"]["care_ci90"] = bootstrap_interval(frame, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    report["summary"]["bootstrap_n"] = BOOTSTRAP_N
    report["summary"]["bootstrap_seed"] = BOOTSTRAP_SEED
    return report


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _cache_stem(cache_dir: Path, event_id: int) -> Path:
    return cache_dir / f"event_{event_id}"


def save_event(cache_dir: Path, event: CachedEvent) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(cache_dir, event.event_id)
    meta = {
        "event_id": event.event_id,
        "asset_id": event.asset_id,
        "label": event.label,
        "description": event.description,
        "event_start": pd.Timestamp(event.event_start).isoformat(),
        "event_end": pd.Timestamp(event.event_end).isoformat(),
        "errors": event.errors,
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    arrays = {"pred_ns": np.asarray(event.pred_index.asi8)}
    if event.pred_normal is not None:
        arrays["pred_normal"] = np.asarray(event.pred_normal, dtype=bool)
    for name in STORED_MODELS:
        arrays[f"{name}_pred"] = np.asarray(event.z_pred.get(name, []), dtype=float)
        arrays[f"{name}_cal"] = np.asarray(event.z_cal.get(name, []), dtype=float)
        cal_index = event.cal_index.get(name)
        if cal_index is not None and len(cal_index) == len(event.z_cal.get(name, [])):
            arrays[f"{name}_cal_ns"] = np.asarray(pd.DatetimeIndex(cal_index).asi8)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)


def load_event(cache_dir: Path, event_id: int) -> CachedEvent:
    stem = _cache_stem(cache_dir, event_id)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    arrays = np.load(stem.with_suffix(".npz"))
    index = pd.to_datetime(arrays["pred_ns"], utc=True)
    event = CachedEvent(
        event_id=int(meta["event_id"]),
        asset_id=str(meta["asset_id"]),
        label=str(meta["label"]),
        description=str(meta["description"]),
        event_start=pd.Timestamp(meta["event_start"]),
        event_end=pd.Timestamp(meta["event_end"]),
        pred_index=pd.DatetimeIndex(index),
        errors=dict(meta.get("errors") or {}),
    )
    if "pred_normal" in arrays:
        event.pred_normal = np.asarray(arrays["pred_normal"], dtype=bool)
    for name in STORED_MODELS:
        pred_key = f"{name}_pred"
        cal_key = f"{name}_cal"
        cal_ns = f"{name}_cal_ns"
        if pred_key in arrays:
            event.z_pred[name] = np.asarray(arrays[pred_key], dtype=float)
        if cal_key in arrays:
            event.z_cal[name] = np.asarray(arrays[cal_key], dtype=float)
        if cal_ns in arrays:
            event.cal_index[name] = pd.DatetimeIndex(pd.to_datetime(arrays[cal_ns], utc=True))
    return event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument("--out", type=Path, default=CARE_RESULTS_DIR / "wfa_compare.json")
    parser.add_argument("--cache-dir", type=Path, default=CARE_RESULTS_DIR / "wfa_score_cache")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALERT_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    if not (args.farm_dir / "event_info.csv").exists() and not (
        args.farm_dir / "datasets" / "event_info.csv"
    ).exists():
        print(f"CARE data missing in {args.farm_dir}; run scripts/download_care.py")
        return 1

    channel_map = load_frozen_channel_map()
    skip_lstm = args.skip_lstm or torch is None
    if torch is None and not args.skip_lstm:
        print("torch unavailable; skipping LSTM autoencoder", flush=True)
    models = tuple(name for name in MODEL_NAMES if not (skip_lstm and name == "lstm_ae"))
    fresh_models = models + RISE_MODEL_NAMES
    cached: list[CachedEvent] = []
    event_71: dict = {}

    for ds in iter_care_events(args.farm_dir):
        stem = _cache_stem(args.cache_dir, ds.event.event_id)
        if args.reuse_cache and stem.with_suffix(".npz").exists():
            print(f"cache event {ds.event.event_id}", flush=True)
            event = load_event(args.cache_dir, ds.event.event_id)
        else:
            print(
                f"scoring event {ds.event.event_id} asset {ds.event.asset_id} ({ds.event.label})",
                flush=True,
            )
            event = score_care_event(ds, channel_map, seed=args.seed, models=fresh_models)
        if event.pred_normal is None:
            attach_prediction_normal(event, ds.prediction)
        print(f"  rise models {RISE_MODEL_NAMES}", flush=True)
        score_rise_into_event(event, ds, channel_map, RISE_MODEL_NAMES)
        if ds.event.event_id == 71:
            bias = gearbox_rise_bias_c(ds, channel_map)
            bins = gearbox_fit_bin_bias_c(ds, channel_map)
            print(f"  event 71 rise bias C: {bias}", flush=True)
            print(f"  event 71 fit-bin bias C: {bins}", flush=True)
            event_71 = {"prediction_bias_c": bias, "fit_bin_bias_c": bins}
        if event.errors:
            print(f"  errors: {event.errors}", flush=True)
        cached.append(event)

    _apply_rise_pvalues(cached, args.alpha, excluded=set())
    for event in cached:
        save_event(args.cache_dir, event)

    def _has_scores(name: str) -> bool:
        return any(
            np.isfinite(np.asarray(event.z_pred.get(name, []), dtype=float)).any() for event in cached
        )

    reports = []
    report_names = [name for name in STORED_MODELS if _has_scores(name)]
    fix_names = [
        name
        for name in ("hybrid", "hybrid_rise", "hybrid_multi", FUSED_MODEL_NAME, "dense_ae", "isolation_forest")
        if name in report_names
    ]
    for name in report_names:
        print(f"CARE {name} loto", flush=True)
        reports.append(_with_interval(cached, name, alpha=args.alpha, mode="loto"))
        reports.append(
            _with_interval(cached, name, alpha=args.alpha, mode="loto", ewma_span=RESIDUAL_EWMA_SPAN)
        )
    for name in fix_names:
        reports.append(
            _with_interval(cached, name, alpha=args.alpha, mode="loto", gearbox_only=True)
        )
    if _has_scores("hybrid_multi") and _has_scores("dense_ae"):
        reports.append(_union_with_interval(cached, ("hybrid_multi", "dense_ae"), alpha=args.alpha))
        reports.append(
            _union_with_interval(
                cached,
                ("hybrid_multi", "dense_ae"),
                alpha=args.alpha,
                gearbox_only=True,
            )
        )
        reports.append(
            _union_with_interval(
                cached,
                ("hybrid_multi", "dense_ae"),
                alpha=args.alpha,
                ewma_span=RESIDUAL_EWMA_SPAN,
            )
        )
    differences = []
    for right in ("hybrid", "hybrid_multi", FUSED_MODEL_NAME):
        if _has_scores("dense_ae") and _has_scores(right):
            differences.append(
                bootstrap_care_difference(
                    cached,
                    "dense_ae",
                    right,
                    n=BOOTSTRAP_N,
                    seed=BOOTSTRAP_SEED,
                    alpha=args.alpha,
                    mode="loto",
                )
            )
    _apply_rise_pvalues(cached, args.alpha, excluded=set(FARM_A_LOCKED_TEST_ASSETS))
    for name in fix_names:
        if name in RISE_CHANNEL_COUNTS or name == FUSED_MODEL_NAME or _has_scores(name):
            reports.append(
                _with_interval(
                    cached,
                    name,
                    alpha=args.alpha,
                    mode="locked",
                    locked_assets=FARM_A_LOCKED_TEST_ASSETS,
                )
            )

    payload = {
        "farm_dir": str(args.farm_dir),
        "alpha": args.alpha,
        "ewma_span": RESIDUAL_EWMA_SPAN,
        "power_spline_knots": 5,
        "threshold": (
            "Matrix models: identity-disjoint conformal rank on other assets. "
            "Rise and min-p fusion: per-channel conformal p-values on other assets, "
            "Bonferroni min-p, stored as -log10(p)."
        ),
        "normal_status_ids": [0, 2],
        "locked_test_assets": sorted(FARM_A_LOCKED_TEST_ASSETS),
        "paper_ae_care_95_events": 0.66,
        "event_71": event_71,
        "differences_loto": differences,
        "strategies": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    for report in reports:
        summary = report["summary"]
        print(
            f"{summary.get('model')} mode={summary.get('mode')} "
            f"gearbox_only={summary.get('gearbox_only')} "
            f"normal_index={summary.get('use_normal_index')} ewma={summary.get('ewma_span')} "
            f"CARE={summary.get('care')} ci90={summary.get('care_ci90')} cov={summary.get('coverage')} "
            f"acc={summary.get('accuracy')} rel={summary.get('reliability')} "
            f"earl={summary.get('earliness')}",
            flush=True,
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
