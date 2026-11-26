#!/usr/bin/env python3
"""Sanity-check CARE score on Wind Farm A with trivial baselines + paper IF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_digital_twin.config import CARE_RESULTS_DIR, DATA_RAW_CARE_WFA  # noqa: E402
from wind_digital_twin.data.load_care import iter_care_events, load_care_dataset  # noqa: E402
from wind_digital_twin.eval.care_score import CAREScore  # noqa: E402

META = frozenset({"id", "time_stamp", "asset_id", "train_test", "status_type_id"})


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in META]
    return df[cols].select_dtypes(include=[np.number]).fillna(0.0)


def _evaluate_strategy(
    farm_dir: Path,
    name: str,
    predict_fn,
    *,
    ignore_normal_index: bool = True,
) -> dict:
    scorer = CAREScore()
    for ds in iter_care_events(farm_dir):
        pred = ds.prediction.set_index("time_stamp") if "time_stamp" in ds.prediction.columns else ds.prediction
        idx = pd.to_datetime(pred.index, utc=True)
        predictions = predict_fn(ds, pred, idx)
        predictions = pd.Series(predictions, index=idx).astype(bool)
        scorer.evaluate_event(
            event_start=ds.event.event_start,
            event_end=ds.event.event_end,
            event_label=ds.event.label,
            predicted_anomalies=predictions,
            normal_index=None,
            ignore_normal_index=ignore_normal_index,
            event_id=ds.event.event_id,
        )
    return {
        "name": name,
        "care": scorer.get_final_score(),
        "coverage": scorer.calculate_avg_coverage(),
        "accuracy": scorer.calculate_avg_accuracy(),
        "reliability": scorer.calculate_reliability(),
        "earliness": scorer.calculate_avg_earliness(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument("--out", type=Path, default=CARE_RESULTS_DIR / "wfa_baselines.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.farm_dir / "event_info.csv").exists() and not (args.farm_dir / "datasets" / "event_info.csv").exists():
        print(f"CARE data missing in {args.farm_dir}; run scripts/download_care.py")
        return 1

    rng = np.random.default_rng(args.seed)
    results: list[dict] = []

    results.append(
        _evaluate_strategy(
            args.farm_dir,
            "all_normal",
            lambda _ds, _pred, idx: np.zeros(len(idx), dtype=bool),
        )
    )
    results.append(
        _evaluate_strategy(
            args.farm_dir,
            "all_anomaly",
            lambda _ds, _pred, idx: np.ones(len(idx), dtype=bool),
        )
    )

    def random_pred(_ds, _pred, idx):
        return rng.random(len(idx)) < 0.5

    results.append(_evaluate_strategy(args.farm_dir, "random", random_pred))

    def if_pred(ds, pred, idx):
        train = ds.train.loc[ds.train_normal_mask]
        x_train = _feature_matrix(train)
        x_pred = _feature_matrix(pred)
        if x_train.empty or x_pred.empty:
            return np.zeros(len(idx), dtype=bool)
        pca = PCA(n_components=0.99, svd_solver="full")
        x_train_p = pca.fit_transform(x_train)
        x_pred_p = pca.transform(x_pred)
        clf = IsolationForest(n_estimators=100, contamination=0.09, random_state=args.seed)
        clf.fit(x_train_p)
        scores = clf.predict(x_pred_p)
        return scores == -1

    results.append(_evaluate_strategy(args.farm_dir, "isolation_forest_paper", if_pred))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"farm_dir": str(args.farm_dir), "strategies": results}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())