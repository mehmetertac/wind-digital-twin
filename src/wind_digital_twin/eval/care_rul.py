"""Real CARE Wind Farm A windows as leave-one-failure-out RUL cases.

Local EDP SCADA in this checkout is the synthetic stand-in. Wind Farm A is
the real farm, with anonymized asset ids. Failure time is the CARE
``event_start`` (annotated onset). Overlapping labels on the same asset are
one episode: the later onset is not an independent failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_digital_twin.anomaly.hybrid_residual_detector import fit_hybrid_residual_detector
from wind_digital_twin.config import CARE_NORMAL_STATUS_IDS
from wind_digital_twin.data.clean import clean_turbine_df
from wind_digital_twin.data.load_care import CareDataset, CareEvent, to_twin_frame
from wind_digital_twin.eval.care_hybrid_eval import is_drivetrain_anomaly, is_gearbox_anomaly, raw_to_operating
from wind_digital_twin.eval.identity_calibration import MAX_FIT_ROWS
from wind_digital_twin.eval.twin_pass import build_twin_layer_frame
from wind_digital_twin.residual.fusion_features import build_fusion_feature_frame

OVERLAP_GAP = pd.Timedelta(days=14)


def select_failure_events(events: list[CareEvent], mode: str) -> list[CareEvent]:
    """Anomaly events for ``gearbox`` or ``drivetrain`` (gearbox + generator bearing)."""
    if mode not in {"gearbox", "drivetrain"}:
        raise ValueError("mode must be 'gearbox' or 'drivetrain'")
    chosen: list[CareEvent] = []
    for event in events:
        if event.label != "anomaly":
            continue
        keep = is_gearbox_anomaly(event.description) if mode == "gearbox" else is_drivetrain_anomaly(event.description)
        if keep:
            chosen.append(event)
    return chosen


def independent_onsets(
    events: list[CareEvent],
    gap: pd.Timedelta = OVERLAP_GAP,
) -> tuple[list[CareEvent], list[dict]]:
    """Keep the earlier onset when two labels on one asset overlap or sit within ``gap``."""
    by_asset: dict[str, list[CareEvent]] = {}
    for event in sorted(events, key=lambda item: (str(item.asset_id), item.event_start)):
        by_asset.setdefault(str(event.asset_id), []).append(event)

    kept: list[CareEvent] = []
    dropped: list[dict] = []
    for asset, group in by_asset.items():
        previous: CareEvent | None = None
        for event in group:
            if previous is not None and event.event_start <= previous.event_end + gap:
                dropped.append(
                    {
                        "event_id": event.event_id,
                        "asset_id": asset,
                        "merged_into": previous.event_id,
                        "description": event.description,
                        "reason": (
                            f"overlaps event {previous.event_id} on asset {asset}; "
                            "not an independent failure"
                        ),
                    }
                )
                if event.event_end > previous.event_end:
                    previous = CareEvent(
                        event_id=previous.event_id,
                        asset_id=previous.asset_id,
                        label=previous.label,
                        event_start=previous.event_start,
                        event_end=event.event_end,
                        description=previous.description,
                    )
                    kept[-1] = previous
                continue
            kept.append(event)
            previous = event
    return kept, dropped


def failure_log_frame(events: list[CareEvent]) -> pd.DataFrame:
    """Failure log without a component column, so every row is kept downstream.

    ``remarks`` starts with the CARE event id so lead-time text can name it.
    Timestamp is the annotated onset.
    """
    rows = [
        {
            "turbine_id": str(event.asset_id),
            "timestamp": pd.Timestamp(event.event_start),
            "remarks": f"{event.event_id}|{event.description}",
        }
        for event in events
    ]
    return pd.DataFrame(rows)


def care_id_from_remarks(remarks: str) -> str:
    return str(remarks).split("|", 1)[0]


def _failure_ids(events: list[CareEvent]) -> set[int]:
    return {event.event_id for event in events}


def scored_frame_for_asset(
    datasets: list[CareDataset],
    channel_map: pd.DataFrame,
    failure_events: list[CareEvent],
    *,
    max_fit_rows: int = MAX_FIT_ROWS,
) -> pd.DataFrame:
    """Fit the thermal twin on status-normal rows outside failure windows and score the asset."""
    if not datasets:
        raise ValueError("no datasets for asset")
    failure_ids = _failure_ids(failure_events)
    fit_ns: set[int] = set()
    withhold_ns: set[int] = set()
    parts: list[pd.DataFrame] = []

    for ds in datasets:
        frame = ds.frame.copy()
        stamps = pd.to_datetime(frame["time_stamp"], utc=True)
        frame["time_stamp"] = stamps
        parts.append(frame)
        ns = stamps.astype("int64").to_numpy()
        train = frame["train_test"].astype(str).str.lower().eq("train").to_numpy()
        normal = frame["status_type_id"].isin(CARE_NORMAL_STATUS_IDS).to_numpy()
        if ds.event.event_id in failure_ids:
            withhold_ns.update(int(value) for value in ns[~train])
            inside = ((stamps >= ds.event.event_start) & (stamps <= ds.event.event_end)).to_numpy()
            withhold_ns.update(int(value) for value in ns[inside])
        fit_ns.update(int(value) for value in ns[train & normal])

    fit_ns -= withhold_ns
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates("time_stamp").sort_values("time_stamp")
    operating = raw_to_operating(raw, channel_map)
    if operating.empty:
        twin = to_twin_frame(raw, channel_map=channel_map, rescale_power=True)
        operating = clean_turbine_df(twin, min_power_kw=0.0)

    wanted = np.fromiter(fit_ns, dtype=np.int64, count=len(fit_ns)) if fit_ns else np.array([], dtype=np.int64)
    fit_df = operating.loc[np.isin(operating.index.asi8, wanted)]
    if len(fit_df) > max_fit_rows:
        fit_df = fit_df.iloc[-max_fit_rows:]
    if len(fit_df) < 100:
        raise ValueError(f"asset {datasets[0].event.asset_id} has {len(fit_df)} healthy operating rows")

    pipeline = fit_hybrid_residual_detector(fit_df)
    layer = build_twin_layer_frame(operating, pipeline)
    fusion = build_fusion_feature_frame(operating)
    theta_cols = [col for col in ("theta_oil", "theta_bear", "delta_bear_oil") if col in fusion.columns]
    scored = layer.join(fusion[theta_cols], how="left")
    scored = scored.rename(
        columns={
            "oil_hybrid_residual": "oil_residual",
            "bear_hybrid_residual": "bear_residual",
        }
    )
    keep = ["oil_residual", "bear_residual", "anomaly_score", *theta_cols]
    out = scored.loc[:, keep].dropna(subset=["anomaly_score"]).copy()
    out["turbine_id"] = str(datasets[0].event.asset_id)
    out["timestamp"] = out.index
    return out.reset_index(drop=True)
