"""Load CARE-to-Compare Wind Farm A event CSVs and align with twin column names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wind_digital_twin.config import (
    CARE_NORMAL_STATUS_IDS,
    CARE_WFA_RATED_KW,
    DATA_RAW_CARE_WFA,
)
from wind_digital_twin.data.care_channels import (
    load_frozen_channel_map,
    role_rename_columns,
)

META_COLUMNS = frozenset({"id", "time_stamp", "asset_id", "train_test", "status_type_id"})


@dataclass(frozen=True)
class CareEvent:
    event_id: int
    asset_id: str
    label: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    description: str


@dataclass
class CareDataset:
    event: CareEvent
    frame: pd.DataFrame
    train: pd.DataFrame
    prediction: pd.DataFrame
    train_normal_mask: pd.Series

    @property
    def prediction_index(self) -> pd.DatetimeIndex:
        ts = self.prediction.set_index("time_stamp") if "time_stamp" in self.prediction.columns else self.prediction
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index, utc=True)
        return ts.index


def _first_line_sep(path: Path) -> str:
    line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return ";" if ";" in line and line.count(";") > line.count(",") else ","


def load_event_info(farm_dir: Path | str) -> list[CareEvent]:
    farm_dir = Path(farm_dir)
    path = farm_dir / "event_info.csv"
    if not path.exists():
        path = farm_dir / "datasets" / "event_info.csv"
    sep = ";" if path.suffix == ".csv" and _first_line_sep(path) == ";" else ","
    df = pd.read_csv(path, sep=sep)
    col_start = "event_start" if "event_start" in df.columns else "start"
    col_end = "event_end" if "event_end" in df.columns else "end"
    label_col = "event_label" if "event_label" in df.columns else "label"
    events: list[CareEvent] = []
    for _, row in df.iterrows():
        eid = int(row.get("event_id", row.get("id", len(events))))
        asset = row.get("asset_id", row.get("asset", ""))
        events.append(
            CareEvent(
                event_id=eid,
                asset_id=str(asset),
                label=str(row[label_col]).strip().lower(),
                event_start=pd.to_datetime(row[col_start], utc=True),
                event_end=pd.to_datetime(row[col_end], utc=True),
                description=str(row.get("event_description", row.get("description", ""))),
            )
        )
    return events


def _datasets_dir(farm_dir: Path) -> Path:
    nested = farm_dir / "datasets"
    return nested if nested.is_dir() else farm_dir


def load_care_dataset(farm_dir: Path | str, event_id: int) -> CareDataset:
    farm_dir = Path(farm_dir)
    events = {e.event_id: e for e in load_event_info(farm_dir)}
    if event_id not in events:
        raise FileNotFoundError(f"event_id {event_id} not in event_info")
    event = events[event_id]
    csv_path = _datasets_dir(farm_dir) / f"{event_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    sep = ";" if _first_line_sep(csv_path) == ";" else ","
    frame = pd.read_csv(csv_path, sep=sep, low_memory=False)
    if "time_stamp" in frame.columns:
        frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], utc=True)

    train_mask = frame["train_test"].astype(str).str.lower() == "train"
    # CARE uses the literal value "prediction" for the hold-out window
    train = frame.loc[train_mask].copy()
    prediction = frame.loc[~train_mask].copy()

    train_normal = train["status_type_id"].isin(CARE_NORMAL_STATUS_IDS)
    train_normal_mask = pd.Series(train_normal.to_numpy(), index=train.index, name="train_normal")

    return CareDataset(
        event=event,
        frame=frame,
        train=train,
        prediction=prediction,
        train_normal_mask=train_normal_mask,
    )


def iter_care_events(farm_dir: Path | str):
    farm_dir = Path(farm_dir)
    for event in load_event_info(farm_dir):
        yield load_care_dataset(farm_dir, event.event_id)


def to_twin_frame(
    df: pd.DataFrame,
    channel_map: pd.DataFrame | None = None,
    *,
    rescale_power: bool = True,
    rated_kw: float = CARE_WFA_RATED_KW,
) -> pd.DataFrame:
    """Rename mapped avg columns to EDP names; rescale per-unit power to kW."""
    if channel_map is None:
        channel_map = load_frozen_channel_map()
    rename = role_rename_columns(channel_map)
    out = df.copy()
    cols_present = {k: v for k, v in rename.items() if k in out.columns}
    out = out.rename(columns=cols_present)

    if rescale_power and "Grd_Prod_Pwr_Avg" in out.columns:
        out["Grd_Prod_Pwr_Avg"] = out["Grd_Prod_Pwr_Avg"] * rated_kw

    if "time_stamp" in out.columns:
        out = out.set_index("time_stamp")
        out.index = pd.to_datetime(out.index, utc=True)
    return out


def default_wfa_dir() -> Path:
    return DATA_RAW_CARE_WFA