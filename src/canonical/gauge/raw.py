"""Read Component 01's raw gauge output. The only module in 02 that opens raw files.

Reads index.csv first, then only the files it marks ok, so a failed download can
never be mistaken for a dry month.
"""

import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd

TIP_COLUMNS = ["station_id", "timestamp", "value", "quality"]


def read_index(raw_dir: Path) -> pd.DataFrame:
    """Component 01's manifest: one row per raw file, with its download status."""
    return pd.read_csv(raw_dir / "index.csv", dtype={"station_id": str})


def period_from_index(index: pd.DataFrame) -> tuple[date, date]:
    """The tip period Component 01 actually covered, taken from the index rather than
    from a config, so the two cannot silently disagree."""
    t = index[index["source"] == "gauge_tips"]
    if t.empty:
        raise ValueError(
            "index.csv has no gauge_tips rows: Component 01 has not fetched any tip "
            "series for this period, so there is nothing for Component 02 to build"
        )
    return pd.Timestamp(t["start"].min()).date(), pd.Timestamp(t["end"].max()).date()


def read_stations(raw_dir: Path) -> pd.DataFrame:
    with (raw_dir / "gauge" / "stations.json").open() as fh:
        header, *rows = json.load(fh)
    df = pd.DataFrame(rows, columns=header)
    df["latitude"] = pd.to_numeric(df["station_latitude"])
    df["longitude"] = pd.to_numeric(df["station_longitude"])
    return df[["station_id", "station_no", "station_name", "latitude", "longitude"]]


def _records(raw_dir: Path, index: pd.DataFrame, source: str) -> pd.DataFrame:
    frames = []
    ok = index[(index["source"] == source) & (index["status"] == "ok")]
    for row in ok.itertuples():
        with gzip.open(raw_dir / row.path) as fh:
            data = json.load(fh)[0]["data"]
        if data:
            f = pd.DataFrame(data, columns=["timestamp", "value", "quality"])
            f.insert(0, "station_id", row.station_id)
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=TIP_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    # BoM stamps a fixed +10:00 offset; parse as timezone-aware and convert to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["quality"] = df["quality"].astype(int)
    return df


def read_tips(raw_dir: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Every tip record, UTC. One row per record, values as sent (0.0 markers, nulls kept)."""
    return _records(raw_dir, index, "gauge_tips")


def read_daily(raw_dir: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Every daily-total record, UTC. Timestamp is the END of the rain day (09:00 local)."""
    return _records(raw_dir, index, "gauge_daily")
