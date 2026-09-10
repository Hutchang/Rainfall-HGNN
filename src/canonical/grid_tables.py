"""Read Component 02 standard grid tables for the Component 04 builder."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical.grids import GridData, utc_hours


def load_standard_grid(directory, layer):
    if layer not in {"radar", "satellite"}:
        raise ValueError("layer must be radar or satellite")
    directory = Path(directory)
    coords = pd.read_csv(directory / f"{layer}_nodes.csv")
    table = pd.read_csv(directory / f"{layer}_hourly.csv")
    columns = {"node_id", "timestamp", "rain_rate"}
    if layer == "satellite":
        columns.add("valid")
    if set(table.columns) != columns or set(coords.columns) != {
        "node_id",
        "latitude",
        "longitude",
    }:
        raise ValueError(f"{layer}: unexpected standard grid columns")
    count = len(coords)
    if count == 0 or not np.array_equal(coords.node_id, np.arange(count)):
        raise ValueError(f"{layer}: coordinate node_id must be 0..N-1 in tensor order")
    for name, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
        if not coords[name].between(low, high).all():
            raise ValueError(f"{layer}: invalid coordinate {name}")
    hours = utc_hours(table.timestamp.drop_duplicates().tolist(), layer)
    table["timestamp"] = pd.to_datetime(table.timestamp, utc=True)
    index = pd.MultiIndex.from_product(
        [hours, range(count)], names=["timestamp", "node_id"]
    )
    indexed = table.set_index(["timestamp", "node_id"]).sort_index()
    if not indexed.index.equals(index):
        raise ValueError(
            f"{layer}: each present hour needs exactly one row per coordinate node"
        )
    features = ["rain_rate", "valid"] if layer == "satellite" else ["rain_rate"]
    values = (
        indexed[features]
        .to_numpy(dtype=float)
        .reshape(len(hours), count, len(features))
    )
    observed = np.ones((len(hours), count), dtype=bool)
    if layer == "satellite":
        if not np.isin(values[:, :, 1], [0, 1]).all():
            raise ValueError("satellite: availability must be exactly 0 or 1")
        observed = values[:, :, 1].astype(bool)
    rain = values[:, :, 0][observed]
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError(f"{layer}: observed rainfall must be finite and nonnegative")
    return GridData(layer, hours, values, coords)
