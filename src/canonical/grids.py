"""Shape-preserving adapters for the team's Australian radar/satellite exports."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.canonical.raw_index import read_index


def utc_hours(values, label):
    """Reject ambiguous, duplicate or off-hour timestamps; preserve UTC instants."""
    stamps = [pd.Timestamp(value) for value in values]
    if not stamps or any(pd.isna(t) or t.tzinfo is None for t in stamps):
        raise ValueError(f"{label}: nonempty timezone-aware timestamps required")
    result = pd.DatetimeIndex([t.tz_convert("UTC") for t in stamps])
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps must be unique and sorted")
    if not result.equals(result.floor("h")):
        raise ValueError(f"{label}: timestamps must fall on hourly boundaries")
    return result


@dataclass
class GridData:
    """Values are [time,node,feature]; coordinates retain the source array order."""

    layer: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    coordinates: pd.DataFrame

    def to_heterodata(self, target_timestamps):
        """Return a node layer for 04; topology remains the graph builder's job.

        The caller supplies its approved graph timeline. Missing radar frames fail
        rather than silently becoming observed zero rainfall. Satellite's source
        availability remains a feature, with invalid payloads masked only in x.
        """
        import torch
        from torch_geometric.data import HeteroData

        target = utc_hours(target_timestamps, "graph timeline")
        positions = self.timestamps.get_indexer(target)
        if (positions < 0).any():
            missing = target[positions < 0]
            raise ValueError(
                f"{self.layer}: {len(missing)} requested hours are absent; "
                f"first={missing[0].isoformat()}. Agree the common experiment "
                "timeline or a missing-modality interface with Component 04."
            )
        values = self.values[positions].copy()
        observed = np.isfinite(values[:, :, 0])
        if self.layer == "satellite":
            observed = values[:, :, 1].astype(bool)
            values[:, :, 0][~observed] = 0.0
        elif not observed.all():
            raise ValueError(
                "radar: missing pixel values cannot enter the one-feature graph "
                "contract; Component 04 needs an agreed validity-mask interface"
            )
        graph = HeteroData()
        graph[self.layer].x = torch.from_numpy(
            np.ascontiguousarray(values.transpose(1, 0, 2), dtype=np.float32)
        )
        if not torch.isfinite(graph[self.layer].x).all():
            raise ValueError(f"{self.layer}: values overflow float32 model features")
        graph[self.layer].observed = torch.from_numpy(observed.T.copy())
        graph[self.layer].node_id = torch.arange(len(self.coordinates))
        graph.timestamps = [t.isoformat() for t in target]
        return graph, self.coordinates[["latitude", "longitude"]].copy()

    def write_csv(self, directory):
        """Stream hours to the existing CSV contract without expanding a year in RAM."""
        directory = Path(directory)
        hourly = directory / f"{self.layer}_hourly.csv"
        with hourly.open("w", encoding="utf-8", newline="") as stream:
            for position, timestamp in enumerate(self.timestamps):
                data = {
                    "node_id": np.arange(len(self.coordinates)),
                    "timestamp": timestamp.isoformat(),
                    "rain_rate": self.values[position, :, 0],
                }
                if self.layer == "satellite":
                    data["valid"] = self.values[position, :, 1].astype(int)
                pd.DataFrame(data).to_csv(
                    stream, index=False, header=position == 0, lineterminator="\n"
                )
        self.coordinates.to_csv(
            directory / f"{self.layer}_nodes.csv", index=False, lineterminator="\n"
        )


def load_grid(data_file, coordinates_file, layer):
    """Read only trusted team-produced pickle files; pickle is executable input."""
    if layer not in {"radar", "satellite"}:
        raise ValueError("layer must be radar or satellite")
    data_file, coordinates_file = Path(data_file), Path(coordinates_file)
    for path in (data_file, coordinates_file):
        if not path.is_file():
            raise FileNotFoundError(f"BLOCKED: missing {layer} input: {path}")
    frame = pd.read_pickle(data_file)
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != {
        "timestamp",
        "data",
    }:
        raise ValueError(f"{layer}: expected DataFrame columns timestamp, data")
    timestamps = utc_hours(frame.timestamp, layer)
    arrays = [np.asarray(value) for value in frame.data]
    shape = arrays[0].shape
    if not shape or shape[0] == 0 or any(a.shape != shape for a in arrays):
        raise ValueError(f"{layer}: arrays must have a fixed, nonempty node shape")
    if (layer == "radar" and len(shape) != 1) or (
        layer == "satellite" and (len(shape) != 2 or shape[1] != 2)
    ):
        raise ValueError(f"{layer}: expected [nodes] radar or [nodes,2] satellite")
    if any(a.dtype.kind not in "fiu" for a in arrays):
        raise ValueError(f"{layer}: rain values must be numeric")
    values = np.stack(arrays)
    if layer == "radar":
        values = values[:, :, None]
    rain = values[:, :, 0]
    if np.isinf(rain).any():
        raise ValueError(f"{layer}: infinite rain payload")
    if layer == "satellite":
        valid = values[:, :, 1]
        if not np.isin(valid, [0, 1]).all():
            raise ValueError("satellite: availability flag must be exactly 0 or 1")
        if not np.isfinite(rain[valid == 1]).all() or (rain[valid == 1] < 0).any():
            raise ValueError("satellite: observed rain must be finite and nonnegative")
    elif (rain[np.isfinite(rain)] < 0).any():
        raise ValueError("radar: observed rain must be nonnegative")
    coords = pd.read_csv(coordinates_file)
    if (
        set(coords.columns)
        not in ({"latitude", "longitude"}, {"node_id", "latitude", "longitude"})
        or len(coords) != shape[0]
    ):
        raise ValueError(f"{layer}: coordinates must match the source node count")
    if "node_id" in coords and not np.array_equal(coords.node_id, np.arange(shape[0])):
        raise ValueError(f"{layer}: node_id must be 0..N-1 in source array order")
    coords = coords.copy()
    coords["node_id"] = np.arange(shape[0])
    for key, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
        coords[key] = pd.to_numeric(coords[key], errors="raise")
        if not coords[key].between(low, high).all():
            raise ValueError(f"{layer}: invalid {key}")
    return GridData(
        layer, timestamps, values, coords[["node_id", "latitude", "longitude"]]
    )


def indexed_path(raw_dir, index, relative_path):
    path = (raw_dir / relative_path).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError("input path must stay within raw_dir")
    rows = index[index.path == relative_path]
    if len(rows) != 1 or rows.iloc[0].status != "ok":
        raise ValueError(f"BLOCKED: {relative_path} needs exactly one ok raw index row")
    return path


def build(config_path, check_only=False):
    """Validate all enabled layers before writing; never modify gauge outputs/raw."""
    config = yaml.safe_load(Path(config_path).read_text())
    raw_dir, output_dir = Path(config["raw_dir"]), Path(config["output_dir"])
    if output_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("output_dir must not be inside immutable raw_dir")
    index = read_index(raw_dir, dtype=str, keep_default_na=False)
    if not {"path", "status"}.issubset(index.columns):
        raise ValueError("raw index needs path and status columns")
    layers = config["layers"]
    if not layers or set(layers) - {"radar", "satellite"}:
        raise ValueError("configure at least one of radar or satellite")
    grids = []
    report = [
        "# Radar and satellite adapter",
        "",
        "Values are reshaped without unit conversion, resampling or gap filling.",
        "The contract's rain_rate column retains the source values; its name is not evidence of units.",
        "",
    ]
    for layer, settings in layers.items():
        data_file = indexed_path(raw_dir, index, settings["data_file"])
        coordinates_file = indexed_path(raw_dir, index, settings["coordinates_file"])
        grid = load_grid(
            data_file,
            coordinates_file,
            layer,
        )
        grids.append(grid)
        expected = pd.date_range(grid.timestamps[0], grid.timestamps[-1], freq="h")
        missing = expected.difference(grid.timestamps)
        report.extend(
            [
                f"## {layer}",
                "",
                f"Nodes: {len(grid.coordinates)}; source hours: {len(grid.timestamps)}; rows: {len(grid.coordinates) * len(grid.timestamps)}.",
                f"Range: {grid.timestamps[0].isoformat()} to {grid.timestamps[-1].isoformat()}.",
                f"Absent hours within that range: {len(missing)}. Boundaries outside the source range are not inferred.",
                f"Missing rain payloads: {np.isnan(grid.values[:, :, 0]).sum()}.",
            ]
        )
        if layer == "satellite":
            report.append(
                f"Source unavailable node-hours: {(grid.values[:, :, 1] == 0).sum()}."
            )
        for path in (data_file, coordinates_file):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            report.append(f"Input: {path}; SHA256: {digest}.")
        for field in (
            "product",
            "units",
            "timestamp_convention",
            "gauge_correction",
            "evidence",
        ):
            report.append(f"{field}: {settings.get(field, 'UNKNOWN')}.")
        report.extend(
            [
                "",
                "Scientific readiness requires reviewed units/time convention, coordinate ordering and held-out-gauge independence.",
                "",
            ]
        )
        report.extend(f"- Absent hour: {stamp.isoformat()}" for stamp in missing)
        report.append("")
    if check_only:
        print("Inputs valid: " + ", ".join(grid.layer for grid in grids))
        return
    targets = [output_dir / "grid_report.md"] + [
        output_dir / f"{grid.layer}_{suffix}.csv"
        for grid in grids
        for suffix in ("hourly", "nodes")
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "grid outputs already exist; choose a fresh output_dir for a reproducible rerun"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for grid in grids:
        grid.write_csv(output_dir)
    (output_dir / "grid_report.md").write_text("\n".join(report), encoding="utf-8")
    print(output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    try:
        build(args.config, args.check_inputs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
