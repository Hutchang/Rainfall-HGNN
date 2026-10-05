"""Shape-preserving adapters for the team's Australian radar/satellite exports."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.canonical.gauge.build import load_stations
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


def single_declaration(values, name, root):
    """The one value every archive file read gives for `name`, or "UNKNOWN
    (...)" naming the gap when no file gives one (FR04). Files that disagree
    are mixed products or sites, so they stop the build rather than pick one.
    """
    given = {value for value in values if value is not None}
    if len(given) > 1:
        raise ValueError(f"files under {root} disagree on {name}: {sorted(given)}")
    return given.pop() if given else f"UNKNOWN (no file gives {name})"


@dataclass
class GridData:
    """Values are [time,node,feature]; coordinates retain the source array order.

    `provenance` (D13.4) carries extra facts an archive reader took from its
    own files -- coordinate system, native time step and, for radar, each
    written hour's `contributing_gauges` union -- for the report to state.
    `None` for a legacy pickle-mode grid, which has no file attributes to
    take these from.
    """

    layer: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    coordinates: pd.DataFrame
    provenance: dict | None = None

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


def _period_coverage_line(grid: GridData, start, end) -> str:
    """D13.4: days and hours present/missing against the *configured* period,
    not just the span of hours the run actually found -- a source that is
    entirely absent for a leading or trailing stretch doesn't hide inside
    "absent hours within the observed range" the way it would above. A
    single count, not an itemised list: the itemised list below stays scoped
    to the observed range so a real multi-year run doesn't flood the report.
    Also how ISS-04's "120 missing hours" gets answered: compare this
    archive's missing count against that hand-export number.
    """
    expected = pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
    )
    present = len(expected.intersection(grid.timestamps))
    return (
        f"Configured period {start} to {end}: {len(expected)} hours possible, "
        f"{present} present, {len(expected) - present} missing."
    )


def _archive_declarations(provenance: dict) -> list[str]:
    """FR04: what an archive reader took from the files themselves, and the
    rules it applied to them. grids.yaml's own declarations are not used."""
    lines = [
        f"product (from the files): {provenance['product']}.",
        f"units (from the files): {provenance['units']}; hourly value: "
        f"{provenance['hourly_rule']}.",
        "time convention: hour beginning, UTC; each hour is labelled by the start "
        "of its first file.",
        f"gauge correction (from the files): {provenance['gauge_correction']}.",
    ]
    if "site" in provenance:
        lines.append(f"site (from the files): {provenance['site']}.")
    if "crop" in provenance:
        lines.append(f"crop: {provenance['crop']}.")
    lines.append(f"subsample rule: {provenance['subsample_rule']}.")
    return lines


def _contributing_gauges_report(
    gauges_by_hour: dict, output_dir: Path
) -> list[str]:
    """D13.4: how many *study* stations appear in radar's `contributing_gauges`
    and in how many written hours -- the actual size of D09's leakage caveat,
    not just "yes, it's gauge-corrected". Matches on `station_no`, the only
    key the archive gives us; `station_no` is not a unique key project-wide
    (docs/ingestion/README.md), so this count is named as a `station_no`
    match, not a `station_id` one. Graceful, not a hard failure, when
    Component 02 hasn't run yet: this is a soft cross-component dependency.
    """
    stations_path = output_dir / "stations.csv"
    if not stations_path.is_file():
        return [
            f"Contributing-gauges cross-reference: not computed -- "
            f"{stations_path} not found; run Component 02 first."
        ]
    stations = load_stations(stations_path)
    use_in_study = stations["use_in_study"]
    if use_in_study.dtype != bool:
        use_in_study = use_in_study.astype(str).str.strip().str.lower() == "true"
    usable = stations[use_in_study]
    station_nos = pd.to_numeric(usable["station_no"], errors="coerce").dropna()
    study_station_nos = set(station_nos.astype(int))
    all_gauges = (
        frozenset().union(*gauges_by_hour.values()) if gauges_by_hour else frozenset()
    )
    matched = study_station_nos & all_gauges
    hours_with_a_match = sum(
        1 for gauges in gauges_by_hour.values() if gauges & study_station_nos
    )
    return [
        f"Contributing-gauges cross-reference (by station_no, not a unique "
        f"key project-wide): {len(matched)} of {len(study_station_nos)} study "
        f"stations appear in at least one written hour's contributing_gauges, "
        f"in {hours_with_a_match} of {len(gauges_by_hour)} written hours.",
    ]


# D13.5: the committed offline samples --sample points archive-mode layers
# at, so the readers can be tried with no Gadi access. Real, complete for
# satellite (one hour, six 10-minute files); real but incomplete for radar
# (two of the four 15-minute files a complete hour needs) -- see
# tests/data/ingestion/README.md.
SAMPLE_ROOTS = {
    "radar": "tests/data/ingestion/radar",
    "satellite": "tests/data/ingestion/satellite",
}
DEFAULT_SAMPLE_OUTPUT_DIR = "local/grid-sample"


def build(config_path, check_only=False, sample=False):
    """Validate all enabled layers before writing; never modify gauge outputs/raw."""
    config = yaml.safe_load(Path(config_path).read_text())
    if sample:
        for layer, settings in config.get("layers", {}).items():
            if settings.get("root") and layer in SAMPLE_ROOTS:
                settings["root"] = SAMPLE_ROOTS[layer]
        config["output_dir"] = DEFAULT_SAMPLE_OUTPUT_DIR
    layers = config["layers"]
    if not layers or set(layers) - {"radar", "satellite"}:
        raise ValueError("configure at least one of radar or satellite")
    raw_dir, output_dir = Path(config["raw_dir"]), Path(config["output_dir"])
    if output_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("output_dir must not be inside immutable raw_dir")
    # A config where every layer reads the archive in place (D13.2/D13.3) has
    # no pickle input, so Component 01's index -- which only that path needs
    # -- is not required either. Reading it unconditionally would block
    # --sample (and any all-archive config) on a raw_dir that never exists.
    index = None
    if any(not settings.get("root") for settings in layers.values()):
        index = read_index(raw_dir, dtype=str, keep_default_na=False)
        if not {"path", "status"}.issubset(index.columns):
            raise ValueError("raw index needs path and status columns")
    grids = []
    report = [
        "# Radar and satellite adapter",
        "",
        "Pickle inputs are reshaped without unit conversion or resampling.",
        "Archive inputs are summed or averaged to hours and subsampled by the",
        "rules stated under each layer. No layer is gap-filled.",
        "The contract's rain_rate column retains the source values; its name is not evidence of units.",
        "",
    ]
    for layer, settings in layers.items():
        archive_root = settings.get("root")
        if archive_root and layer == "radar":
            # D13.2: read the archive in place rather than a team-produced
            # pickle. Local import avoids a circular import with
            # radar_archive, which needs GridData from this module.
            from src.canonical.radar_archive import build_radar_grid_data

            grid = build_radar_grid_data(
                Path(archive_root),
                config["start"],
                config["end"],
                settings.get("subsample_stride", 10),
            )
        elif archive_root and layer == "satellite":
            # D13.3: same idea, a different archive shape -- see
            # docs/grids/SATELLITE_SOURCE.md for why crop/subsample_stride
            # default the way they do.
            from src.canonical.satellite_archive import build_satellite_grid_data

            grid = build_satellite_grid_data(
                Path(archive_root),
                config["start"],
                config["end"],
                settings["crop"],
                settings.get("subsample_stride", 1),
            )
        else:
            data_file = indexed_path(raw_dir, index, settings["data_file"])
            coordinates_file = indexed_path(
                raw_dir, index, settings["coordinates_file"]
            )
            grid = load_grid(data_file, coordinates_file, layer)
        grids.append(grid)
        expected = pd.date_range(grid.timestamps[0], grid.timestamps[-1], freq="h")
        missing = expected.difference(grid.timestamps)
        provenance = grid.provenance or {}
        report.extend(
            [
                f"## {layer}",
                "",
                f"Nodes: {len(grid.coordinates)}; source hours: {len(grid.timestamps)}; rows: {len(grid.coordinates) * len(grid.timestamps)}.",
                f"Coordinate system: {provenance.get('coordinate_system', 'UNKNOWN')}.",
                "Native time step: "
                f"{provenance.get('native_time_step_minutes', 'UNKNOWN')} minutes.",
                f"Range: {grid.timestamps[0].isoformat()} to {grid.timestamps[-1].isoformat()}.",
                f"Absent hours within that range: {len(missing)}. Boundaries outside the source range are not inferred.",
                _period_coverage_line(grid, config["start"], config["end"]),
                f"Missing rain payloads: {np.isnan(grid.values[:, :, 0]).sum()}.",
            ]
        )
        if layer == "satellite":
            report.append(
                f"Source unavailable node-hours: {(grid.values[:, :, 1] == 0).sum()}."
            )
        if layer == "radar" and "contributing_gauges_by_hour" in provenance:
            report.extend(
                _contributing_gauges_report(
                    provenance["contributing_gauges_by_hour"], output_dir
                )
            )
        if archive_root:
            report.append(f"Input root: {archive_root} (read in place, not hashed).")
            report.extend(_archive_declarations(provenance))
            report.append(
                f"evidence (grids.yaml): {settings.get('evidence', 'UNKNOWN')}."
            )
        else:
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
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "D13.5: point every archive-mode layer (root already set) at its "
            "committed offline sample under tests/data/ingestion/ instead of "
            f"the configured root, and write to {DEFAULT_SAMPLE_OUTPUT_DIR}/ "
            "instead of the real output_dir -- try the readers with no Gadi "
            "access"
        ),
    )
    args = parser.parse_args(argv)
    try:
        build(args.config, args.check_inputs, args.sample)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
