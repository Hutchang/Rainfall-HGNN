"""Build all Component 04 graphs from published 02 tables and the 03 split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.canonical.gauge.build import load_stations
from src.canonical.grid_tables import load_standard_grid
from src.canonical.grids import utc_hours
from src.graph.gaugegraphnew import GaugeGraphNew
from src.study.split import load_seed_config


def build_baseline_graphs(
    gauge_config="config/graphs/gauge_baseline.yaml",
    data_dir=None,
    output_dir=None,
    split_file=None,
    seeds_file="config/seeds.yaml",
):
    config = yaml.safe_load(Path(gauge_config).read_text())
    data_dir = Path(data_dir or config.get("data_dir", "database/standard"))
    output_dir = Path(
        output_dir or config.get("output_dir", "database/graphs/gauge_baseline")
    )
    split_file = Path(
        split_file or config.get("split_file", "config/splits/sydney_5fold.json")
    )
    layers = config.get("input_layers", ["raingauge"])
    if (
        not isinstance(layers, list)
        or any(not isinstance(x, str) for x in layers)
        or "raingauge" not in layers
        or len(layers) != len(set(layers))
        or set(layers) - {"raingauge", "radar", "satellite"}
    ):
        raise ValueError(
            "input_layers must select raingauge and optional radar/satellite"
        )
    if output_dir.exists():
        raise FileExistsError(f"graph output already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("graph output must not be inside the standard input directory")
    required = (
        "k_neighbours",
        "distance",
        "edge_normalisation",
        "both_directions",
        "include_lpe",
        "lpe_dimensions",
        "layer_is_directed",
    )
    missing = set(required) - set(config)
    if missing:
        raise ValueError(f"missing explicit graph settings: {sorted(missing)}")
    if config["include_lpe"] is not True or config["lpe_dimensions"] != 4:
        raise ValueError("the current 06 contract requires four positional features")
    mapping = load_stations(data_dir / "stations.csv")
    if (
        mapping.station_id.duplicated().any()
        or not mapping.use_in_study.isin([True, False]).all()
    ):
        raise ValueError("stations need unique IDs and boolean use_in_study")
    mapping = (
        mapping.loc[mapping.use_in_study]
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    mapping = mapping.rename(columns={"station_id": "id"})
    mapping["order"] = np.arange(len(mapping))
    table = pd.read_csv(data_dir / "gauge_hourly.csv", dtype={"station_id": str})
    if set(table.columns) != {"station_id", "timestamp", "rainfall_mm"}:
        raise ValueError("unexpected gauge_hourly columns")
    stamps = sorted(table.timestamp.unique(), key=pd.Timestamp)
    hours = utc_hours(stamps, "gauge timeline")
    table["timestamp"] = pd.to_datetime(table.timestamp, utc=True)
    if not np.isfinite(table.rainfall_mm).all() or (table.rainfall_mm < 0).any():
        raise ValueError("canonical rainfall must be finite nonnegative millimetres")
    wide = table.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    ).reindex(index=hours, columns=mapping.id)
    split = json.loads(split_file.read_text())
    folds = split["folds"]
    if len(folds) != 5 or [fold["fold_index"] for fold in folds] != list(range(5)):
        raise ValueError("expected five ordered released folds")
    eligible = set(mapping.id)
    all_test = []
    for fold in folds:
        roles = [fold["ml"][role] for role in ("train", "validation", "test")]
        assigned = [sid for role in roles for sid in role]
        if (
            any(not role for role in roles)
            or len(assigned) != len(set(assigned))
            or set(assigned) != eligible
        ):
            raise ValueError(
                f"fold {fold['fold_index']}: roles must partition eligible station IDs"
            )
        all_test.extend(fold["ml"]["test"])
    if len(all_test) != len(set(all_test)) or set(all_test) != eligible:
        raise ValueError("test folds must cover eligible stations exactly once")
    seeds = load_seed_config(Path(seeds_file))
    np.random.seed(seeds.model_seed)
    torch.manual_seed(seeds.model_seed)
    sensors = {}
    for layer in layers:
        if layer != "raingauge":
            sensors[layer] = load_standard_grid(data_dir, layer).to_heterodata(hours)
    output_dir.mkdir(parents=True)
    report = [
        "# Component 04 graph build",
        "",
        "Status: BUILDING",
        "",
        f"Settings: {config}",
        f"Shared model_seed for LPE: {seeds.model_seed}",
        "",
        (
            "Numerical fixes: retain directed gauge edges; normalize by receiving "
            "node; add both directions for every neighbour pick when "
            "both_directions is set, so no gauge is left without an incoming edge "
            "(DEC-4), which also makes the graph Laplacian symmetric so the "
            "positional encoding uses the symmetric solver; finite 1e6 "
            "inverse-distance weight for coincident coordinates; "
            "remove self neighbours by identity; preserve binary validity and physical "
            "gauge millimetres; independent y storage; sensor coordinates consistently "
            "latitude/longitude."
        ),
        "",
        (
            "Sensor topology: each gauge selects the configured nearest sensor nodes; "
            "both directions are stored and normalized independently. Source product/"
            "units, coverage and held-out-gauge independence remain subject to review."
        ),
        "",
    ]
    paths = [
        Path(gauge_config),
        split_file,
        Path(seeds_file),
        data_dir / "gauge_hourly.csv",
        data_dir / "stations.csv",
    ]
    paths += [
        data_dir / f"{layer}_{suffix}.csv"
        for layer in sensors
        for suffix in ("hourly", "nodes")
    ]
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report.append(f"Input: {path}; SHA256: {digest}")
    try:
        for fold in folds:
            builder = GaugeGraphNew(
                wide,
                mapping,
                fold,
                knn=config["k_neighbours"],
                distance=config["distance"],
                edge_normalisation=config["edge_normalisation"],
                both_directions=config["both_directions"],
                include_lpe=config["include_lpe"],
                lpe_dimensions=config["lpe_dimensions"],
                layer_is_directed=config["layer_is_directed"],
            )
            for layer, (features, coords) in sensors.items():
                builder.add_heterodata(
                    features, coords, layer, knn=config["layer_k_neighbours"]
                )
            folder = output_dir / f"fold-{fold['fold_index']}"
            folder.mkdir()
            for role in ("train", "validation", "test"):
                graph = getattr(builder, f"get_{role}_heterodata")()
                torch.save(graph, folder / f"{role}.pt")
                counts = {kind: graph[kind].x.shape[0] for kind in graph.node_types}
                report.append(
                    f"fold {fold['fold_index']} {role}: nodes {counts}; "
                    f"hours {len(hours)}"
                )
                for relation in graph.edge_types:
                    edge = graph[relation]
                    report.append(
                        f"  {relation}: {edge.edge_index.shape[1]} edges; weights "
                        f"{edge.edge_attr.min().item():.6g} to "
                        f"{edge.edge_attr.max().item():.6g}"
                    )
                incoming = set(
                    graph["raingauge", "connects", "raingauge"].edge_index[1].tolist()
                )
                isolated = [
                    sid for i, sid in enumerate(graph.station_ids) if i not in incoming
                ]
                if isolated:
                    report.append(
                        f"  Gauge nodes with no incoming gauge edge: {isolated}"
                    )
        report[2] = "Status: BUILT — validate with Component 06 before training"
    except Exception as error:
        report[2] = f"Status: FAILED: {type(error).__name__}: {error}"
        raise
    finally:
        (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", default="config/graphs/gauge_baseline.yaml"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--seeds", default="config/seeds.yaml")
    args = parser.parse_args(argv)
    print(
        build_baseline_graphs(
            args.config, args.data_dir, args.output_dir, args.split_file, args.seeds
        )
    )


if __name__ == "__main__":
    main()
