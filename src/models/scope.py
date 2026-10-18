"""The data a model run covered: its period, gauge positions and node counts (ISS-39).

Provenance in config.yaml proves which files a run read, by path and hash, but not
what they hold, so a run folder copied off Gadi cannot say which hours, which region
or how many nodes its scores rest on. Everything here is measured from the inputs the
run itself loaded, so it cannot drift from what was trained and scored.
"""

from pathlib import Path

import pandas as pd

ROLES = ("train", "validation", "test")
SENSOR_LAYERS = ("radar", "satellite")


def _span(values):
    return [round(float(values.min()), 4), round(float(values.max()), 4)]


def measure_tables(gauge, stations, split):
    """Period and gauge positions, from the gauge table and stations of the split."""
    hours = gauge.timestamp.drop_duplicates()
    used = sorted(
        {sid for fold in split["folds"] for role in ROLES for sid in fold["ml"][role]}
    )
    placed = stations.set_index("station_id").loc[used]
    return {
        "period": {
            "first_hour_utc": hours.min().isoformat(),
            "last_hour_utc": hours.max().isoformat(),
            "hours": len(hours),
        },
        "gauges": {
            "stations": len(used),
            "latitude": _span(placed.latitude),
            "longitude": _span(placed.longitude),
            "per_fold": {
                fold["fold_index"]: {role: len(fold["ml"][role]) for role in ROLES}
                for fold in split["folds"]
            },
        },
    }


def record_graph_nodes(scope, fold_index, graphs):
    """Node count per layer in each of one fold's train, validation and test graphs."""
    scope.setdefault("graph_nodes", {})[fold_index] = {
        role: {kind: int(graph[kind].x.shape[0]) for kind in graph.node_types}
        for role, graph in graphs.items()
    }


def measure_sensor_layers(scope, gauge_file, digest):
    """Extent of each radar or satellite layer the graphs hold.

    Graph files carry no sensor coordinates, so positions come from Component 02's
    ``<layer>_nodes.csv`` beside the gauge table. The extent is recorded only when that
    file has exactly as many nodes as the graphs; otherwise it is marked UNKNOWN with
    the reason rather than guessed.
    """
    counts = {}
    for fold in scope.get("graph_nodes", {}).values():
        for nodes in fold.values():
            for kind, count in nodes.items():
                if kind in SENSOR_LAYERS:
                    counts.setdefault(kind, set()).add(count)
    layers = {}
    for kind, found in sorted(counts.items()):
        layer = {"nodes": found.pop() if len(found) == 1 else sorted(found)}
        path = Path(gauge_file).parent / f"{kind}_nodes.csv"
        if not path.is_file():
            layer["extent"] = f"UNKNOWN: no {path}"
        else:
            table = pd.read_csv(path)
            if not isinstance(layer["nodes"], int) or len(table) != layer["nodes"]:
                layer["extent"] = (
                    f"UNKNOWN: {path} has {len(table)} nodes, the graphs have "
                    f"{layer['nodes']}"
                )
            else:
                layer["latitude"] = _span(table.latitude)
                layer["longitude"] = _span(table.longitude)
                layer["node_file"] = {
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                }
        layers[kind] = layer
    if layers:
        scope["layers"] = layers


def scope_lines(scope):
    """The data scope as a Markdown section for report.md."""
    period, gauges = scope["period"], scope["gauges"]
    lines = [
        "## Data scope",
        "",
        "Measured from the inputs this run read; also under data_scope in config.yaml.",
        "",
        f"- Period: {period['first_hour_utc']} to {period['last_hour_utc']}, "
        f"{period['hours']:,} hours in the gauge table",
        f"- Gauges: {gauges['stations']} in the split; latitude "
        f"{gauges['latitude'][0]} to {gauges['latitude'][1]}, longitude "
        f"{gauges['longitude'][0]} to {gauges['longitude'][1]}",
    ]
    for kind, layer in scope.get("layers", {}).items():
        if "extent" in layer:
            lines.append(f"- {kind}: {layer['nodes']} nodes; extent {layer['extent']}")
        else:
            lines.append(
                f"- {kind}: {layer['nodes']} nodes; latitude {layer['latitude'][0]} "
                f"to {layer['latitude'][1]}, longitude {layer['longitude'][0]} to "
                f"{layer['longitude'][1]}"
            )
    for name, report in scope.get("upstream_reports", {}).items():
        if "sha256" in report:
            lines.append(f"- Copied {report['source']} to inputs/{name}")
        else:
            lines.append(f"- No {report['source']} to copy to inputs/{name}")
    nodes = scope.get("graph_nodes")
    if nodes:
        kinds = sorted(
            {kind for fold in nodes.values() for role in fold.values() for kind in role}
        )
        lines += [
            "",
            "| Fold | Graph | " + " | ".join(kinds) + " |",
            "|---|---|" + "---:|" * len(kinds),
        ]
        for index, fold in sorted(nodes.items()):
            for role in ROLES:
                counts = fold[role]
                lines.append(
                    f"| {index} | {role} | "
                    + " | ".join(str(counts.get(kind, 0)) for kind in kinds)
                    + " |"
                )
    return lines
