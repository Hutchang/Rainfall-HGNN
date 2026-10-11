"""Draw a built fold's gauge graph on a map: who is connected to whom.

    uv run python -m src.graph.edge_map database/graphs/gauge_blocked \
        --stations database/standard/stations.csv

Writes ``fold-<i>/edge_map.png`` inside the graph directory, which is gitignored: the
figure is a local review aid, regenerated from the graph files at any time.
"""

import argparse
from math import cos, radians
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

from src.models.graphs import load_graph

GAUGE_EDGES = ("raingauge", "connects", "raingauge")
ROLE_STYLES = {
    "train": {"color": "#BBBBBB", "marker": "o", "label": "train"},
    "validation": {"color": "#E69F00", "marker": "s", "label": "validation"},
    "test": {"color": "#0072B2", "marker": "^", "label": "hidden (test)"},
}


def fold_roles(train_graph, validation_graph, test_graph):
    """Each station's role, read from which of the fold's graphs first contains it."""
    roles = dict.fromkeys(test_graph.station_ids, "test")
    roles.update(dict.fromkeys(validation_graph.station_ids, "validation"))
    roles.update(dict.fromkeys(train_graph.station_ids, "train"))
    return roles


def edge_segments(graph, positions, roles):
    """Split a graph's connections into three kinds, each drawn once per pair.

    ``visible``: both ends can be read. ``into_hidden``: a hidden gauge and a visible
    one, the direct route for information to reach it. ``between_hidden``: two hidden
    gauges, which can only relay what reached one of them in an earlier layer.
    """
    ids = list(graph.station_ids)
    senders, receivers = graph[GAUGE_EDGES].edge_index.tolist()
    kinds = {"visible": set(), "into_hidden": set(), "between_hidden": set()}
    for sender, receiver in zip(senders, receivers, strict=True):
        pair = tuple(sorted((ids[sender], ids[receiver])))
        hidden_ends = sum(roles[station_id] == "test" for station_id in pair)
        kind = ("visible", "into_hidden", "between_hidden")[hidden_ends]
        kinds[kind].add(pair)
    return {
        kind: [(positions[a], positions[b]) for a, b in sorted(pairs)]
        for kind, pairs in kinds.items()
    }


def draw_fold(fold_dir, positions, path):
    graphs = {
        role: load_graph(fold_dir / f"{role}.pt")
        for role in ("train", "validation", "test")
    }
    roles = fold_roles(graphs["train"], graphs["validation"], graphs["test"])
    missing = sorted(set(roles) - set(positions))
    if missing:
        raise ValueError(
            f"{fold_dir}: stations absent from the station table: {missing}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    panels = (
        ("train", "Training graph: what the model learns on"),
        ("test", "Test graph: what it predicts on"),
    )
    for axis, (role, title) in zip(axes, panels, strict=True):
        graph = graphs[role]
        segments = edge_segments(graph, positions, roles)
        for kind, colour, width, label in (
            ("visible", "#999999", 0.6, "between visible gauges"),
            ("between_hidden", "#56B4E9", 0.9, "between two hidden gauges (relay)"),
            ("into_hidden", "#D55E00", 1.3, "hidden gauge to a visible gauge"),
        ):
            if segments[kind]:
                axis.add_collection(
                    LineCollection(
                        segments[kind],
                        colors=colour,
                        linewidths=width,
                        label=f"{label} ({len(segments[kind])})",
                        zorder=1,
                    )
                )
        for name, style in ROLE_STYLES.items():
            points = [positions[s] for s in graph.station_ids if roles[s] == name]
            if points:
                axis.scatter(
                    [x for x, _ in points],
                    [y for _, y in points],
                    s=26,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.3,
                    label=f"{style['label']} ({len(points)})",
                    zorder=2,
                )
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="lower right")
    mean_latitude = sum(y for _, y in positions.values()) / len(positions)
    for axis in axes:
        # Keep distances on the page proportional to kilometres.
        axis.set_aspect(1 / cos(radians(mean_latitude)))
    axes[0].set_ylabel("latitude")
    fig.suptitle(f"{fold_dir.parent.name} / {fold_dir.name}: gauge connections")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.graph.edge_map")
    parser.add_argument("graphs_dir", type=Path)
    parser.add_argument("--stations", type=Path, required=True)
    arguments = parser.parse_args(argv)

    stations = pd.read_csv(arguments.stations, dtype={"station_id": str})
    positions = {
        row.station_id: (row.longitude, row.latitude) for row in stations.itertuples()
    }
    fold_dirs = sorted(arguments.graphs_dir.glob("fold-*"))
    if not fold_dirs:
        raise FileNotFoundError(f"no fold-* directories under {arguments.graphs_dir}")
    for fold_dir in fold_dirs:
        path = fold_dir / "edge_map.png"
        draw_fold(fold_dir, positions, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
