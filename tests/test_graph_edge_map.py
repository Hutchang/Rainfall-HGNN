"""The fold connection map is drawn from built graphs without changing them."""

import runpy
from pathlib import Path

import pandas as pd
import pytest

from src.graph.edge_map import GAUGE_EDGES, edge_segments, fold_roles, main
from src.models.graphs import load_graph
from src.models.run import read_settings


@pytest.fixture
def built(tmp_path):
    make = runpy.run_path(str(Path(__file__).parent / "data/models/make_fixture.py"))[
        "make_fixture"
    ]
    return read_settings(make(tmp_path / "sample"))


def test_every_connection_is_drawn_once_and_sorted_by_what_it_reaches(built):
    fold_dir = Path(built["graphs_dir"]) / "fold-0"
    graphs = {
        role: load_graph(fold_dir / f"{role}.pt")
        for role in ("train", "validation", "test")
    }
    roles = fold_roles(graphs["train"], graphs["validation"], graphs["test"])
    stations = pd.read_csv(built["stations_file"], dtype={"station_id": str})
    positions = {
        row.station_id: (row.longitude, row.latitude) for row in stations.itertuples()
    }
    split = pd.read_json(built["split_file"])["folds"][0]["ml"]
    assert roles == {station_id: role for role in split for station_id in split[role]}

    test_graph = graphs["test"]
    ids = list(test_graph.station_ids)
    senders, receivers = test_graph[GAUGE_EDGES].edge_index.tolist()
    pairs = {
        frozenset((ids[a], ids[b])) for a, b in zip(senders, receivers, strict=True)
    }
    segments = edge_segments(test_graph, positions, roles)
    assert sum(len(lines) for lines in segments.values()) == len(pairs)
    # The training graph holds no hidden gauge, so nothing can touch one.
    training = edge_segments(graphs["train"], positions, roles)
    assert not training["into_hidden"] and not training["between_hidden"]
    assert segments["into_hidden"] or segments["between_hidden"]


def test_command_writes_one_map_per_fold_and_refuses_an_empty_directory(
    built, tmp_path, capsys
):
    graphs_dir = Path(built["graphs_dir"])
    before = {path: path.stat().st_mtime_ns for path in graphs_dir.rglob("*.pt")}

    assert main([str(graphs_dir), "--stations", built["stations_file"]]) == 0

    written = capsys.readouterr().out.split()
    assert written == [str(graphs_dir / f"fold-{i}" / "edge_map.png") for i in range(5)]
    for path in written:
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert before == {
        path: path.stat().st_mtime_ns for path in graphs_dir.rglob("*.pt")
    }
    with pytest.raises(FileNotFoundError, match="no fold-"):
        main([str(tmp_path), "--stations", built["stations_file"]])
