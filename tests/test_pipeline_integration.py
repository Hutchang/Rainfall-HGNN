"""Run real component implementations against invented raw-format data."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.canonical.config import load_config
from src.canonical.gauge.build import run as build_gauges
from src.canonical.grids import build as build_grids
from src.graph.builder_new import build_baseline_graphs
from src.graph.gaugegraphnew import GaugeGraphNew
from src.models.graphs import load_graph
from src.models.run import check_inputs, read_settings, run
from src.study.split import main as split_main


@pytest.fixture
def pipeline(tmp_path):
    make = runpy.run_path(str(Path(__file__).parent / "data/pipeline/make_fixture.py"))[
        "make_fixture"
    ]
    directory = make(tmp_path / "pipeline")
    build_gauges(load_config(directory / "canonical.yaml"))
    build_grids(directory / "grids.yaml")
    split_main([str(directory / "split.yaml"), "--created", "2026-09-10"])
    return directory


def test_raw_tables_to_splits_graphs_training_and_replay(pipeline):
    directory = pipeline
    build_baseline_graphs(directory / "graph.yaml")
    settings = read_settings(directory / "model.yaml")
    assert check_inputs(dict(settings)) == {"folds": 5, "target_rows": 288}
    first = run(settings, directory / "run")
    replay = run(settings, directory / "replay", first / "best.pt")
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "predictions.csv"),
        pd.read_csv(replay / "predictions.csv"),
        check_exact=True,
    )
    assert len(pd.read_csv(first / "predictions.csv")) == 864
    original = load_graph(directory / "graphs/fold-0/train.pt")
    second_dir = directory / "graphs-repeat"
    build_baseline_graphs(directory / "graph.yaml", output_dir=second_dir)
    repeated = load_graph(second_dir / "fold-0/train.pt")
    for kind in original.node_types:
        torch.testing.assert_close(
            original[kind].x, repeated[kind].x, equal_nan=True, atol=0, rtol=0
        )
    # All edge relations retain their own receiving-node normalization.
    for graph_file in (directory / "graphs").glob("fold-*/*.pt"):
        graph = load_graph(graph_file)
        for relation in graph.edge_types:
            edge = graph[relation]
            assert torch.isfinite(edge.edge_attr).all()
            for destination in edge.edge_index[1].unique():
                assert edge.edge_attr[edge.edge_index[1] == destination].max() == 1
        gauge_edges = graph["raingauge", "connects", "raingauge"].edge_index
        assert (gauge_edges[0] != gauge_edges[1]).all()
        # both_directions adds both directions per pick, so a gauge may now send to more
        # than k_neighbours gauges. What must hold instead is that every gauge
        # receives from at least one gauge, and that no edge lacks its reverse.
        pairs = set(zip(*gauge_edges.tolist(), strict=True))
        assert {destination for _, destination in pairs} == set(
            range(graph["raingauge"].x.shape[0])
        )
        assert all((j, i) in pairs for i, j in pairs)
        assert graph["raingauge"].mask.dtype == torch.bool
        assert (
            graph["raingauge"].x.untyped_storage().data_ptr()
            != graph["raingauge"].y.untyped_storage().data_ptr()
        )
    # Removing a radar hour must fail, not change the 288 required targets.
    radar = pd.read_csv(directory / "standard/radar_hourly.csv")
    radar.iloc[3:].to_csv(directory / "standard/radar_hourly.csv", index=False)
    with pytest.raises(ValueError, match="requested hours are absent"):
        build_baseline_graphs(
            directory / "graph.yaml", output_dir=directory / "missing-radar"
        )


def test_gauge_only_and_held_out_rainfall_cannot_change_training_graph(pipeline):
    cfg = yaml.safe_load((pipeline / "graph.yaml").read_text())
    cfg["input_layers"] = ["raingauge"]
    (pipeline / "gauge-graph.yaml").write_text(yaml.safe_dump(cfg))
    first_dir = build_baseline_graphs(pipeline / "gauge-graph.yaml")
    original = load_graph(first_dir / "fold-0/train.pt")
    # The five positional arguments documented by Component 04 remain callable.
    mapping = pd.read_csv(pipeline / "standard/stations.csv", dtype={"station_id": str})
    mapping = (
        mapping[mapping.use_in_study].sort_values("station_id").reset_index(drop=True)
    )
    mapping = mapping.rename(columns={"station_id": "id"})
    mapping["order"] = range(len(mapping))
    canonical = pd.read_csv(
        pipeline / "standard/gauge_hourly.csv", dtype={"station_id": str}
    )
    canonical["timestamp"] = pd.to_datetime(canonical.timestamp, utc=True)
    wide = canonical.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    split_payload = json.loads((pipeline / "split.json").read_text())
    positional = GaugeGraphNew(
        wide,
        mapping,
        split_payload["folds"][0],
        2,
        "per_node",
        distance="haversine_km",
        both_directions=True,
    )
    assert torch.equal(
        positional.get_train_heterodata()[
            "raingauge", "connects", "raingauge"
        ].edge_attr,
        original["raingauge", "connects", "raingauge"].edge_attr,
    )
    assert original.node_types == ["raingauge"]
    split = json.loads((pipeline / "split.json").read_text())
    table_path = pipeline / "standard/gauge_hourly.csv"
    table = pd.read_csv(table_path, dtype={"station_id": str})
    table.loc[
        table.station_id.isin(split["folds"][0]["ml"]["test"]), "rainfall_mm"
    ] += 1000
    table.to_csv(table_path, index=False)
    changed_dir = build_baseline_graphs(
        pipeline / "gauge-graph.yaml", output_dir=pipeline / "changed-graphs"
    )
    changed = load_graph(changed_dir / "fold-0/train.pt")
    torch.testing.assert_close(
        original["raingauge"].x, changed["raingauge"].x, atol=0, rtol=0
    )
    for relation in original.edge_types:
        assert torch.equal(original[relation].edge_index, changed[relation].edge_index)
        assert torch.equal(original[relation].edge_attr, changed[relation].edge_attr)


def test_sensor_neighbour_uses_correct_coordinate_axes():
    builder = object.__new__(GaugeGraphNew)
    # Latitude-only and longitude-only separations have different distances here.
    gauges = [[-33.0, 151.0]]
    sensors = pd.DataFrame({"latitude": [-33.1, -33.0], "longitude": [151.0, 151.1]})
    pairs, weights = builder.connect_graphs(gauges, sensors, knn=1)
    assert pairs == [(0, 1)]
    assert weights[0] == pytest.approx(1 / 9.32559, rel=1e-4)


def test_released_split_is_preserved_on_rerun(pipeline):
    path = pipeline / "split.json"
    before = path.read_bytes()
    split_main([str(pipeline / "split.yaml"), "--created", "2026-09-10"])
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="released split already exists"):
        split_main([str(pipeline / "split.yaml"), "--created", "2026-09-11"])
    assert path.read_bytes() == before
