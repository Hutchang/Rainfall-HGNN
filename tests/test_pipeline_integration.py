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
from src.graph.builder_new import main as graph_main
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


@pytest.mark.parametrize("reverse_cross_support", [False, True])
def test_raw_tables_to_splits_graphs_training_and_replay(
    pipeline, reverse_cross_support
):
    directory = pipeline
    config_path = directory / "graph.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["layer_k_neighbours"] == 9
    # Omitting the new key must default to one-way edges, even for an older config.
    config.pop("reverse_cross_support")
    if reverse_cross_support:
        config["reverse_cross_support"] = True
    config_path.write_text(yaml.safe_dump(config))
    build_baseline_graphs(directory / "graph.yaml")
    report = (directory / "graphs/report.md").read_text()
    assert f"reverse_cross_support={reverse_cross_support}" in report
    for kind in ("radar", "satellite"):
        assert f"('{kind}', 'connects', 'raingauge')" in report
        assert (f"('raingauge', 'rev_connects', '{kind}')" in report) == (
            reverse_cross_support
        )
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
        for kind in ("radar", "satellite"):
            edge = graph[kind, "connects", "raingauge"].edge_index
            assert torch.bincount(edge[1]).tolist() == [9] * len(graph.station_ids)
            reverse = ("raingauge", "rev_connects", kind)
            assert (reverse in graph.edge_types) == reverse_cross_support
            if reverse_cross_support:
                assert torch.equal(graph[reverse].edge_index, edge.flip(0))
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
    radar[radar.timestamp != radar.timestamp.iloc[0]].to_csv(
        directory / "standard/radar_hourly.csv", index=False
    )
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


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_invalid_reverse_setting_fails_before_writing(tmp_path, value):
    config = yaml.safe_load(Path("config/graphs/gauge_baseline.yaml").read_text())
    config["reverse_cross_support"] = value
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "graphs"
    with pytest.raises(ValueError, match="reverse_cross_support must be a boolean"):
        build_baseline_graphs(path, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("input_layers", ["radar"], "input_layers must select raingauge"),
        ("include_lpe", False, "requires four positional features"),
        ("k_neighbours", None, "missing explicit graph settings"),
    ],
)
def test_invalid_graph_settings_never_create_outputs(tmp_path, key, value, message):
    config = yaml.safe_load(Path("config/graphs/gauge_baseline.yaml").read_text())
    if value is None:
        config.pop(key)
    else:
        config[key] = value
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "graphs"
    with pytest.raises(ValueError, match=message):
        build_baseline_graphs(path, output_dir=output)
    assert not output.exists()


def test_graph_cli_reports_failure_and_refuses_to_overwrite(pipeline, capsys):
    config_path = pipeline / "graph.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["layer_k_neighbours"] = 13  # Only twelve sensor nodes are available.
    config_path.write_text(yaml.safe_dump(config))
    output = pipeline / "failed-graphs"
    with pytest.raises(ValueError, match="exceeds the available sensor nodes"):
        graph_main([str(config_path), "--output-dir", str(output)])
    report_path = output / "report.md"
    before = report_path.read_bytes()
    assert b"Status: FAILED: ValueError" in before
    assert not list(output.glob("fold-*/*.pt"))
    with pytest.raises(FileExistsError, match="graph output already exists"):
        graph_main([str(config_path), "--output-dir", str(output)])
    assert report_path.read_bytes() == before

    config["layer_k_neighbours"] = 9
    config_path.write_text(yaml.safe_dump(config))
    completed = pipeline / "completed-graphs"
    graph_main([str(config_path), "--output-dir", str(completed)])
    assert str(completed) in capsys.readouterr().out
    assert "Status: BUILT" in (completed / "report.md").read_text()
    assert len(list(completed.glob("fold-*/*.pt"))) == 15
