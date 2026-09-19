"""Offline tests for identity, missingness, alignment and the CSV/04 interfaces."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.canonical.grids import build, load_grid, main


@pytest.fixture
def bundle(tmp_path):
    sample = json.loads((Path(__file__).parent / "data/grids/sample.json").read_text())
    raw = tmp_path / "raw"
    raw.mkdir()
    rows, layers = [], {}
    for layer in ("radar", "satellite"):
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(sample["timestamps"]),
                "data": [np.array(value, dtype=np.float32) for value in sample[layer]],
            }
        ).to_pickle(raw / f"{layer}.pkl")
        pd.DataFrame(sample["coordinates"], columns=["latitude", "longitude"]).to_csv(
            raw / f"{layer}_coords.csv", index=False
        )
        layers[layer] = {
            "data_file": f"{layer}.pkl",
            "coordinates_file": f"{layer}_coords.csv",
        }
        rows.extend({"path": name, "status": "ok"} for name in layers[layer].values())
    pd.DataFrame(rows).to_csv(raw / "index.csv", index=False)
    config = tmp_path / "grids.yaml"
    settings = {
        "raw_dir": str(raw),
        "output_dir": str(tmp_path / "standard"),
        "layers": layers,
    }
    config.write_text(yaml.safe_dump(settings))
    return raw, config, settings


def read(bundle, layer="radar"):
    raw = bundle[0]
    return load_grid(raw / f"{layer}.pkl", raw / f"{layer}_coords.csv", layer)


def test_csv_preserves_values_flags_identity_and_absent_hours(bundle):
    raw, config, settings = bundle
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    build(config)
    out = Path(settings["output_dir"])
    radar = pd.read_csv(out / "radar_hourly.csv")
    satellite = pd.read_csv(out / "satellite_hourly.csv")
    assert radar.rain_rate.tolist() == [0, 1.5, 2.5, 0]
    assert radar.node_id.tolist() == [0, 1, 0, 1]
    assert satellite.rain_rate.tolist() == [0, 999, 1.25, 0]
    assert satellite.valid.tolist() == [1, 0, 1, 1]
    assert set(pd.to_datetime(radar.timestamp).dt.hour) == {0, 2}
    assert radar.timestamp.str.endswith("+00:00").all()
    assert pd.read_csv(out / "radar_nodes.csv").longitude.tolist() == [151.1, 151.2]
    report = (out / "grid_report.md").read_text()
    assert "Absent hour: 2022-01-01T01:00:00+00:00" in report
    assert "units: UNKNOWN" in report
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()
    }


def test_reproducible_csv_and_no_overwrite(bundle, tmp_path):
    _, config, settings = bundle
    build(config)
    first = Path(settings["output_dir"])
    with pytest.raises(FileExistsError):
        build(config)
    settings["output_dir"] = str(tmp_path / "repeat")
    config.write_text(yaml.safe_dump(settings))
    build(config)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in Path(settings["output_dir"]).iterdir()
    }


def test_graph_order_alignment_and_invalid_satellite_mask(bundle):
    grid = read(bundle, "satellite")
    graph, coords = grid.to_heterodata([grid.timestamps[1]])
    assert tuple(graph["satellite"].x.shape) == (2, 1, 2)
    assert graph["satellite"].x[:, 0, 0].tolist() == [1.25, 0]
    graph, _ = grid.to_heterodata(grid.timestamps)
    assert graph["satellite"].x[1, 0].tolist() == [0, 0]
    assert grid.values[0, 1, 0] == 999
    assert graph["satellite"].observed.tolist() == [[True, True], [False, True]]
    assert coords.columns.tolist() == ["latitude", "longitude"]
    assert coords.longitude.tolist() == [151.1, 151.2]
    assert graph.edge_types == []


def test_no_silent_missing_radar_frame_or_pixel(bundle):
    grid = read(bundle)
    with pytest.raises(ValueError, match="1 requested hours are absent"):
        grid.to_heterodata(pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC"))
    grid.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="validity-mask interface"):
        grid.to_heterodata(grid.timestamps)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("naive", "timezone-aware"),
        ("duplicate", "unique"),
        ("off_hour", "hourly boundaries"),
        ("unsorted", "sorted"),
        ("shape", "fixed"),
        ("negative", "nonnegative"),
        ("infinite", "infinite"),
    ],
)
def test_malformed_radar_rejected(bundle, mutation, match):
    raw = bundle[0]
    path = raw / "radar.pkl"
    frame = pd.read_pickle(path)
    if mutation == "naive":
        frame.timestamp = frame.timestamp.dt.tz_localize(None)
    elif mutation == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif mutation == "off_hour":
        frame.timestamp += pd.Timedelta(minutes=30)
    elif mutation == "unsorted":
        frame = frame.iloc[::-1]
    elif mutation == "shape":
        frame.at[0, "data"] = np.array([1.0])
    else:
        frame.at[0, "data"][0] = -1 if mutation == "negative" else np.inf
    frame.to_pickle(path)
    with pytest.raises(ValueError, match=match):
        read(bundle)


@pytest.mark.parametrize("bad_flag", [0.5, 2, np.nan])
def test_satellite_flags_are_not_reinterpreted(bundle, bad_flag):
    path = bundle[0] / "satellite.pkl"
    frame = pd.read_pickle(path)
    frame.at[0, "data"][0, 1] = bad_flag
    frame.to_pickle(path)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        read(bundle, "satellite")


def test_missing_coordinates_and_failed_index_block_before_writing(bundle):
    raw, config, settings = bundle
    index = pd.read_csv(raw / "index.csv")
    index.loc[index.path == "satellite_coords.csv", "status"] = "failed"
    index.to_csv(raw / "index.csv", index=False)
    with pytest.raises(ValueError, match="ok raw index row"):
        build(config)
    assert not Path(settings["output_dir"]).exists()
    (raw / "satellite_coords.csv").unlink()
    with pytest.raises(FileNotFoundError, match="missing satellite input"):
        read(bundle, "satellite")


def test_coordinates_must_match_array_order(bundle):
    path = bundle[0] / "radar_coords.csv"
    coords = pd.read_csv(path)
    coords["node_id"] = [1, 0]
    coords.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source array order"):
        read(bundle)


def test_check_inputs_cli_does_not_write(bundle):
    main([str(bundle[1]), "--check-inputs"])
    assert not Path(bundle[2]["output_dir"]).exists()


def test_output_may_not_modify_raw(bundle):
    raw, config, settings = bundle
    settings["output_dir"] = str(raw / "changed")
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="immutable raw_dir"):
        build(config)


def test_node_features_work_with_inherited_model(bundle):
    """Exercise both feature widths through a forward/backward pass, not real topology."""
    import torch

    from src.models.models import GNNInductiveHetero

    x = {"raingauge": torch.ones((2, 6))}
    for name in ("radar", "satellite"):
        grid = read(bundle, name)
        graph, _ = grid.to_heterodata([grid.timestamps[0]])
        x[name] = graph[name].x[:, 0, :]
    # Deliberately synthetic bidirectional test edges, not a geographic graph policy.
    relations = [("raingauge", "connects", "raingauge")]
    for name in ("radar", "satellite"):
        relations.extend(
            [(name, "connects", "raingauge"), ("raingauge", "connects", name)]
        )
    indices = {edge: torch.tensor([[0, 1], [1, 0]]) for edge in relations}
    weights = {edge: torch.ones(2) for edge in relations}
    model = GNNInductiveHetero(
        {name: values.shape[1] for name, values in x.items()}, 4, 1, 2, relations
    )
    result = model(x, indices, weights)["raingauge"]
    assert result.shape == (2, 1) and torch.isfinite(result).all()
    (result - 1).square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
