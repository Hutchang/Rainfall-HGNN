"""Exercise selected modalities through five-fold training and checkpoint replay."""

import runpy
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from models.gnn import GNNInductiveHetero
from src.models.contracts import load_inputs, validate_predictions
from src.models.graphs import HourlyGraphDataset, load_graph, validate_graph
from src.models.run import (
    check_inputs,
    read_fold,
    read_settings,
    run,
    validate_settings,
)
from src.models.trainer import masked_forward, model_spec


@pytest.fixture
def sample(tmp_path):
    make = runpy.run_path(str(Path(__file__).parent / "data/models/make_fixture.py"))[
        "make_fixture"
    ]
    settings = read_settings(make(tmp_path / "sample", ["radar", "satellite"]))
    settings["num_layers"] = 2
    settings["dropout"] = 0.0
    return settings


@pytest.mark.parametrize(
    "modalities", [("radar",), ("satellite",), ("radar", "satellite")]
)
def test_five_fold_multimodal_training_and_replay(sample, tmp_path, modalities):
    # Each experiment supplies matching 04 graphs; the production runner does not
    # modify graph structure to create a private ablation.
    sample["input_layers"] = ["raingauge", *modalities]
    for path in Path(sample["graphs_dir"]).glob("fold-*/*.pt"):
        graph = load_graph(path)
        for kind in {"radar", "satellite"} - set(modalities):
            for edge in list(graph.edge_types):
                if kind in (edge[0], edge[2]):
                    del graph[edge]
            del graph[kind]
        torch.save(graph, path)
    assert check_inputs(dict(sample)) == {"folds": 5, "target_rows": 29}
    first = run(sample, tmp_path / "first")
    repeat = run(sample, tmp_path / "repeat")
    replay = run(sample, tmp_path / "replay", first / "best.pt")

    def predictions(directory):
        result = pd.read_csv(directory / "predictions.csv", dtype={"station_id": str})
        result.timestamp = pd.to_datetime(result.timestamp)
        return result

    result = predictions(first)
    pd.testing.assert_frame_equal(result, predictions(repeat), check_exact=True)
    pd.testing.assert_frame_equal(result, predictions(replay), check_exact=True)
    gauge, split, _ = load_inputs(sample["gauge_file"], sample["split_file"])
    validate_predictions(result, gauge, split)
    assert len(result) == 87
    assert "tensor checks do not certify" in (first / "report.md").read_text()
    checkpoint = torch.load(first / "best.pt", weights_only=True)
    for fold in checkpoint["folds"].values():
        assert set(fold["model_spec"]["in_channels_dict"]) == set(
            sample["input_layers"]
        )
        assert all(torch.isfinite(value).all() for value in fold["state_dict"].values())


def test_modalities_reach_output_and_hidden_gauge_cannot_leak(sample):
    gauge, split, _ = load_inputs(sample["gauge_file"], sample["split_file"])
    graph = read_fold(sample, split["folds"][0], gauge)["test"]
    batch = Batch.from_data_list(
        [HourlyGraphDataset(graph)[0], HourlyGraphDataset(graph)[1]]
    )
    assert batch["satellite"].x[2].tolist() == [0.0, 0.0]
    assert load_graph(Path(sample["graphs_dir"]) / "fold-0/test.pt")["satellite"].x[
        0, 1
    ].tolist() == [999.0, 0.0]
    model = GNNInductiveHetero(**model_spec(graph, sample))
    model.eval()
    masked_forward(model, batch, batch["raingauge"].mask)  # materialize lazy layers
    # Positive fixed weights make the causal test independent of random ReLU gates.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    original = masked_forward(model, batch, batch["raingauge"].mask)
    for kind in ("radar", "satellite"):
        changed = batch.clone()
        changed[kind].x[:, 0] += 2
        assert not torch.equal(
            original, masked_forward(model, changed, changed["raingauge"].mask)
        ), f"{kind} values never reached the prediction"
    changed = batch.clone()
    changed["raingauge"].x[changed["raingauge"].mask, :2] = 10000
    torch.testing.assert_close(
        original,
        masked_forward(model, changed, changed["raingauge"].mask),
        rtol=0,
        atol=0,
    )
    model.zero_grad()
    original.sum().backward()
    for kind in ("radar", "satellite"):
        assert any(
            kind in name
            and parameter.grad is not None
            and parameter.grad.abs().sum() > 0
            for name, parameter in model.named_parameters()
        ), f"{kind} relation received no gradient"


@pytest.mark.parametrize(
    "damage, message",
    [
        ("radar_missing", "observed rainfall"),
        ("radar_width", "radar.x"),
        ("satellite_flag", "validity"),
        ("satellite_observed_missing", "observed rainfall"),
        ("no_incoming", "incoming relation"),
        ("no_outgoing", "relation to raingauge"),
        ("missing_hour", "nodes,time"),
    ],
)
def test_bad_modality_contract_fails_before_training(sample, damage, message):
    gauge, split, _ = load_inputs(sample["gauge_file"], sample["split_file"])
    graph = load_graph(Path(sample["graphs_dir"]) / "fold-0/test.pt")
    if damage == "radar_missing":
        graph["radar"].x[0, 0, 0] = float("nan")
    elif damage == "radar_width":
        graph["radar"].x = graph["radar"].x.expand(-1, -1, 2)
    elif damage == "satellite_flag":
        graph["satellite"].x[0, 0, 1] = 0.5
    elif damage == "satellite_observed_missing":
        graph["satellite"].x[0, 0, 0] = float("nan")
    elif damage == "no_incoming":
        del graph["raingauge", "rev_connects", "radar"]
    elif damage == "no_outgoing":
        del graph["radar", "connects", "raingauge"]
    elif damage == "missing_hour":
        graph["satellite"].x = graph["satellite"].x[:, :-1]
    with pytest.raises(ValueError, match=message):
        validate_graph(graph, split["folds"][0], "test", gauge, sample["input_layers"])


@pytest.mark.parametrize(
    "layers",
    [[], ["radar"], ["raingauge", "radar", "radar"], ["raingauge", "unsupported"]],
)
def test_invalid_layer_selection(sample, layers):
    sample["input_layers"] = layers
    with pytest.raises(ValueError, match="input_layers"):
        validate_settings(sample)
