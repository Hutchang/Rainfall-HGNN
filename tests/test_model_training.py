"""Component 06 acceptance: five folds, missingness, identities and checkpoint replay."""

import runpy
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import Batch

from src.models.contracts import load_inputs, target_rows, validate_predictions
from src.models.graphs import HourlyGraphDataset, load_graph, validate_graph
from src.models.run import check_inputs, read_fold, read_settings, run
from src.models.scaling import rainfall_scaling
from src.models.trainer import masked_forward, predict_fold, restore_model


@pytest.fixture
def inputs(tmp_path):
    make = runpy.run_path(str(Path(__file__).parent / "data/models/make_fixture.py"))[
        "make_fixture"
    ]
    config_path = make(tmp_path / "sample")
    return read_settings(config_path)


def read_predictions(directory):
    result = pd.read_csv(directory / "predictions.csv", dtype={"station_id": str})
    result.timestamp = pd.to_datetime(result.timestamp)
    return result


def test_five_fold_cpu_reproducibility_and_checkpoint_replay(
    inputs, tmp_path, monkeypatch
):
    original_step = torch.optim.Adam.step
    parameter_updates = []

    def inspect_update(optimizer, *args, **kwargs):
        before = [
            (parameter, parameter.detach().clone())
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        result = original_step(optimizer, *args, **kwargs)
        parameter_updates.append(
            any(not torch.equal(parameter, previous) for parameter, previous in before)
        )
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", inspect_update)
    start = time.monotonic()
    first = run(inputs, tmp_path / "first")
    assert time.monotonic() - start < 60, (
        "two-epoch five-fold CPU fixture exceeded one minute"
    )
    second = run(inputs, tmp_path / "second")
    relocated = tmp_path / "relocated-inputs"
    shutil.copytree(Path(inputs["gauge_file"]).parent, relocated)
    replay_inputs = dict(inputs)
    for key in (
        "gauge_file",
        "stations_file",
        "split_file",
        "seeds_file",
        "graphs_dir",
    ):
        replay_inputs[key] = str(relocated / Path(inputs[key]).name)
    restored = run(replay_inputs, tmp_path / "restored", first / "best.pt")
    original = read_predictions(first)
    pd.testing.assert_frame_equal(original, read_predictions(second), check_exact=True)
    pd.testing.assert_frame_equal(
        original, read_predictions(restored), check_exact=True
    )
    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    validate_predictions(original, gauge, split)
    assert len(original) == 3 * len(target_rows(gauge, split)) == 87
    assert set(original.method) == {"hgnn", "idw", "network_mean"}
    assert original.station_id.str.startswith("0").all()
    hgnn = original.loc[original.method.eq("hgnn")]
    assert hgnn.predicted_mm.gt(0).all()
    assert hgnn.groupby("fold").predicted_mm.nunique().gt(1).all()
    assert parameter_updates and all(parameter_updates), (
        "an optimizer step failed to update any parameter"
    )
    history = pd.read_csv(first / "history.csv")
    assert set(history.columns) == {
        "fold",
        "epoch",
        "train_loss",
        "val_loss",
        "best_val_loss",
        "deteriorations",
        "patience_remaining",
        "early_stop_status",
    }
    assert history.early_stop_status.isin(["IMPROVED", "WATCHING", "TRIGGERED"]).all()
    assert history.groupby("fold").size().tolist() == [2] * 5
    checkpoint = torch.load(first / "best.pt", weights_only=True)
    repeated_checkpoint = torch.load(second / "best.pt", weights_only=True)
    for fold in range(5):
        assert checkpoint["folds"][fold]["val_loss"] == pytest.approx(
            history.loc[history.fold.eq(fold), "val_loss"].min()
        )
        first_state = checkpoint["folds"][fold]["state_dict"]
        repeated_state = repeated_checkpoint["folds"][fold]["state_dict"]
        assert first_state.keys() == repeated_state.keys()
        for name, parameter in first_state.items():
            assert torch.equal(parameter, repeated_state[name]), (
                f"fold {fold}: {name} is not reproducible"
            )
    assert "SYNTHETIC FIXTURE ONLY" in (first / "report.md").read_text()

    # Deliberately force a known positive output so replay cannot pass just because
    # the evaluation clamp turned two unrelated negative models into zero.
    probe = checkpoint["folds"][0]
    probe["state_dict"]["lin.weight"].zero_()
    probe["state_dict"]["lin.bias"].fill_(1.25)
    probe_path = tmp_path / "nonzero-probe.pt"
    torch.save(probe, probe_path)
    replayed_probe = torch.load(probe_path, weights_only=True)
    original_model = restore_model(probe, checkpoint["settings"])
    replayed_model = restore_model(replayed_probe, checkpoint["settings"])
    for name, parameter in original_model.state_dict().items():
        assert torch.equal(parameter, replayed_model.state_dict()[name])
    graph = read_fold(inputs, split["folds"][0], gauge)["test"]
    probe_predictions = predict_fold(original_model, graph, 0, checkpoint["settings"])
    pd.testing.assert_frame_equal(
        probe_predictions,
        predict_fold(replayed_model, graph, 0, checkpoint["settings"]),
        check_exact=True,
    )
    assert probe_predictions.predicted_mm.eq(1.25).all()

    # The five outer slots are insufficient: each leaf must carry its own fold.
    checkpoint = torch.load(first / "best.pt", weights_only=True)
    checkpoint["folds"][0], checkpoint["folds"][1] = (
        checkpoint["folds"][1],
        checkpoint["folds"][0],
    )
    swapped_path = tmp_path / "swapped-folds.pt"
    torch.save(checkpoint, swapped_path)
    with pytest.raises(ValueError, match="fold identity mismatch"):
        run(inputs, tmp_path / "swapped-replay", swapped_path)
    assert not (tmp_path / "swapped-replay/predictions.csv").exists()
    checkpoint = torch.load(first / "best.pt", weights_only=True)
    checkpoint["folds"][0]["state_dict"]["lin.weight"] = torch.zeros((1, 1))
    bad_shape_path = tmp_path / "bad-shape.pt"
    torch.save(checkpoint, bad_shape_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        run(inputs, tmp_path / "bad-shape-replay", bad_shape_path)
    assert not (tmp_path / "bad-shape-replay/predictions.csv").exists()


def test_epoch_output_reports_deterioration_and_early_stop(
    inputs, tmp_path, monkeypatch, capsys
):
    settings = dict(inputs)
    settings["epochs"] = 5
    settings["early_stop_patience"] = 1
    validation_losses = iter([1.0, 1.0] * 5)
    monkeypatch.setattr(
        "src.models.trainer.validation_loss",
        lambda _model, _graph, _settings: next(validation_losses),
    )

    directory = run(settings, tmp_path / "early-stop")

    history = pd.read_csv(directory / "history.csv")
    assert history.groupby("fold").size().tolist() == [2] * 5
    assert (
        history.groupby("fold").early_stop_status.apply(list).tolist()
        == [["IMPROVED", "TRIGGERED"]] * 5
    )
    output = capsys.readouterr().out
    assert "deteriorations=1/1" in output
    assert "patience_remaining=0" in output
    assert "early_stop=TRIGGERED" in output


def test_graph_identity_batching_masking_and_missing_targets(inputs):
    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    graph = load_graph(Path(inputs["graphs_dir"]) / "fold-0/test.pt")
    graph = validate_graph(graph, split["folds"][0], "test", gauge)
    dataset = HourlyGraphDataset(graph)
    batch = Batch.from_data_list([dataset[0], dataset[3]])
    restored = batch.to_data_list()
    assert [x["raingauge"].node_position.tolist() for x in restored] == [
        list(range(5))
    ] * 2
    target_position = graph.station_ids.index("00001")
    assert not restored[1]["raingauge"].observed[target_position]
    original_x, original_y = batch["raingauge"].x.clone(), batch["raingauge"].y.clone()

    class InspectModel:
        def __call__(self, x, edges, weights):
            features = x["raingauge"]
            mask = batch["raingauge"].mask
            assert torch.equal(features[mask, :2], torch.zeros_like(features[mask, :2]))
            assert torch.equal(features[:, 2:], original_x[:, 2:])
            assert torch.equal(features[~mask], original_x[~mask])
            return {"raingauge": features[:, :1]}

    masked_forward(InspectModel(), batch, batch["raingauge"].mask)
    assert torch.equal(batch["raingauge"].x, original_x)
    torch.testing.assert_close(batch["raingauge"].y, original_y, equal_nan=True)


def test_graph_rejects_normalization_identity_and_label_corruption(inputs):
    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    path = Path(inputs["graphs_dir"]) / "fold-0/test.pt"
    graph = load_graph(path)
    graph["raingauge"].x[:, :, 0] += 1
    with pytest.raises(ValueError, match="physical rainfall_mm"):
        validate_graph(graph, split["folds"][0], "test", gauge)
    graph = load_graph(path)
    graph.station_ids[0] = "not-in-split"
    with pytest.raises(ValueError, match="prescribed split"):
        validate_graph(graph, split["folds"][0], "test", gauge)
    graph = load_graph(path)
    graph["raingauge"].y += 1
    with pytest.raises(ValueError, match="observed y"):
        validate_graph(graph, split["folds"][0], "test", gauge)
    graph = load_graph(path)
    graph["radar"].x = torch.zeros((1, 6, 1))
    with pytest.raises(ValueError, match="input_layers"):
        validate_graph(graph, split["folds"][0], "test", gauge)
    graph = load_graph(path)
    graph["raingauge"].x[0, 1, 2] += 1
    with pytest.raises(ValueError, match="constant across hours"):
        validate_graph(graph, split["folds"][0], "test", gauge)


def test_zscore_scaling_uses_training_statistics_and_leaves_the_rest_alone(inputs):
    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    fold = split["folds"][0]
    physical = read_fold(dict(inputs, input_rainfall_scale="physical_mm"), fold, gauge)
    settings = dict(inputs, input_rainfall_scale="zscore_train")
    scaled = read_fold(settings, fold, gauge)

    statistics = settings["input_scaling"][0]["raingauge"]
    train = physical["train"]["raingauge"]
    train_rain = train.x[:, :, 0][train.observed].double()
    assert statistics["mean_mm"] == pytest.approx(train_rain.mean().item())
    assert statistics["std_mm"] == pytest.approx(train_rain.std(correction=0).item())
    assert statistics["observed_values"] == int(train.observed.sum())

    # The training graph's observed rainfall is centred with unit spread.
    scaled_train = scaled["train"]["raingauge"]
    centred = scaled_train.x[:, :, 0][scaled_train.observed].double()
    assert centred.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert centred.std(correction=0).item() == pytest.approx(1.0, abs=1e-5)

    for role in ("train", "validation", "test"):
        before, after = physical[role]["raingauge"], scaled[role]["raingauge"]
        # Every role is scaled with the training numbers, never its own.
        torch.testing.assert_close(
            after.x[:, :, 0][after.observed],
            (before.x[:, :, 0][before.observed] - statistics["mean_mm"])
            / statistics["std_mm"],
        )
        # Unobserved rainfall keeps the zero placeholder; validity says it is absent.
        assert torch.equal(
            after.x[:, :, 0][~after.observed],
            torch.zeros_like(after.x[:, :, 0][~after.observed]),
        )
        # Validity, positional features and millimetre targets are untouched.
        assert torch.equal(after.x[:, :, 1:], before.x[:, :, 1:])
        torch.testing.assert_close(after.y, before.y, equal_nan=True)


def test_zscore_run_records_its_scaling_and_replays_from_a_checkpoint(inputs, tmp_path):
    settings = dict(inputs, input_rainfall_scale="zscore_train")
    first = run(settings, tmp_path / "first")
    recorded = yaml.safe_load((first / "config.yaml").read_text())

    assert recorded["input_rainfall_scale"] == "zscore_train"
    assert sorted(recorded["input_scaling"]) == [0, 1, 2, 3, 4]
    for fold in range(5):
        values = recorded["input_scaling"][fold]["raingauge"]
        assert values["std_mm"] > 0
        assert values["observed_values"] > 0
    assert "z-scored per node type" in (first / "report.md").read_text()

    # Predictions are still millimetres on the canonical target rows.
    original = read_predictions(first)
    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    validate_predictions(original, gauge, split)

    restored = run(settings, tmp_path / "restored", first / "best.pt")
    pd.testing.assert_frame_equal(
        original, read_predictions(restored), check_exact=True
    )
    # A checkpoint trained on scaled inputs cannot be replayed on raw millimetres.
    with pytest.raises(ValueError, match="input_rainfall_scale"):
        run(
            dict(inputs, input_rainfall_scale="physical_mm"),
            tmp_path / "mismatch",
            first / "best.pt",
        )


def test_scaling_rejects_an_unknown_setting_and_rainfall_without_spread(inputs):
    with pytest.raises(ValueError, match="physical_mm or zscore_train"):
        check_inputs(dict(inputs, input_rainfall_scale="log"))

    gauge, split, _ = load_inputs(inputs["gauge_file"], inputs["split_file"])
    fold = split["folds"][0]
    graph = read_fold(dict(inputs), fold, gauge)["train"]
    graph["raingauge"].x[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="no spread"):
        rainfall_scaling(graph, "fold 0 train")
    graph["raingauge"].observed[:] = False
    with pytest.raises(ValueError, match="no observed raingauge rainfall"):
        rainfall_scaling(graph, "fold 0 train")


def test_readonly_preflight_and_failed_run_keep_error(inputs, tmp_path):
    assert check_inputs(dict(inputs)) == {"folds": 5, "target_rows": 29}
    Path(inputs["graphs_dir"]).joinpath("fold-4/test.pt").unlink()
    with pytest.raises(FileNotFoundError, match="fold-4/test"):
        run(inputs, tmp_path / "failed")
    assert "Status: FAILED" in (tmp_path / "failed/report.md").read_text()
    assert not (tmp_path / "failed/predictions.csv").exists()
