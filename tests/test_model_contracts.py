"""Executable Component 06 target, leakage and scoring acceptance checks."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import predict_at_targets, predict_baseline
from src.models.contracts import load_inputs, target_rows, validate_predictions
from src.models.evaluate import score_predictions


@pytest.fixture
def inputs():
    ids = ["001", "002", "003", "004", "005"]
    times = pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")
    gauges = pd.DataFrame(
        [
            {
                "station_id": sid,
                "timestamp": timestamp,
                "rainfall_mm": float(index + hour),
            }
            for index, sid in enumerate(ids)
            for hour, timestamp in enumerate(times)
        ]
    )
    stations = pd.DataFrame(
        {
            "station_id": ids,
            "latitude": [-33.8] * 5,
            "longitude": [151 + i * 0.01 for i in range(5)],
            "use_in_study": [True] * 5,
        }
    )
    split = {
        "folds": [
            {
                "fold_index": i,
                "ml": {
                    "test": [sid],
                    "validation": [ids[(i + 1) % 5]],
                    "train": [s for s in ids if s not in (sid, ids[(i + 1) % 5])],
                },
            }
            for i, sid in enumerate(ids)
        ]
    }
    return gauges, stations, split


@pytest.mark.parametrize("method", ["network_mean", "idw"])
def test_baselines_share_all_observed_targets(inputs, method):
    gauges, stations, split = inputs
    # Missing hours are absent targets, and real dry hours remain targets.
    gauges = gauges.drop(index=1)
    predictions = predict_baseline(method, gauges, stations, split)
    validate_predictions(predictions, gauges, split, method=method)
    assert len(predictions) == len(gauges)
    assert predictions.actual_mm.eq(0).any()


@pytest.mark.parametrize("method", ["network_mean", "idw"])
def test_deleting_target_observations_cannot_change_predictions(inputs, method):
    gauges, stations, split = inputs
    fixed_targets = target_rows(gauges, split)
    # One fold's labels are deleted from every gauge the method can read, while the
    # independent evaluation target table retains its labels.
    targets = fixed_targets.loc[fixed_targets.fold.eq(0)]
    expected = predict_at_targets(method, gauges, stations, split, targets)
    visible_only = gauges.loc[~gauges.station_id.isin(split["folds"][0]["ml"]["test"])]
    actual = predict_at_targets(method, visible_only, stations, split, targets)
    pd.testing.assert_frame_equal(actual, expected)
    mutated = gauges.copy()
    mutated.loc[mutated.station_id.eq("001"), "rainfall_mm"] = 10000
    # The hidden gauge's own readings never reach the baselines. Validation gauges
    # do (D08.2): test_baselines_read_the_validation_gauges covers that.
    pd.testing.assert_frame_equal(
        predict_at_targets(method, mutated, stations, split, targets), expected
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "wrong_truth",
        "negative",
        "nan",
        "unsorted",
        "naive",
    ],
)
def test_checker_rejects_invalid_result(inputs, change):
    gauges, stations, split = inputs
    predictions = predict_baseline("network_mean", gauges, stations, split)
    if change == "missing":
        predictions = predictions.iloc[1:]
    elif change == "extra":
        predictions.loc[0, "station_id"] = "unknown"
        predictions = predictions.sort_values(
            ["method", "fold", "station_id", "timestamp"]
        )
    elif change == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]], ignore_index=True)
    elif change == "wrong_truth":
        predictions.loc[0, "actual_mm"] += 1
    elif change == "negative":
        predictions.loc[0, "predicted_mm"] = -1
    elif change == "nan":
        predictions.loc[0, "predicted_mm"] = np.nan
    elif change == "unsorted":
        predictions = predictions.iloc[::-1]
    elif change == "naive":
        predictions["timestamp"] = predictions.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError):
        validate_predictions(predictions, gauges, split)


def test_idw_collocated_station_uses_only_the_observed_colocated_gauge(inputs):
    gauges, stations, split = inputs
    stations.loc[stations.station_id.eq("003"), "longitude"] = 151.0
    result = predict_baseline("idw", gauges, stations, split)
    assert result.loc[result.fold.eq(0), "predicted_mm"].tolist() == [2, 3, 4]


def test_missing_gauges_fail_instead_of_fabricating_rainfall(inputs):
    gauges, stations, split = inputs
    targets = target_rows(gauges, split)
    roles = split["folds"][0]["ml"]
    visible_ids = roles["train"] + roles["validation"]
    gauges = gauges.loc[
        ~(
            gauges.station_id.isin(visible_ids)
            & gauges.timestamp.eq(gauges.timestamp.min())
        )
    ]
    with pytest.raises(ValueError, match="no observed visible gauge"):
        predict_at_targets("network_mean", gauges, stations, split, targets)


def test_network_mean_averages_training_and_validation_gauges(inputs):
    gauges, stations, split = inputs
    result = predict_baseline("network_mean", gauges, stations, split)
    # Fold 0 hides 001. Visible: 002 (validation) and 003, 004, 005 (train), which
    # read 1, 2, 3, 4 mm in the first hour and one more each hour after.
    assert result.loc[result.fold.eq(0), "predicted_mm"].tolist() == [2.5, 3.5, 4.5]


@pytest.mark.parametrize("method", ["network_mean", "idw"])
def test_baselines_read_the_validation_gauges(inputs, method):
    gauges, stations, split = inputs
    validation_id = split["folds"][0]["ml"]["validation"][0]
    wetter = gauges.copy()
    wetter.loc[wetter.station_id.eq(validation_id), "rainfall_mm"] += 10.0

    before = predict_baseline(method, gauges, stations, split)
    after = predict_baseline(method, wetter, stations, split)

    # More rain at fold 0's validation gauge must raise fold 0's predictions.
    assert (
        after.loc[after.fold.eq(0), "predicted_mm"].to_numpy()
        > before.loc[before.fold.eq(0), "predicted_mm"].to_numpy()
    ).all()


def test_scoring_uses_identical_rows_and_known_errors(inputs):
    gauges, stations, split = inputs
    a = predict_baseline("network_mean", gauges, stations, split)
    a["predicted_mm"] = a.actual_mm + 2
    b = a.assign(method="idw")
    table = pd.concat([b, a], ignore_index=True)
    scores = score_predictions(table)
    assert np.allclose(scores.rmse_mm, 2)
    assert np.allclose(scores.mae_mm, 2)
    assert np.allclose(scores.correlation, 1)
    with pytest.raises(ValueError, match="identical target rows"):
        score_predictions(table.iloc[1:])


def test_constant_prediction_correlation_is_explicitly_undefined(inputs):
    gauges, stations, split = inputs
    table = predict_baseline("network_mean", gauges, stations, split)
    table["predicted_mm"] = 0.0
    assert score_predictions(table).correlation.isna().all()


def test_input_loader_preserves_ids_and_rejects_role_leakage(inputs, tmp_path):
    gauges, stations, split = inputs
    gauges.to_csv(tmp_path / "gauge.csv", index=False)
    stations.to_csv(tmp_path / "stations.csv", index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    loaded, _, _ = load_inputs(
        tmp_path / "gauge.csv", split_path, tmp_path / "stations.csv"
    )
    assert loaded.station_id.iloc[0] == "001"
    invalid = copy.deepcopy(split)
    invalid["folds"][0]["ml"]["train"].append("001")
    split_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="overlapping"):
        load_inputs(tmp_path / "gauge.csv", split_path)


def test_input_loader_reports_upstream_blocker(tmp_path):
    with pytest.raises(FileNotFoundError, match="required upstream input"):
        load_inputs(tmp_path / "missing.csv", tmp_path / "split.json")


def test_scorer_refuses_truncated_fold_export(inputs):
    gauges, stations, split = inputs
    table = predict_baseline("network_mean", gauges, stations, split)
    with pytest.raises(ValueError, match="all five folds"):
        score_predictions(table.loc[table.fold.eq(0)])


def test_checker_refuses_a_whole_missing_station(inputs):
    gauges, _, split = inputs
    with pytest.raises(ValueError, match="no observations"):
        target_rows(gauges.loc[~gauges.station_id.eq("001")], split)


@pytest.mark.parametrize("fold", [5, -1, 0.5])
def test_baseline_rejects_unknown_target_fold(inputs, fold):
    gauges, stations, split = inputs
    targets = target_rows(gauges, split)
    targets["fold"] = targets.fold.astype(float) if fold == 0.5 else targets.fold
    targets.loc[0, "fold"] = fold
    with pytest.raises(ValueError, match="integer folds"):
        predict_at_targets("network_mean", gauges, stations, split, targets)
