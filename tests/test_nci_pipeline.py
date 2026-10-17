"""The config helper the Gadi pipeline jobs call (scripts/nci/pipeline_config.py)."""

import importlib.util
from datetime import date
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "pipeline_config", Path("scripts/nci/pipeline_config.py")
)
pipeline_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline_config)


def grid_config(tmp_path, root=None):
    root = root or tmp_path / "archive"
    Path(root).mkdir(exist_ok=True)
    config = {
        "raw_dir": str(tmp_path / "raw"),
        "output_dir": str(tmp_path / "standard"),
        "start": date(2021, 1, 1),
        "end": date(2022, 12, 31),
        "layers": {
            "radar": {"root": str(root), "subsample_stride": 10},
            "satellite": {"root": str(root), "min_images_per_hour": 3},
        },
    }
    path = tmp_path / "grids.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_the_real_grid_config_names_its_outputs_and_both_archives():
    config = pipeline_config.read("config/canonical/grids.yaml")
    assert config["output_dir"] == "database/standard"
    assert set(config["layers"]) == {"radar", "satellite"}
    assert config["layers"]["satellite"]["min_images_per_hour"] == 3


def test_a_clean_grid_config_passes(tmp_path):
    assert pipeline_config.grid_problems(grid_config(tmp_path)) == []


def test_existing_grid_outputs_stop_the_job_before_the_archive_is_read(tmp_path):
    path = grid_config(tmp_path)
    (tmp_path / "standard").mkdir()
    (tmp_path / "standard/satellite_hourly.csv").write_text("x")
    problems = pipeline_config.grid_problems(path)
    assert problems == [f"{tmp_path / 'standard/satellite_hourly.csv'} already exists"]
    assert pipeline_config.main(["check-grids", str(path)]) == 1


def test_an_unreadable_archive_names_the_layer_and_the_mount(tmp_path):
    path = grid_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["layers"]["radar"]["root"] = str(tmp_path / "not-mounted")
    path.write_text(yaml.safe_dump(config))
    (problem,) = pipeline_config.grid_problems(path)
    assert problem.startswith("radar archive root")
    assert "storage directive" in problem


def test_a_trial_copies_the_config_with_a_shorter_period_and_its_own_folder(tmp_path):
    path = grid_config(tmp_path)
    trial = pipeline_config.write_trial(path, "2022-01-01:2022-01-07", tmp_path / "t")
    config = yaml.safe_load(trial.read_text())
    assert (config["start"], config["end"]) == (date(2022, 1, 1), date(2022, 1, 7))
    assert config["output_dir"] == str(tmp_path / "t")
    assert config["layers"] == yaml.safe_load(path.read_text())["layers"]
    with pytest.raises(FileExistsError):
        pipeline_config.write_trial(path, "2022-01-01:2022-01-07", tmp_path / "t")


@pytest.mark.parametrize(
    "period", ["2022-01-07:2022-01-01", "2022-01-01", "soon:later"]
)
def test_a_bad_trial_period_is_refused(tmp_path, period, capsys):
    path = grid_config(tmp_path)
    assert pipeline_config.main(["trial", str(path), period, str(tmp_path / "t")]) == 1
    assert "BLOCKED" in capsys.readouterr().err
    assert not (tmp_path / "t").exists()


def test_get_prints_a_key_and_refuses_a_missing_one(tmp_path, capsys):
    path = grid_config(tmp_path)
    assert pipeline_config.main(["get", str(path), "output_dir"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "standard")
    assert pipeline_config.main(["get", str(path), "graphs_dir"]) == 1


@pytest.mark.parametrize("name", ["g", "gr", "grs"])
def test_each_d09_run_has_matching_graph_and_model_configs(name):
    graph = pipeline_config.read(f"config/graphs/d09_round_robin_{name}.yaml")
    model = pipeline_config.read(f"config/models/d09_round_robin_{name}.yaml")
    assert graph["split_file"] == model["split_file"]
    assert graph["split_file"] == "config/splits/sydney_5fold_round_robin.json"
    assert graph["output_dir"] == model["graphs_dir"]
    assert graph["input_layers"] == model["input_layers"]
    # Every run's timeline covers both grid layers, so all are scored on the same hours.
    inputs = set(graph["input_layers"]) - {"raingauge"}
    assert inputs | set(graph["timeline_layers"]) == {"radar", "satellite"}
    assert model["device"] == "cuda"
