"""Acceptance tests for the researcher experiment TUI and its workspace service."""

import asyncio
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from textual.widgets import Input, TextArea

from src.experiments.chart import loss_chart
from src.experiments.tui import ExperimentApp
from src.experiments.workspace import (
    STAGE_BY_KEY,
    ExperimentError,
    PipelineRunner,
    configuration_locked,
    create_experiment,
    discover_experiments,
    ensure_fresh_outputs,
    list_config_files,
    load_experiment,
    load_history,
    read_config_text,
    read_status,
    save_config_text,
    selected_stages,
    stage_commands,
    write_effective_configs,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    configs = {
        "ingestion/sydney.yaml": {
            "region": "sydney",
            "output_dir": "database/raw",
        },
        "canonical/standard.yaml": {
            "raw_dir": "will/be/replaced",
            "output_dir": "will/be/replaced",
        },
        "splits/sydney_5fold.yaml": {
            "stations_file": "will/be/replaced",
            "output": "config/splits/custom-name.json",
        },
        "graphs/gauge_baseline.yaml": {
            "data_dir": "will/be/replaced",
            "split_file": "will/be/replaced",
            "output_dir": "will/be/replaced",
        },
        "models/hgnn.yaml": {
            "method": "hgnn",
            "gauge_file": "will/be/replaced",
            "stations_file": "will/be/replaced",
            "split_file": "will/be/replaced",
            "seeds_file": "will/be/replaced",
            "graphs_dir": "will/be/replaced",
            "dropout": 0.15,
        },
        "seeds.yaml": {"split_seed": 123, "model_seed": 123},
        "extra/ablation.yaml": {"researcher_setting": True},
    }
    for relative, payload in configs.items():
        _write_yaml(root / "config" / relative, payload)
    (root / "config/splits/sydney_5fold.json").write_text(
        json.dumps({"folds": []}), encoding="utf-8"
    )
    return root


def test_create_clone_discover_and_refuse_reuse(project: Path) -> None:
    original = create_experiment("baseline", project_root=project)
    assert original.root == project / "experiments/baseline"
    assert (original.root / "report.md").read_text().endswith("Status: READY\n")
    assert "extra/ablation.yaml" in list_config_files(original)
    assert load_experiment(original.metadata_path) == original
    assert discover_experiments(project / "experiments") == [original]

    save_config_text(original, "models/hgnn.yaml", "dropout: 0.3\n")
    clone = create_experiment(
        "dropout study",
        project / "custom-location",
        project_root=project,
        template=original.root,
    )
    assert clone.root == project / "custom-location/dropout study"
    assert clone.config_path("model").read_text() == "dropout: 0.3\n"
    assert not (clone.root / "model").exists()
    assert read_status(clone)["state"] == "READY"

    with pytest.raises(FileExistsError, match="already exists"):
        create_experiment("baseline", project_root=project)
    with pytest.raises(ExperimentError, match="without slashes"):
        create_experiment("../unsafe", project_root=project)


def test_config_validation_and_lock_preserve_the_snapshot(project: Path) -> None:
    experiment = create_experiment("editable", project_root=project)
    path = experiment.config_path("model")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ExperimentError, match="Invalid YAML"):
        save_config_text(experiment, "models/hgnn.yaml", "broken: [")
    assert path.read_text(encoding="utf-8") == before

    save_config_text(experiment, "models/hgnn.yaml", "dropout: 0.25\n")
    assert yaml.safe_load(path.read_text()) == {"dropout": 0.25}
    status = read_status(experiment)
    status["state"] = "DONE"
    experiment.status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ExperimentError, match="locked"):
        save_config_text(experiment, "models/hgnn.yaml", "dropout: 0.4\n")


def test_workspace_validation_reports_bad_metadata_and_paths(project: Path) -> None:
    with pytest.raises(ExperimentError, match="required"):
        create_experiment("   ", project_root=project)
    with pytest.raises(ExperimentError, match="control"):
        create_experiment("bad\nname", project_root=project)

    experiment = create_experiment("validation", project_root=project)
    with pytest.raises(ExperimentError, match="escapes"):
        read_config_text(experiment, "../outside.yaml")
    with pytest.raises(ExperimentError, match="Unknown editable"):
        read_config_text(experiment, "not-there.yaml")
    with pytest.raises(ExperimentError, match="cannot be empty"):
        save_config_text(experiment, "models/hgnn.yaml", "")
    with pytest.raises(ExperimentError, match="Unknown pipeline"):
        selected_stages("unknown", "train")

    experiment.status_path.unlink()
    assert not configuration_locked(experiment)
    experiment.status_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ExperimentError, match="Invalid status"):
        read_status(experiment)

    invalid = project / "experiments/invalid"
    invalid.mkdir()
    (invalid / "experiment.yaml").write_text("format: wrong\n", encoding="utf-8")
    assert discover_experiments(project / "experiments") == [experiment]
    with pytest.raises(ExperimentError, match="Unsupported"):
        load_experiment(invalid)


def test_effective_configs_connect_every_pipeline_output(project: Path) -> None:
    experiment = create_experiment("paths", project_root=project)
    paths = write_effective_configs(experiment)
    effective = {
        role: yaml.safe_load(path.read_text(encoding="utf-8"))
        for role, path in paths.items()
    }
    assert effective["canonical"]["raw_dir"] == str(project / "database/raw")
    assert effective["canonical"]["output_dir"] == str(experiment.root / "canonical")
    assert effective["split"]["stations_file"] == str(
        experiment.root / "canonical/stations.csv"
    )
    assert effective["split"]["output"] == str(
        experiment.root / "split/custom-name.json"
    )
    assert effective["graphs"]["output_dir"] == str(experiment.root / "graphs")
    assert effective["model"]["graphs_dir"] == str(experiment.root / "graphs")
    assert effective["model"]["seeds_file"] == str(
        experiment.root / "effective-config/seeds.yaml"
    )

    commands = stage_commands(
        experiment, "canonical", "train", executable="python-test"
    )
    assert [stage.key for stage, _ in commands] == [
        "canonical",
        "split",
        "graphs",
        "train",
    ]
    assert commands[-1][1][-2:] == ["--output-dir", str(experiment.root / "model")]
    with pytest.raises(ExperimentError, match="after the start"):
        stage_commands(experiment, "train", "canonical")

    (experiment.root / "graphs").mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        ensure_fresh_outputs(experiment, selected_stages("graphs", "graphs"))


def test_pipeline_streams_output_and_records_status(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = create_experiment("runner", project_root=project)
    command = [
        sys.executable,
        "-u",
        "-c",
        "print('epoch 1 train=0.4 validation=0.5')",
    ]
    monkeypatch.setattr(
        "src.experiments.workspace.stage_commands",
        lambda *_args, **_kwargs: [(STAGE_BY_KEY["canonical"], command)],
    )
    output: list[str] = []
    runner = PipelineRunner(experiment)
    asyncio.run(runner.run("canonical", "canonical", on_output=output.append))
    status = read_status(experiment)
    assert status["state"] == "DONE"
    assert status["stages"]["canonical"] == "DONE"
    assert any("epoch 1" in line for line in output)
    assert "epoch 1" in (experiment.root / "log.txt").read_text()
    assert "Status: DONE" in (experiment.root / "report.md").read_text()


def test_failed_pipeline_keeps_a_visible_failure(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = create_experiment("failed", project_root=project)
    command = [sys.executable, "-u", "-c", "raise SystemExit(7)"]
    monkeypatch.setattr(
        "src.experiments.workspace.stage_commands",
        lambda *_args, **_kwargs: [(STAGE_BY_KEY["canonical"], command)],
    )
    with pytest.raises(ExperimentError, match="status 7"):
        asyncio.run(PipelineRunner(experiment).run("canonical", "canonical"))
    status = read_status(experiment)
    assert status["state"] == "FAILED"
    assert status["stages"]["canonical"] == "FAILED"
    assert "Status: FAILED" in (experiment.root / "report.md").read_text()


def test_pipeline_can_be_stopped_without_losing_the_failure_record(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = create_experiment("stopped", project_root=project)
    partial_report = experiment.root / "model/report.md"
    script = (
        "import time; from pathlib import Path; "
        f"p=Path({str(partial_report)!r}); p.parent.mkdir(); "
        "p.write_text('# Model run\\n\\nStatus: RUNNING\\n'); "
        "print('started', flush=True); time.sleep(30)"
    )
    command = [
        sys.executable,
        "-u",
        "-c",
        script,
    ]
    monkeypatch.setattr(
        "src.experiments.workspace.stage_commands",
        lambda *_args, **_kwargs: [(STAGE_BY_KEY["train"], command)],
    )
    runner = PipelineRunner(experiment)

    async def exercise() -> None:
        started = asyncio.Event()

        async def output(line: str) -> None:
            if line == "started":
                started.set()

        task = asyncio.create_task(runner.run("train", "train", on_output=output))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert runner.running
        with pytest.raises(ExperimentError, match="already running"):
            await runner.run("train", "train")
        await runner.stop()
        await task
        await runner.stop()  # stopping an idle runner is safe

    asyncio.run(exercise())
    status = read_status(experiment)
    assert status["state"] == "STOPPED"
    assert status["stages"]["train"] == "STOPPED"
    assert "partial artifacts were retained" in status["message"]
    assert "Status: STOPPED" in partial_report.read_text()
    assert "Status: RUNNING" not in partial_report.read_text()


def test_history_load_and_loss_chart(project: Path) -> None:
    experiment = create_experiment("results", project_root=project)
    model = experiment.root / "model"
    model.mkdir()
    (model / "history.csv").write_text(
        "fold,epoch,train_loss,val_loss\n0,1,1.0,1.4\n0,2,0.7,1.1\n1,1,2.0,2.2\n",
        encoding="utf-8",
    )
    chart = loss_chart(load_history(experiment), fold=0, width=72, height=18)
    assert "Fold 0" in chart.plain
    assert "Epoch 2" in chart.plain
    assert "Train" in chart.plain
    assert "Validation" in chart.plain
    assert "Loss" in chart.plain
    assert "Epoch" in chart.plain
    assert "●" in chart.plain
    assert {str(span.style) for span in chart.spans} >= {
        "bold cyan",
        "bold magenta",
    }


def test_textual_app_supports_create_clone_edit_and_open(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = create_experiment("visible", project_root=project)
    app = ExperimentApp(project_root=project)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.current == experiment
            assert app.query_one("#config-editor").text
            assert "visible" in str(app.query_one("#workspace-header").render())

            app.query_one("#config-file").value = "models/hgnn.yaml"
            await pilot.pause()
            editor = app.query_one("#config-editor", TextArea)
            editor.load_text("method: hgnn\ndropout: 0.2\n")
            app.save_config()
            assert "dropout: 0.2" in experiment.config_path("model").read_text()
            app.reload_config()
            app.refresh_experiments()

            app.query_one("#open-path", Input).value = ""
            app.open_experiment()
            app.query_one("#open-path", Input).value = str(experiment.root)
            app.open_experiment()
            app._open(project / "missing")

            app.action_new_experiment()
            await pilot.pause()
            dialog = app.screen
            dialog.query_one("#new-name", Input).value = "created"
            dialog.query_one("#new-location", Input).value = str(
                project / "experiments"
            )
            dialog.create()
            await pilot.pause()
            assert app.current is not None and app.current.name == "created"

            app.clone_experiment()
            await pilot.pause()
            clone_dialog = app.screen
            clone_dialog.query_one("#new-name", Input).value = "cloned"
            clone_dialog.create()
            await pilot.pause()
            assert app.current is not None and app.current.name == "cloned"

            app.new_experiment()
            await pilot.pause()
            blank_dialog = app.screen
            blank_dialog.create()
            blank_dialog.cancel()
            await pilot.pause()

            current = app.current
            app.current = None
            app.clone_experiment()
            app.action_save_config()
            app.action_run_pipeline()
            app._refresh_live_views()
            app._load_config("models/hgnn.yaml")
            app.current = current
            assert current is not None
            app._set_current(current)

            async def fake_run(
                runner: PipelineRunner,
                _start: str,
                _end: str,
                *,
                on_output=None,
                on_status=None,
            ) -> None:
                status = read_status(runner.experiment)
                status["state"] = "DONE"
                runner.experiment.status_path.write_text(json.dumps(status))
                if on_output:
                    on_output("live output")
                if on_status:
                    on_status("canonical", "DONE")

            monkeypatch.setattr(PipelineRunner, "run", fake_run)
            await app._run_pipeline("canonical", "canonical")
            await app.stop_pipeline()

    asyncio.run(exercise())


def test_real_offline_pipeline_runs_from_preprocessing_to_training(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    make_fixture = runpy.run_path(
        str(repository / "tests/data/pipeline/make_fixture.py")
    )["make_fixture"]
    fixture = make_fixture(tmp_path / "fixture")
    experiment = create_experiment(
        "offline-pipeline",
        tmp_path / "experiments",
        project_root=repository,
    )

    ingestion = {
        "region": "synthetic-test",
        "output_dir": str(fixture / "raw"),
    }
    graph = yaml.safe_load((fixture / "graph.yaml").read_text())
    graph["input_layers"] = ["raingauge"]
    model = yaml.safe_load((fixture / "model.yaml").read_text())
    model["input_layers"] = ["raingauge"]
    replacements = {
        "ingestion/sydney.yaml": ingestion,
        "canonical/standard.yaml": yaml.safe_load(
            (fixture / "canonical.yaml").read_text()
        ),
        "splits/sydney_5fold.yaml": yaml.safe_load(
            (fixture / "split.yaml").read_text()
        ),
        "graphs/gauge_baseline.yaml": graph,
        "models/hgnn.yaml": model,
    }
    for relative, payload in replacements.items():
        save_config_text(
            experiment,
            relative,
            yaml.safe_dump(payload, sort_keys=False),
        )

    asyncio.run(PipelineRunner(experiment).run("canonical", "train"))

    predictions = (experiment.root / "model/predictions.csv").read_text().splitlines()
    assert len(predictions) == 865  # header plus 864 exact target/method rows
    assert (experiment.root / "model/history.csv").is_file()
    log = (experiment.root / "log.txt").read_text()
    assert "fold=0 epoch=1/2" in log
    assert "deteriorations=" in log
    assert "patience_remaining=" in log
    assert "early_stop=" in log
    assert "Status: DONE" in (experiment.root / "report.md").read_text()
    assert read_status(experiment)["stages"] == {
        "fetch": "PENDING",
        "canonical": "DONE",
        "split": "DONE",
        "graphs": "DONE",
        "train": "DONE",
    }
