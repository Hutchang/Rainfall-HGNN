"""Experiment workspaces and pipeline orchestration used by the TUI.

The terminal widgets deliberately do not know how component paths fit together. This
module owns that small piece of orchestration so it can be tested without a terminal and
later replaced by the complete D07 experiment runner.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_FORMAT = "experiment-tui-v1"
CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
DEFAULT_CONFIGS = {
    "ingestion": "ingestion/sydney.yaml",
    "canonical": "canonical/standard.yaml",
    "split": "splits/sydney_5fold.yaml",
    "graphs": "graphs/gauge_baseline.yaml",
    "model": "models/hgnn.yaml",
    "seeds": "seeds.yaml",
}


class ExperimentError(ValueError):
    """An experiment request is unsafe or incomplete."""


@dataclass(frozen=True)
class Stage:
    """One runnable component in pipeline order."""

    key: str
    label: str
    module: str


STAGES = (
    Stage("fetch", "01 · Fetch gauge data", "src.ingestion.gauge.fetch"),
    Stage("canonical", "02 · Pre-process tables", "src.canonical.gauge.build"),
    Stage("split", "03 · Create folds", "src.study.split"),
    Stage("graphs", "04 · Build graphs", "src.graph.builder_new"),
    Stage("train", "06 · Train and evaluate", "src.models.run"),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}

OutputCallback = Callable[[str], Awaitable[None] | None]
StatusCallback = Callable[[str, str], Awaitable[None] | None]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ExperimentError("Experiment name is required")
    if name in {".", ".."} or any(char in name for char in ("/", "\\", "\0")):
        raise ExperimentError("Experiment name must be one folder name without slashes")
    if any(ord(char) < 32 for char in name):
        raise ExperimentError("Experiment name cannot contain control characters")
    return name


def _project_root(project_root: Path | str | None = None) -> Path:
    return Path(project_root or Path(__file__).parents[2]).resolve()


@dataclass(frozen=True)
class Experiment:
    """A saved experiment and its immutable identity."""

    root: Path
    name: str
    project_root: Path
    created_utc: str
    configs: dict[str, str]

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def metadata_path(self) -> Path:
        return self.root / "experiment.yaml"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def config_path(self, role: str) -> Path:
        try:
            relative = self.configs[role]
        except KeyError as error:
            raise ExperimentError(f"Experiment has no {role!r} config") from error
        path = (self.config_dir / relative).resolve()
        if not path.is_relative_to(self.config_dir.resolve()):
            raise ExperimentError(f"Config path escapes experiment/config: {relative}")
        return path


def _metadata(name: str, root: Path, project_root: Path) -> dict[str, Any]:
    return {
        "format": EXPERIMENT_FORMAT,
        "name": name,
        "created_utc": _now(),
        "project_root": str(project_root),
        "configs": dict(DEFAULT_CONFIGS),
        "output_root": str(root),
    }


def initial_status() -> dict[str, Any]:
    """Return the status stored for a newly-created experiment."""
    return {
        "state": "READY",
        "updated_utc": _now(),
        "current_stage": None,
        "stages": {stage.key: "PENDING" for stage in STAGES},
        "message": "Configuration can be edited before the first pipeline run.",
    }


def create_experiment(
    name: str,
    base_dir: Path | str | None = None,
    *,
    project_root: Path | str | None = None,
    template: Path | str | None = None,
) -> Experiment:
    """Create a new, non-overwriting experiment with a complete config snapshot.

    ``template`` is another experiment directory. Its editable config snapshot is
    copied, but no outputs, logs, effective configs, or statuses are inherited.
    """
    name = _validate_name(name)
    project = _project_root(project_root)
    base = Path(base_dir or project / "experiments").expanduser().resolve()
    destination = base / name
    if destination.exists():
        raise FileExistsError(
            f"Experiment folder already exists: {destination}; choose another name"
        )

    if template is None:
        config_source = project / "config"
    else:
        config_source = load_experiment(template).config_dir
    if not config_source.is_dir():
        raise ExperimentError(f"Config source does not exist: {config_source}")

    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".creating-experiment-", dir=base))
    try:
        shutil.copytree(config_source, temporary / "config")
        metadata = _metadata(name, destination, project)
        _write_yaml(temporary / "experiment.yaml", metadata)
        _write_json(temporary / "status.json", initial_status())
        (temporary / "report.md").write_text(
            f"# Experiment: {name}\n\nStatus: READY\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_experiment(destination)


def load_experiment(path: Path | str) -> Experiment:
    """Load and validate an experiment directory or its metadata file."""
    root = Path(path).expanduser().resolve()
    if root.name == "experiment.yaml":
        root = root.parent
    metadata_path = root / "experiment.yaml"
    if not metadata_path.is_file():
        raise ExperimentError(f"No experiment.yaml found in {root}")
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != EXPERIMENT_FORMAT:
        raise ExperimentError(f"Unsupported experiment format in {metadata_path}")
    configs = payload.get("configs")
    if not isinstance(configs, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in configs.items()
    ):
        raise ExperimentError(f"Invalid config map in {metadata_path}")
    missing_roles = sorted(set(DEFAULT_CONFIGS) - set(configs))
    if missing_roles:
        raise ExperimentError(
            f"{metadata_path} is missing config roles: {', '.join(missing_roles)}"
        )
    experiment = Experiment(
        root=root,
        name=str(payload.get("name", root.name)),
        project_root=Path(payload.get("project_root", _project_root())).resolve(),
        created_utc=str(payload.get("created_utc", "UNKNOWN")),
        configs=dict(configs),
    )
    for role in DEFAULT_CONFIGS:
        path = experiment.config_path(role)
        if not path.is_file():
            raise ExperimentError(f"Missing {role} config: {path}")
    return experiment


def discover_experiments(base_dir: Path | str) -> list[Experiment]:
    """Return valid experiment folders newest first; ignore unrelated folders."""
    base = Path(base_dir).expanduser()
    if not base.is_dir():
        return []
    experiments = []
    for metadata in base.glob("*/experiment.yaml"):
        try:
            experiments.append(load_experiment(metadata))
        except ExperimentError, OSError, yaml.YAMLError:
            continue
    return sorted(
        experiments,
        key=lambda experiment: experiment.created_utc,
        reverse=True,
    )


def list_config_files(experiment: Experiment) -> list[str]:
    """List every editable YAML and JSON file in the experiment snapshot."""
    return [
        str(path.relative_to(experiment.config_dir))
        for path in sorted(experiment.config_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES
    ]


def read_status(experiment: Experiment) -> dict[str, Any]:
    if not experiment.status_path.is_file():
        return initial_status()
    payload = json.loads(experiment.status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentError(f"Invalid status file: {experiment.status_path}")
    return payload


def configuration_locked(experiment: Experiment) -> bool:
    """Return whether a run has begun and the config snapshot is now evidence."""
    return read_status(experiment).get("state") != "READY"


def read_config_text(experiment: Experiment, relative: str) -> str:
    path = _safe_config_path(experiment, relative)
    return path.read_text(encoding="utf-8")


def _safe_config_path(experiment: Experiment, relative: str) -> Path:
    path = (experiment.config_dir / relative).resolve()
    if not path.is_relative_to(experiment.config_dir.resolve()):
        raise ExperimentError(f"Config path escapes experiment/config: {relative}")
    if not path.is_file() or path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ExperimentError(f"Unknown editable config file: {relative}")
    return path


def save_config_text(experiment: Experiment, relative: str, text: str) -> None:
    """Validate and atomically save one editable config file."""
    if configuration_locked(experiment):
        raise ExperimentError(
            "This experiment has started and its config is locked. Clone it to make "
            "an ablation or corrected run."
        )
    path = _safe_config_path(experiment, relative)
    try:
        parsed = (
            json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ExperimentError(f"Invalid {path.suffix[1:].upper()}: {error}") from error
    if parsed is None:
        raise ExperimentError("A config file cannot be empty")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentError(f"Config must contain a mapping: {path}")
    return payload


def _configured_path(project: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def write_effective_configs(experiment: Experiment) -> dict[str, Path]:
    """Write configs whose inter-component paths stay inside this run folder.

    The editable copies keep the researcher's exact text. Effective copies make the
    path changes imposed by the per-run output contract explicit.
    """
    runtime = experiment.root / "effective-config"
    runtime.mkdir(parents=True, exist_ok=True)

    ingestion = _load_mapping(experiment.config_path("ingestion"))
    raw_dir = _configured_path(
        experiment.project_root,
        ingestion.get("output_dir", "database/raw"),
        "ingestion.output_dir",
    )
    ingestion["output_dir"] = str(raw_dir)

    canonical = _load_mapping(experiment.config_path("canonical"))
    canonical["raw_dir"] = str(raw_dir)
    canonical["output_dir"] = str(experiment.root / "canonical")
    canonical["ingestion_config"] = str(runtime / "ingestion.yaml")

    split = _load_mapping(experiment.config_path("split"))
    split["stations_file"] = str(experiment.root / "canonical" / "stations.csv")
    split_name = Path(str(split.get("output", "split.json"))).name
    split_output = experiment.root / "split" / split_name
    split["output"] = str(split_output)

    graphs = _load_mapping(experiment.config_path("graphs"))
    graphs["data_dir"] = str(experiment.root / "canonical")
    graphs["split_file"] = str(split_output)
    graphs["output_dir"] = str(experiment.root / "graphs")

    model = _load_mapping(experiment.config_path("model"))
    model["gauge_file"] = str(experiment.root / "canonical" / "gauge_hourly.csv")
    model["stations_file"] = str(experiment.root / "canonical" / "stations.csv")
    model["split_file"] = str(split_output)
    model["seeds_file"] = str(runtime / "seeds.yaml")
    model["graphs_dir"] = str(experiment.root / "graphs")

    payloads = {
        "ingestion": ingestion,
        "canonical": canonical,
        "split": split,
        "graphs": graphs,
        "model": model,
    }
    paths = {role: runtime / f"{role}.yaml" for role in payloads}
    for role, payload in payloads.items():
        _write_yaml(paths[role], payload)
    shutil.copy2(experiment.config_path("seeds"), runtime / "seeds.yaml")
    paths["seeds"] = runtime / "seeds.yaml"
    return paths


def selected_stages(start: str, end: str) -> tuple[Stage, ...]:
    """Validate a pipeline interval and return its stages in order."""
    if start not in STAGE_BY_KEY or end not in STAGE_BY_KEY:
        raise ExperimentError("Unknown pipeline start or end stage")
    keys = [stage.key for stage in STAGES]
    first, last = keys.index(start), keys.index(end)
    if first > last:
        raise ExperimentError("The end stage must come after the start stage")
    return STAGES[first : last + 1]


def stage_commands(
    experiment: Experiment,
    start: str,
    end: str,
    *,
    executable: str | None = None,
) -> list[tuple[Stage, list[str]]]:
    """Build component commands against this experiment's effective paths."""
    paths = write_effective_configs(experiment)
    python = executable or sys.executable
    commands = {
        "fetch": [
            python,
            "-u",
            "-m",
            STAGE_BY_KEY["fetch"].module,
            str(paths["ingestion"]),
        ],
        "canonical": [
            python,
            "-u",
            "-m",
            STAGE_BY_KEY["canonical"].module,
            str(paths["canonical"]),
        ],
        "split": [
            python,
            "-u",
            "-m",
            STAGE_BY_KEY["split"].module,
            str(paths["split"]),
            "--seeds",
            str(paths["seeds"]),
        ],
        "graphs": [
            python,
            "-u",
            "-m",
            STAGE_BY_KEY["graphs"].module,
            str(paths["graphs"]),
            "--seeds",
            str(paths["seeds"]),
        ],
        "train": [
            python,
            "-u",
            "-m",
            STAGE_BY_KEY["train"].module,
            str(paths["model"]),
            "--output-dir",
            str(experiment.root / "model"),
        ],
    }
    return [(stage, commands[stage.key]) for stage in selected_stages(start, end)]


def _existing_stage_artifacts(experiment: Experiment, stage: Stage) -> list[Path]:
    candidates = {
        "fetch": (),
        "canonical": tuple(
            experiment.root / "canonical" / name
            for name in ("gauge_hourly.csv", "stations.csv", "days.csv", "report.md")
        ),
        "split": tuple((experiment.root / "split").glob("*")),
        "graphs": (experiment.root / "graphs",),
        "train": (experiment.root / "model",),
    }[stage.key]
    return [path for path in candidates if path.exists()]


def ensure_fresh_outputs(experiment: Experiment, stages: Sequence[Stage]) -> None:
    """Refuse a selected stage that would replace an experiment artifact."""
    for stage in stages:
        existing = _existing_stage_artifacts(experiment, stage)
        if existing:
            rendered = ", ".join(str(path) for path in existing[:3])
            raise FileExistsError(
                f"{stage.label} output already exists: {rendered}. Clone this "
                "experiment before running changed settings."
            )


async def _call(callback: OutputCallback | StatusCallback | None, *args: str) -> None:
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


class PipelineRunner:
    """Run a selected pipeline interval and stream its combined output."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self._process: asyncio.subprocess.Process | None = None
        self._stop_requested = False

    @property
    def running(self) -> bool:
        return self._process is not None

    def _save_status(self, payload: dict[str, Any]) -> None:
        payload["updated_utc"] = _now()
        _write_json(self.experiment.status_path, payload)

    def _write_report(self, status: str, message: str, payload: dict[str, Any]) -> None:
        lines = [
            f"# Experiment: {self.experiment.name}",
            "",
            f"Status: {status}",
            "",
            message,
            "",
            "## Pipeline stages",
            "",
        ]
        lines.extend(
            f"- {STAGE_BY_KEY[key].label}: {value}"
            for key, value in payload["stages"].items()
        )
        (self.experiment.root / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _mark_partial_component_stopped(self, stage: str | None) -> None:
        """Correct a component report when termination prevents its own cleanup."""
        if stage != "train":
            return
        report = self.experiment.root / "model" / "report.md"
        if not report.is_file():
            return
        content = report.read_text(encoding="utf-8")
        if "Status: RUNNING" in content:
            report.write_text(
                content.replace(
                    "Status: RUNNING",
                    "Status: STOPPED\n\nTraining was stopped by the experiment runner; "
                    "partial artifacts are not a completed result.",
                    1,
                ),
                encoding="utf-8",
            )

    async def run(
        self,
        start: str,
        end: str,
        *,
        on_output: OutputCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        if self.running:
            raise ExperimentError("A pipeline is already running")
        stages = selected_stages(start, end)
        ensure_fresh_outputs(self.experiment, stages)
        commands = stage_commands(self.experiment, start, end)
        status = read_status(self.experiment)
        status.update(
            {
                "state": "RUNNING",
                "started_utc": _now(),
                "finished_utc": None,
                "selection": {"start": start, "end": end},
                "message": f"Running {stages[0].label} through {stages[-1].label}",
            }
        )
        self._stop_requested = False
        self._save_status(status)
        self._write_report("RUNNING", status["message"], status)
        log_path = self.experiment.root / "log.txt"

        try:
            with log_path.open("a", encoding="utf-8") as log:
                for stage, command in commands:
                    status["current_stage"] = stage.key
                    status["stages"][stage.key] = "RUNNING"
                    self._save_status(status)
                    self._write_report("RUNNING", status["message"], status)
                    heading = (
                        f"\n[{datetime.now(UTC).isoformat()}] {stage.label}\n"
                        f"$ {shlex.join(command)}\n"
                    )
                    log.write(heading)
                    log.flush()
                    await _call(on_output, heading.rstrip())
                    await _call(on_status, stage.key, "RUNNING")
                    environment = os.environ.copy()
                    environment["PYTHONUNBUFFERED"] = "1"
                    self._process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=self.experiment.project_root,
                        env=environment,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    assert self._process.stdout is not None
                    while line := await self._process.stdout.readline():
                        text = line.decode(errors="replace").rstrip("\n")
                        log.write(text + "\n")
                        log.flush()
                        await _call(on_output, text)
                    code = await self._process.wait()
                    self._process = None
                    if self._stop_requested:
                        raise asyncio.CancelledError
                    if code != 0:
                        raise ExperimentError(
                            f"{stage.label} exited with status {code}; see {log_path}"
                        )
                    status["stages"][stage.key] = "DONE"
                    self._save_status(status)
                    await _call(on_status, stage.key, "DONE")
            status.update(
                {
                    "state": "DONE",
                    "current_stage": None,
                    "finished_utc": _now(),
                    "message": (
                        f"Completed {stages[0].label} through {stages[-1].label}"
                    ),
                }
            )
            self._save_status(status)
            self._write_report("DONE", status["message"], status)
        except asyncio.CancelledError:
            current = status.get("current_stage")
            if current:
                status["stages"][current] = "STOPPED"
            self._mark_partial_component_stopped(current)
            status.update(
                {
                    "state": "STOPPED",
                    "current_stage": None,
                    "finished_utc": _now(),
                    "message": (
                        "Stopped by the researcher; partial artifacts were retained."
                    ),
                }
            )
            self._save_status(status)
            self._write_report("STOPPED", status["message"], status)
            await _call(on_output, status["message"])
        except BaseException as error:
            current = status.get("current_stage")
            if current:
                status["stages"][current] = "FAILED"
            status.update(
                {
                    "state": "FAILED",
                    "current_stage": None,
                    "finished_utc": _now(),
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            self._save_status(status)
            self._write_report("FAILED", status["message"], status)
            await _call(on_output, status["message"])
            raise
        finally:
            self._process = None

    async def stop(self) -> None:
        """Ask the active component process to stop, then force it if necessary."""
        self._stop_requested = True
        process = self._process
        if process is None or process.returncode is not None:
            return
        status = read_status(self.experiment)
        if status.get("state") == "RUNNING":
            status.update(
                {
                    "state": "STOPPING",
                    "message": (
                        "Stopping the active component; partial artifacts will be "
                        "retained."
                    ),
                }
            )
            self._save_status(status)
            self._write_report("STOPPING", status["message"], status)
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


def load_history(experiment: Experiment) -> list[dict[str, float | int]]:
    """Load finite training history rows without importing pandas into the UI."""
    import csv
    import math

    path = experiment.root / "model" / "history.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            try:
                row: dict[str, float | int] = {
                    "fold": int(record["fold"]),
                    "epoch": int(record["epoch"]),
                    "train_loss": float(record["train_loss"]),
                    "val_loss": float(record["val_loss"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ExperimentError(
                    f"Invalid training history row in {path}"
                ) from error
            if not math.isfinite(row["train_loss"]) or not math.isfinite(
                row["val_loss"]
            ):
                raise ExperimentError(f"Non-finite training history value in {path}")
            rows.append(row)
    return rows
