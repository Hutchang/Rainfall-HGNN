"""Interactive terminal application for configuring and running experiments.

Launch from the repository root with::

    uv run --locked python -m src.experiments.tui
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .chart import loss_chart
from .results import results_summary, save_loss_plot
from .workspace import (
    STAGES,
    Experiment,
    ExperimentError,
    PipelineRunner,
    configuration_locked,
    create_experiment,
    discover_experiments,
    list_config_files,
    load_experiment,
    load_history,
    read_config_text,
    read_status,
    save_config_text,
)


class NewExperimentScreen(ModalScreen[tuple[str, str] | None]):
    """Collect a name and parent folder for a new experiment."""

    DEFAULT_CSS = """
    NewExperimentScreen {
        align: center middle;
        background: transparent;
    }

    #new-dialog {
        width: 74;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: transparent;
    }

    #new-dialog Input {
        margin-bottom: 1;
    }

    #new-dialog .buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def __init__(self, base_dir: Path, *, cloning: bool = False):
        super().__init__()
        self.base_dir = base_dir
        self.cloning = cloning

    def compose(self) -> ComposeResult:
        title = "Clone experiment for an ablation" if self.cloning else "New experiment"
        with Vertical(id="new-dialog"):
            yield Label(title, classes="dialog-title")
            yield Label("Experiment name")
            yield Input(placeholder="e.g. dropout-015", id="new-name")
            yield Label("Parent folder")
            yield Input(value=str(self.base_dir), id="new-location")
            yield Static(
                "A new folder named after the experiment will be created. Existing "
                "folders are never reused.",
                classes="muted",
            )
            with Horizontal(classes="buttons"):
                yield Button("Cancel", id="cancel-new")
                yield Button("Create", id="create-new", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#new-name", Input).focus()

    @on(Button.Pressed, "#cancel-new")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#create-new")
    def create(self) -> None:
        name = self.query_one("#new-name", Input).value
        location = self.query_one("#new-location", Input).value
        if not name.strip() or not location.strip():
            self.notify("Name and parent folder are required", severity="error")
            return
        self.dismiss((name, location))


class LossPlot(Static):
    """A loss chart that expands to the terminal space available."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rows: list[dict[str, float | int | str]] = []
        self._fold: int | None = None

    def set_history(
        self, rows: list[dict[str, float | int | str]], fold: int | None
    ) -> None:
        self._rows = rows
        self._fold = fold
        self.refresh(layout=True)

    def render(self):
        return loss_chart(
            self._rows,
            fold=self._fold,
            width=max(42, self.size.width - 2),
            height=max(12, self.size.height - 2),
        )


class ExperimentApp(App[None]):
    """Researcher-facing experiment configuration and execution application."""

    TITLE = "Rainfall Experiment Workbench"
    SUB_TITLE = "Configure · run · inspect"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen, Header, Footer, TabbedContent, TabPane, RichLog, TextArea {
        background: transparent;
    }

    #workspace-header {
        height: 3;
        padding: 1 2;
        background: transparent;
        color: $text-muted;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }

    .toolbar {
        height: auto;
        margin-bottom: 1;
    }

    .toolbar Button, .toolbar Select {
        margin-right: 1;
    }

    .section-title, .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .muted {
        color: $text-muted;
    }

    #local-experiment-select, #gadi-experiment-select {
        width: 2fr;
    }

    #gadi-note {
        height: auto;
        min-height: 2;
        margin-bottom: 1;
        color: $text-muted;
    }

    #open-path {
        width: 2fr;
    }

    #config-file {
        width: 1fr;
    }

    #config-editor {
        height: 1fr;
        border: round $panel;
    }

    #config-message, #pipeline-status {
        height: auto;
        min-height: 2;
        padding: 0 1;
        color: $text-muted;
    }

    #start-stage, #end-stage {
        width: 1fr;
    }

    #run-log {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }

    #loss-chart {
        width: 1fr;
        height: 24;
        min-height: 18;
        padding: 1;
        border: round $accent;
        background: transparent;
        color: $text;
    }

    #results-scroll {
        width: 1fr;
        height: 1fr;
        background: transparent;
        overflow-y: auto;
    }

    #results-status {
        width: 1fr;
        height: auto;
        padding: 1;
        color: $text-muted;
    }

    #results-fold {
        width: 24;
    }

    #results-summary {
        height: auto;
        min-height: 8;
        padding: 0 1;
        border-top: solid $panel;
        background: transparent;
    }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("ctrl+n", "new_experiment", "New"),
        ("ctrl+s", "save_config", "Save config"),
        ("ctrl+r", "run_pipeline", "Run"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        experiments_dir: Path | str | None = None,
        open_experiment: Path | str | None = None,
    ):
        super().__init__(ansi_color=True)
        self.project_root = Path(project_root or Path(__file__).parents[2]).resolve()
        self.experiments_dir = (
            Path(experiments_dir or self.project_root / "experiments")
            .expanduser()
            .resolve()
        )
        self.initial_experiment = (
            Path(open_experiment).expanduser() if open_experiment else None
        )
        self.current: Experiment | None = None
        self.runner: PipelineRunner | None = None
        self._available_result_folds: tuple[int, ...] = ()
        self._latest_result_fold: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("No experiment open", id="workspace-header")
        with TabbedContent(initial="experiments-tab"):
            with TabPane("Experiments", id="experiments-tab"):
                yield Label("Experiment workspaces", classes="section-title")
                yield Label("Local experiments")
                with Horizontal(classes="toolbar"):
                    yield Select(
                        [],
                        prompt="Choose a saved local experiment",
                        id="local-experiment-select",
                    )
                    yield Button("Refresh", id="refresh-experiments")
                    yield Button("New", id="new-experiment", variant="primary")
                    yield Button("Clone current", id="clone-experiment")
                yield Label("Gadi experiments")
                yield Select(
                    [("NCI connection not configured", "not-connected")],
                    value="not-connected",
                    allow_blank=False,
                    disabled=True,
                    id="gadi-experiment-select",
                )
                yield Static(
                    "Stub only: browsing and opening Gadi results will be enabled "
                    "when the NCI interface is available.",
                    id="gadi-note",
                )
                with Horizontal(classes="toolbar"):
                    yield Input(
                        placeholder="Open any experiment folder or experiment.yaml",
                        id="open-path",
                    )
                    yield Button("Open path", id="open-experiment")
                yield Markdown(
                    "Create an experiment to snapshot every file under `config/`. "
                    "Clone a completed experiment before changing one setting for an "
                    "ablation; results and logs are not copied."
                )
            with TabPane("Configuration", id="configuration-tab"):
                with Horizontal(classes="toolbar"):
                    yield Select([], prompt="Choose any config file", id="config-file")
                    yield Button("Reload", id="reload-config")
                    yield Button("Save (Ctrl+S)", id="save-config", variant="primary")
                yield Static(
                    "Open an experiment to edit its configuration.",
                    id="config-message",
                )
                yield TextArea(
                    language="yaml",
                    show_line_numbers=True,
                    soft_wrap=False,
                    id="config-editor",
                )
            with TabPane("Run pipeline", id="run-tab"):
                yield Label(
                    "Select an inclusive pipeline range", classes="section-title"
                )
                with Horizontal(classes="toolbar"):
                    yield Select(
                        [(stage.label, stage.key) for stage in STAGES],
                        value="fetch",
                        allow_blank=False,
                        id="start-stage",
                    )
                    yield Select(
                        [(stage.label, stage.key) for stage in STAGES],
                        value="train",
                        allow_blank=False,
                        id="end-stage",
                    )
                    yield Button("Run", id="run-pipeline", variant="success")
                    yield Button(
                        "Stop", id="stop-pipeline", variant="error", disabled=True
                    )
                yield Static("No experiment open.", id="pipeline-status")
                yield RichLog(
                    highlight=True,
                    markup=False,
                    wrap=True,
                    max_lines=10_000,
                    id="run-log",
                )
            with TabPane("Results", id="results-tab"):
                with VerticalScroll(id="results-scroll"):
                    yield Label(
                        "Live train vs validation loss", classes="section-title"
                    )
                    with Horizontal(classes="toolbar"):
                        yield Static("No experiment open.", id="results-status")
                        yield Select(
                            [],
                            prompt="Fold",
                            id="results-fold",
                        )
                        yield Button(
                            "Save graph PNG",
                            id="save-loss-chart",
                            disabled=True,
                        )
                    yield LossPlot(id="loss-chart")
                    yield Markdown(
                        "### Results summary\n\n"
                        "Open an experiment to inspect its results.",
                        id="results-summary",
                    )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_experiment_options()
        if self.initial_experiment:
            self._open(self.initial_experiment)
        else:
            found = discover_experiments(self.experiments_dir)
            if found:
                self._set_current(found[0])
        self.set_interval(1.0, self._refresh_live_views)

    def _refresh_experiment_options(self) -> None:
        select = self.query_one("#local-experiment-select", Select)
        experiments = discover_experiments(self.experiments_dir)
        select.set_options(
            [
                (f"{item.name}  ·  {item.created_utc[:16]}", str(item.root))
                for item in experiments
            ]
        )
        if self.current and self.current.root.parent == self.experiments_dir:
            select.value = str(self.current.root)

    def _set_current(self, experiment: Experiment) -> None:
        self.current = experiment
        self.runner = PipelineRunner(experiment)
        status = read_status(experiment)
        self.query_one("#workspace-header", Static).update(
            f"{experiment.name}  ·  {experiment.root}  ·  "
            f"{status.get('state', 'UNKNOWN')}"
        )
        config_select = self.query_one("#config-file", Select)
        files = list_config_files(experiment)
        config_select.set_options([(path, path) for path in files])
        if files:
            config_select.value = files[0]
            self._load_config(files[0])
        locked = configuration_locked(experiment)
        self.query_one("#config-editor", TextArea).read_only = locked
        self.query_one("#save-config", Button).disabled = locked
        message = (
            "Configuration is locked because this run has started. Clone the "
            "experiment to change settings."
            if locked
            else (
                "All YAML and JSON files are editable. Save and Ctrl+S are the "
                "same action; both validate before writing."
            )
        )
        self.query_one("#config-message", Static).update(message)
        self.query_one("#open-path", Input).value = str(experiment.root)
        log = self.query_one("#run-log", RichLog)
        log.clear()
        log_path = experiment.root / "log.txt"
        if log_path.is_file():
            for line in log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                log.write(line)
        self._refresh_live_views()

    def _open(self, path: Path | str) -> None:
        try:
            self._set_current(load_experiment(path))
        except (ExperimentError, OSError) as error:
            self.notify(str(error), severity="error", timeout=8)

    def _load_config(self, relative: str) -> None:
        if not self.current:
            return
        try:
            text = read_config_text(self.current, relative)
        except (ExperimentError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        editor = self.query_one("#config-editor", TextArea)
        editor.language = "json" if relative.endswith(".json") else "yaml"
        editor.load_text(text)

    def _refresh_live_views(self) -> None:
        if not self.current:
            return
        try:
            status = read_status(self.current)
        except ExperimentError, OSError, ValueError:
            return
        stage_text = "  ·  ".join(
            f"{key}: {value}" for key, value in status.get("stages", {}).items()
        )
        self.query_one("#pipeline-status", Static).update(
            f"{status.get('state', 'UNKNOWN')} · "
            f"{status.get('message', '')}\n{stage_text}"
        )
        self.query_one("#workspace-header", Static).update(
            f"{self.current.name}  ·  {self.current.root}  ·  "
            f"{status.get('state', 'UNKNOWN')}"
        )
        self.query_one("#results-status", Static).update(
            f"{status.get('state', 'UNKNOWN')} · {status.get('message', '')}"
        )
        try:
            history = load_history(self.current)
        except ExperimentError, OSError:
            history = []
        folds = tuple(sorted({int(row["fold"]) for row in history}))
        fold_select = self.query_one("#results-fold", Select)
        previous_value = fold_select.value
        previous_latest = self._latest_result_fold
        if folds != self._available_result_folds:
            fold_select.set_options([(f"Fold {fold}", fold) for fold in folds])
            self._available_result_folds = folds
        if folds:
            latest = folds[-1]
            follow_latest = (
                previous_value is Select.BLANK or previous_value == previous_latest
            )
            if follow_latest or previous_value not in folds:
                fold_select.value = latest
            self._latest_result_fold = latest
            selected = int(fold_select.value)
        else:
            selected = None
            self._latest_result_fold = None
        self.query_one("#loss-chart", LossPlot).set_history(history, selected)
        self.query_one("#save-loss-chart", Button).disabled = not history
        self.query_one("#results-summary", Markdown).update(
            results_summary(
                self.current,
                history,
                selected_fold=selected,
                state=str(status.get("state", "UNKNOWN")),
            )
        )

    def _show_new_dialog(self, *, cloning: bool) -> None:
        if cloning and not self.current:
            self.notify("Open an experiment before cloning it", severity="warning")
            return

        def create_from_dialog(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            name, location = result
            try:
                experiment = create_experiment(
                    name,
                    location,
                    project_root=self.project_root,
                    template=self.current.root if cloning and self.current else None,
                )
            except (ExperimentError, OSError) as error:
                self.notify(str(error), severity="error", timeout=8)
                return
            self._set_current(experiment)
            self._refresh_experiment_options()
            self.notify(f"Created {experiment.name}", severity="information")

        self.push_screen(
            NewExperimentScreen(self.experiments_dir, cloning=cloning),
            create_from_dialog,
        )

    def action_new_experiment(self) -> None:
        self._show_new_dialog(cloning=False)

    def action_save_config(self) -> None:
        self._save_config()

    def action_run_pipeline(self) -> None:
        self._start_pipeline()

    @on(Select.Changed, "#local-experiment-select")
    def select_experiment(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._open(str(event.value))

    @on(Select.Changed, "#config-file")
    def select_config(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._load_config(str(event.value))

    @on(Select.Changed, "#results-fold")
    def select_result_fold(self, event: Select.Changed) -> None:
        if not self.current or event.value is Select.BLANK:
            return
        try:
            history = load_history(self.current)
        except ExperimentError, OSError:
            return
        selected = int(event.value)
        self.query_one("#loss-chart", LossPlot).set_history(history, selected)
        self.query_one("#results-summary", Markdown).update(
            results_summary(self.current, history, selected_fold=selected)
        )

    @on(Button.Pressed, "#save-loss-chart")
    def save_loss_chart(self) -> None:
        if not self.current:
            self.notify("Open an experiment first", severity="warning")
            return
        selected = self.query_one("#results-fold", Select).value
        fold = None if selected is Select.BLANK else int(selected)
        try:
            path = save_loss_plot(self.current, fold=fold)
        except (ExperimentError, OSError, ValueError) as error:
            self.notify(str(error), severity="error", timeout=8)
            return
        self.notify(
            f"Saved {path.relative_to(self.current.root)}",
            severity="information",
            timeout=8,
        )

    @on(Button.Pressed, "#refresh-experiments")
    def refresh_experiments(self) -> None:
        self._refresh_experiment_options()

    @on(Button.Pressed, "#new-experiment")
    def new_experiment(self) -> None:
        self._show_new_dialog(cloning=False)

    @on(Button.Pressed, "#clone-experiment")
    def clone_experiment(self) -> None:
        self._show_new_dialog(cloning=True)

    @on(Button.Pressed, "#open-experiment")
    def open_experiment(self) -> None:
        path = self.query_one("#open-path", Input).value
        if not path.strip():
            self.notify("Enter an experiment folder", severity="warning")
            return
        self._open(path)

    @on(Button.Pressed, "#reload-config")
    def reload_config(self) -> None:
        value = self.query_one("#config-file", Select).value
        if value is not Select.BLANK:
            self._load_config(str(value))

    def _save_config(self) -> None:
        if not self.current:
            self.notify("Open an experiment first", severity="warning")
            return
        value = self.query_one("#config-file", Select).value
        if value is Select.BLANK:
            self.notify("Choose a config file", severity="warning")
            return
        try:
            save_config_text(
                self.current,
                str(value),
                self.query_one("#config-editor", TextArea).text,
            )
        except (ExperimentError, OSError) as error:
            self.notify(str(error), severity="error", timeout=8)
            return
        self.query_one("#config-message", Static).update(
            f"Saved and validated config/{value}"
        )
        self.notify("Config saved", severity="information")

    @on(Button.Pressed, "#save-config")
    def save_config(self) -> None:
        self._save_config()

    def _start_pipeline(self) -> None:
        if not self.current or not self.runner:
            self.notify("Open an experiment first", severity="warning")
            return
        if self.runner.running:
            self.notify("A pipeline is already running", severity="warning")
            return
        start = self.query_one("#start-stage", Select).value
        end = self.query_one("#end-stage", Select).value
        if start is Select.BLANK or end is Select.BLANK:
            self.notify("Choose both pipeline stages", severity="warning")
            return
        self.run_worker(
            self._run_pipeline(str(start), str(end)),
            name="pipeline",
            group="pipeline",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#run-pipeline")
    def run_pipeline(self) -> None:
        self._start_pipeline()

    async def _run_pipeline(self, start: str, end: str) -> None:
        assert self.runner is not None
        run_button = self.query_one("#run-pipeline", Button)
        stop_button = self.query_one("#stop-pipeline", Button)
        run_button.disabled = True
        stop_button.disabled = False
        log = self.query_one("#run-log", RichLog)

        def show_output(line: str) -> None:
            log.write(line)

        def show_status(_stage: str, _status: str) -> None:
            self._refresh_live_views()

        try:
            await self.runner.run(
                start,
                end,
                on_output=show_output,
                on_status=show_status,
            )
            final_state = read_status(self.current)["state"]
            if final_state == "DONE":
                self.notify(
                    "Selected pipeline stages completed", severity="information"
                )
            elif final_state == "STOPPED":
                self.notify(
                    "Pipeline stopped; partial artifacts retained", severity="warning"
                )
        except (ExperimentError, FileExistsError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
        finally:
            run_button.disabled = False
            stop_button.disabled = True
            self._set_current(self.current)

    @on(Button.Pressed, "#stop-pipeline")
    async def stop_pipeline(self) -> None:
        if self.runner:
            self.notify("Stopping the active component…", severity="warning")
            await self.runner.stop()
            self._refresh_live_views()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        help="optional experiment directory or experiment.yaml to open",
    )
    parser.add_argument(
        "--experiments-dir",
        help="default parent directory for new and discovered experiments",
    )
    args = parser.parse_args(argv)
    app = ExperimentApp(
        experiments_dir=args.experiments_dir,
        open_experiment=args.experiment,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
