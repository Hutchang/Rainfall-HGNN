# Experiment workbench (TUI)

The terminal workbench lets a researcher create an experiment, edit its configuration,
run any continuous part of the rainfall pipeline, watch live output and inspect training
losses without editing Python or assembling commands.

## Start it

From the repository root, synchronise the locked environment once and launch the app:

```console
uv sync --locked
uv run --locked python -m src.experiments.tui
```

Open a particular experiment directly, including one outside the repository's default
`experiments/` directory:

```console
uv run --locked python -m src.experiments.tui /path/to/my-experiment
```

The interface also supports `--experiments-dir /path/to/parent` to change the default
parent used for discovery and new experiments. Press `Ctrl+Q` to leave the app.

## Research workflow

1. On **Experiments**, choose **New**, name the experiment and accept the default
   `experiments/` parent or enter another parent folder.
2. On **Configuration**, choose any YAML or JSON file. The new experiment contains a
   complete snapshot of the repository's `config/` tree, not only the five currently
   used by the gauge pipeline. **Save** and `Ctrl+S` are the same action: both validate
   the syntax before replacing the file.
3. On **Run pipeline**, choose the first and last component to run. Both ends are
   inclusive. For example, **02 · Pre-process tables** through **06 · Train and
   evaluate** skips the BoM fetch but builds every downstream artifact.
4. Read combined stdout and stderr in the live log. Training prints one line per epoch
   with train loss, validation loss, best validation loss, accumulated deteriorations,
   remaining patience and the early-stopping state. Rainfall prediction is regression,
   so the app does not invent an undefined "accuracy" number.
5. **Stop** changes the run to `STOPPING`, terminates the current component, then records
   `STOPPED`; partial artifacts are retained for diagnosis and are not called a result.
6. On **Results**, follow the live, full-pane train and validation loss line chart. Its
   axes are Loss and Epoch, the two series have distinct colours, and the fold selector
   lets you inspect completed folds while training advances.

Use **Clone current** to start an ablation from an earlier experiment. It copies the
editable config snapshot and deliberately does not copy effective configs, logs, status
or results. This is the safe way to change one setting after a run has begun.

## What is stored

Each experiment is a self-contained folder:

```text
<experiment>/
  experiment.yaml      identity and config-role mapping
  status.json          pipeline and per-stage status
  report.md            top-level READY/RUNNING/DONE/FAILED/STOPPED report
  log.txt               combined component stdout and stderr
  config/               editable snapshot of every repository config file
  effective-config/     exact path-adjusted configs supplied to components
  canonical/            Component 02 outputs
  split/                Component 03 output
  graphs/               Component 04 outputs
  model/                Component 06 history, checkpoint, predictions and report
```

The app never reuses an existing experiment folder. Once a pipeline run starts, its
editable config snapshot is locked so it continues to describe the artifacts in that
folder. It also refuses to run a selected stage when that stage's output already exists.

Component 01 remains the exception to per-run output: the fetch fills the append-only
raw cache selected by the ingestion config. The effective Component 02 config reads that
cache and every later output is routed into the experiment folder.

## Configuration paths

The current gauge pipeline uses these copies:

| Role | Editable experiment copy |
|---|---|
| source selection and raw cache | `config/ingestion/sydney.yaml` |
| standard tables | `config/canonical/standard.yaml` |
| folds | `config/splits/sydney_5fold.yaml` |
| graphs | `config/graphs/gauge_baseline.yaml` |
| model and training | `config/models/hgnn.yaml` |
| shared randomness | `config/seeds.yaml` |

Paths connecting component outputs are written explicitly under `effective-config/`.
The research settings remain the values from the editable copies; only paths are routed
so that Components 02, 03, 04 and 06 exchange files inside this experiment.

## Current boundary

This workbench provides the demonstration-critical subset requested ahead of the full
D07 runner: folders per experiment, config copies, selected pipeline ranges, live
training output and results in the experiment folder. The D07 owner still owns the
registry, full provenance and fingerprints, component caching, run comparison, and
effective-default expansion. The workbench service is separate from the widgets so the
full runner can replace it without redesigning the interface.

Useful next UI additions are structured forms for frequently changed model settings,
side-by-side run comparison backed by D07's future registry, metric filtering across
folds, and a preflight page that checks upstream input availability before a long job.
