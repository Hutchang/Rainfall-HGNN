# COMP3888 rainfall fusion

This repository is turning two inherited rainfall-reconstruction research codebases
into a reproducible Australian data and experiment system. The current scientific
baseline is gauge-only rainfall interpolation over Greater Sydney for
**2021-01-01 through 2022-12-31**. Radar, satellite, temporal modelling, and
Australia-wide scaling follow only after that baseline is valid.

## Geographic scope

This project supports Australian rainfall data only. Singapore data, National
Environment Agency (NEA) file layouts, Singapore timezone assumptions, and commercial
microwave link inputs are outside the project for its entire lifetime. They must not be
retained as optional modes, compatibility flags, examples, or proposed future work.
Inherited source history remains recoverable from Git, but those capabilities do not
belong in the target system. Some restored inherited files still mix reusable algorithms
with those regional assumptions; they are review-only until the useful code is extracted
and the out-of-scope remainder is removed file by file.

## Start here

1. [Current state](docs/CURRENT_STATE.md) — where the project is, what Sprint 2 is doing,
   repository boundaries, and the latest verification run.
2. [Known issues](docs/ISSUES.md) — every open bug, gap and unresolved decision, with a
   permanent id.
3. [Integrated pipeline guide](docs/graphs/README.md) — run tables, splits, graphs and
   training, with an offline example.
4. [Sprint 2 requirements](docs/requirements/README.md) — deliverables D07 to D13, the six
   day-one decisions, and the rules everyone follows.

Sprint 1 is complete and closed; Components 01, 02, 03, 04 and 06 shipped in it. Sprint 2
is organised as deliverables D07 to D13 (experiment runner, Sydney gauge-only baseline,
radar fusion, Australia-wide, report, a stretch region-drawing app, and source grids read
from NCI), numbered on from the Sprint 1 components, each with its own document under
`docs/requirements/`. The
Sprint 1 contracts in [docs/archive/sprint1/](docs/archive/sprint1/) are still the
accurate description of what each component implements today. They are reference, not
current assignments:

- [01 — Get the raw data onto disk](docs/archive/sprint1/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md)
- [02 — Build the standard tables](docs/archive/sprint1/02_CANONICAL_DATA_REQUIREMENTS.md)
- [03 — Pick the train, validation and test stations](docs/archive/sprint1/03_STUDY_SPLIT_REQUIREMENTS.md)
- [04 — Build the graph samples](docs/archive/sprint1/04_GRAPH_SAMPLE_REQUIREMENTS.md)
- [06 — Train the model and write predictions](docs/archive/sprint1/06_MODEL_ADAPTER_REQUIREMENTS.md)

There is no 05. The golden-fixture component was cut on 2026-09-01 and its old document
is in [docs/archive/](docs/archive/05_GOLDEN_FIXTURE_REQUIREMENTS_CUT.md).

## Current truth in one paragraph

The two inherited implementations have been audited, and the retained graph/model core
can run a real gauge-only graph on Sydney data. However, the hand-exported rainfall
values have time- and station-varying
distortion and cannot support scientific millimetre-based results. The production path
therefore starts with reproducibly collecting source-faithful BoM data for 2021–2022,
then preparing the standard dataset, checking splits, creating graph samples, training
through a common interface, and evaluating every method on identical targets.

Every step on that path now exists and is connected, but the result is
software-integration evidence rather than science: provenance, units, coverage and
held-out-gauge independence are still open, and the legacy hand export may not be quoted
in millimetres. How far each component has actually got is recorded once, in
[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md), and what is known to be wrong is in
[docs/ISSUES.md](docs/ISSUES.md). This README does not repeat either — a second copy of a
status is a second answer.

## How the documentation stays true

Two documents carry live information and each has one job. `docs/CURRENT_STATE.md` says
where the project is; it names the latest commit rather than a branch, and on a feature
branch it is main's latest version plus the lines your own work makes untrue.
`docs/ISSUES.md` says what is broken, missing or undecided; it is the only place an issue
is recorded, and every issue keeps a permanent `ISS-xx` id. Other documents may cite an
id, but never restate an issue's severity, owner or status (one exception: the decisions
list in the Sprint 2 requirements names each issue's severity so the decisions can be
ranked). Both are updated in the same
pull request as the change that makes them wrong. The full rules are in
[AGENTS.md](AGENTS.md#keeping-the-documentation-true).

## Repository map

| Path | Purpose |
|---|---|
| `docs/CURRENT_STATE.md`, `docs/ISSUES.md` | Live project state and the issue register; updated with every change that affects them |
| `docs/requirements/` | Sprint 2 deliverable documents, decisions, shared rules, glossary and the authoring standard; primary implementation authority |
| `docs/archive/` | Historical planning, audit evidence, closed-sprint state, and run records; not active requirements |
| `src/ingestion/`, `config/ingestion/`, `docs/ingestion/` | Component 01 source data: BoM gauge fetcher, region config, and operator guide. Production path |
| `src/canonical/`, `config/canonical/`, `docs/canonical/` | Component 02 standard tables: tips to hourly, the station table, verification, and grid reshaping. Production path |
| `src/study/`, `src/reproducibility/`, `config/splits/`, `config/seeds.yaml` | Component 03 study splits: spatial clustering, fold construction, role assignment, and seed control. Production path |
| `src/graph/`, `config/graphs/`, `docs/graphs/` | Component 04 graph samples: directed gauge edges, sensor layers, and the per-fold builder. Production path |
| `src/models/`, `config/models/`, `docs/models/` | Component 06 model adapters: HGNN, IDW and network mean trained and scored through one prediction contract. Production path |
| `models/`, `training/`, `benchmarks/`, `src/cml/`, `src/radar/`, `src/raingauge/`, `src/sampling/`, `src/visualization/` | Restored inherited reference modules awaiting file-by-file classification; review-only |
| `adapters/legacy_sydney_gauge_export.py` | Transitional Sydney legacy-data export; not a canonical ingestion pipeline |
| `train_fused.py` | Validated transitional 2022 smoke runner; not the target architecture |
| `tests/` | Audit/regression tests; each component commits a small offline sample under `tests/data/` |
| `database/`, `experiments/` | Local generated data/results; intentionally ignored by Git. Current adapter data is regression-only, not canonical. |

## Supporting references

These files are useful when the work calls for them, but are not required onboarding
reading:

- [Coding-agent instructions](AGENTS.md) — mandatory repository rules and reading order
  for AI coding agents.
- [Requirements authoring standard](docs/requirements/AUTHORING_STANDARD.md) — rules
  for changing a component contract.
- [Legacy-code guide](docs/LEGACY_CODE.md) — retained transitional code, restored
  reference areas, and removed entry points.
- [Satellite and radar adapter](docs/grids/README.md) — the grid adapter: standard
  tables, node features and an offline example. Component 06's training scope is
  described separately.
- [Historical archive](docs/archive/README.md) — reconciliation audits, smoke records,
  and superseded planning documents.

## Environment

Python 3.14 or newer, managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml`
declares the direct dependencies and `uv.lock` pins the exact resolved environment.
`requirements.txt` and `requirements-dev.txt` were deleted when the project moved to uv
in `a25ac5f`; do not reintroduce them as dependency authority.

```console
uv sync --locked
```

Add and remove dependencies with `uv add` and `uv remove`, using `--dev` for development
tools. Do not install project packages directly with pip.

The default PyTorch package is suitable for CPU onboarding. Install the matching CUDA
wheel from the official PyTorch index only when a component actually needs GPU work.
Record the chosen build in run metadata.

macOS and Windows checkouts support CPU onboarding. Prepare a separate Linux environment
for Gadi; do not copy a local `.venv` or replace locked dependencies with ad-hoc pip
installs. See the [training guide](docs/models/README.md) and
[Gadi setup](docs/nci/README.md) for the Component 06 runner and CUDA compatibility checks.

Run the current checks with:

```console
uv run --locked python -m pytest tests -q -rx
uv run --locked ruff check .
```

`ruff check .` reports a large number of findings across inherited, transitional and other
pre-existing files awaiting classification (ISS-17); the current count is in
[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md). New component modules pass focused Ruff
checks. A clean repository-wide lint run is not yet the baseline; do not mass-fix
inherited files merely to silence it.

Some existing tests skip when the untracked legacy gauge data is absent. Each component
replaces that dependency by committing a small real sample under `tests/data/`, so the
suite runs on a fresh clone with no network.

## Important guardrails

- Do not reintroduce Singapore/NEA paths, timezone assumptions, or commercial microwave
  link support. They are permanently out of scope.
- Do not create private splits, seeds, prediction schemas, or station identifiers.
- Do not silently fix inherited scientific behaviour; label compatibility and corrected
  modes separately.
- Do not commit raw source data, credentials, checkpoints, or experiment outputs.
