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

1. [Component requirements](docs/requirements/README.md) — the shared component map,
   work packages, contracts, and team rules.
2. [Implementation handoff](docs/IMPLEMENTATION_HANDOFF.md) — current sequencing,
   repository boundaries, blockers, and completion criteria.

Everyone should be able to see the complete requirements set. Each owner reads their
own document in full and reviews the input/output sections of adjacent components:

- [01 — Get the raw data onto disk](docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md)
- [02 — Build the standard tables](docs/requirements/02_CANONICAL_DATA_REQUIREMENTS.md)
- [03 — Pick the train, validation and test stations](docs/requirements/03_STUDY_SPLIT_REQUIREMENTS.md)
- [04 — Build the graph samples](docs/requirements/04_GRAPH_SAMPLE_REQUIREMENTS.md)
- [06 — Train the model and write predictions](docs/requirements/06_MODEL_ADAPTER_REQUIREMENTS.md)

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

## Repository map

| Path | Purpose |
|---|---|
| `docs/requirements/` | Current component contracts; primary implementation authority |
| `docs/archive/` | Historical planning, audit evidence, and run records; not active requirements |
| `src/`, `models/`, `training/`, `benchmarks/` | Transitional Sydney gauge core plus restored inherited reference modules awaiting file-by-file classification |
| `adapters/legacy_sydney_gauge_export.py` | Transitional Sydney legacy-data export; not a canonical ingestion pipeline |
| `train_fused.py` | Validated transitional 2022 smoke runner; not the target architecture |
| `tests/` | Audit/regression tests; each component commits a small offline sample under `tests/data/` |
| `database/`, `experiments/` | Local generated data/results; intentionally ignored by Git. Current adapter data is regression-only, not canonical. |

## Supporting references

These files are useful when the work calls for them, but are not required onboarding
reading:

- [Coding-agent instructions](AGENTS.md) — mandatory repository rules and reading order
  for AI coding agents.
- [Glossary](docs/requirements/GLOSSARY.md) — domain and project terminology.
- [Requirements authoring standard](docs/requirements/AUTHORING_STANDARD.md) — rules
  for changing a component contract.
- [Legacy-code guide](docs/LEGACY_CODE.md) — retained transitional code, restored
  reference areas, and removed entry points.
- [Client briefing](docs/CLIENT_BRIEFING.md) — external decisions and approval asks.
- [Historical archive](docs/archive/README.md) — reconciliation audits, smoke records,
  and superseded planning documents.

## Environment

Use the repository's locked uv environment. `pyproject.toml` currently requires Python
3.14 or later and `uv.lock` records the resolved dependencies:

```console
uv sync --locked
```

The macOS environment supports CPU onboarding. Prepare a separate Linux environment
for Gadi; do not copy the macOS `.venv` or replace locked dependencies with ad-hoc pip
installs. See the [training guide](docs/models/README.md) and
[Gadi setup](docs/nci/README.md) for the Component 06 runner and CUDA compatibility checks.

Run the current tests with:

```console
uv run --locked python -m pytest tests -q -rx
```

Some existing tests skip when the untracked legacy gauge data is absent. Each component
replaces that dependency by committing a small real sample under `tests/data/`, so the
suite runs on a fresh clone with no network.

## Important guardrails

- Do not quote current legacy-data RMSE as millimetres or as a paper reproduction.
- Do not add radar, satellite, or fake sensor nodes to make a gauge-only path run.
- Do not reintroduce Singapore/NEA paths, timezone assumptions, or commercial microwave
  link support. They are permanently out of scope.
- Do not create private splits, seeds, prediction schemas, or station identifiers.
- Do not silently fix inherited scientific behaviour; label compatibility and corrected
  modes separately.
- Do not commit raw source data, credentials, checkpoints, or experiment outputs.
- Preserve user changes in a dirty worktree and keep component changes separable.
