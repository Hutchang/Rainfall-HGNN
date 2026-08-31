# COMP3888 rainfall fusion

This repository is turning two inherited rainfall-reconstruction research codebases
into a reproducible Australian data and experiment system. The current scientific
baseline is gauge-only rainfall interpolation over Greater Sydney for
**2021-01-01 through 2022-12-31**. Radar, satellite, temporal modelling, and
Australia-wide scaling follow only after that baseline is valid.

## Start here

1. [Implementation handoff](docs/IMPLEMENTATION_HANDOFF.md) — current state, repository
   map, work order, and teammate workflow.
2. [Component requirements](docs/requirements/README.md) — assignable implementation
   work packages and shared rules.
3. [Requirements authoring standard](docs/requirements/AUTHORING_STANDARD.md) — how
   those briefs are maintained.
4. [Baseline reconciliation status](docs/baseline-reconciliation/STATUS.md) — evidence
   from auditing and running the inherited code.
5. [Legacy-code guide](docs/LEGACY_CODE.md) — what remains transitional and what was
   removed from the project root.

Historical planning documents live under [docs/archive](docs/archive/README.md) and are
not active requirements.

## Current truth in one paragraph

The inherited Niramay and Jylow implementations have been audited. Jylow can run a real
gauge-only graph on Sydney data, and the Niramay-to-Jylow adapter proves the integration
shape. However, the hand-exported rainfall values have time- and station-varying
distortion and cannot support scientific millimetre-based results. The production path
therefore starts with a source-faithful, reproducible BoM ingestion for 2021–2022, then
canonical data, audited splits, graph samples, a model adapter, and identical-target
prediction evaluation.

## Repository map

| Path | Purpose |
|---|---|
| `docs/requirements/` | Current component contracts; primary implementation authority |
| `docs/baseline-reconciliation/` | Audit evidence and historical run records |
| `src/`, `models/`, `training/` | Inherited/transitional implementation to wrap or replace behind component contracts |
| `adapters/niramay_to_jylow.py` | Transitional legacy-data adapter; not a canonical ingestion pipeline |
| `train_fused.py` | Validated transitional 2022 smoke runner; not the target architecture |
| `tests/` | Existing audit/regression tests; component 05 will add an offline golden fixture |
| `database/`, `experiments/` | Local generated data/results; intentionally ignored by Git |

## Environment

Python 3.13 is the recorded working environment. Create a fresh virtual environment and
install the focused dependencies:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

The default PyTorch package is suitable for CPU onboarding. Install the matching CUDA
wheel from the official PyTorch index only when a component actually needs GPU work.
Record the chosen build in run metadata.

Run the current tests with:

```powershell
.venv\Scripts\python.exe -m pytest tests -q -rx
```

Some existing tests skip when the untracked legacy gauge data/worktree is absent. The
golden-fixture component is responsible for replacing that dependency with an offline,
committed test path.

## Important guardrails

- Do not quote current legacy-data RMSE as millimetres or as a paper reproduction.
- Do not add radar, satellite, or fake sensor nodes to make a gauge-only path run.
- Do not create private splits, seeds, prediction schemas, or station identifiers.
- Do not silently fix inherited scientific behaviour; label compatibility and corrected
  modes separately.
- Do not commit raw source data, credentials, checkpoints, or experiment outputs.
- Preserve user changes in a dirty worktree and keep component changes separable.
