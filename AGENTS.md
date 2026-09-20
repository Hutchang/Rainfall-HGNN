# Instructions for coding agents

This file contains repository-wide instructions for AI coding agents. It is a navigation
and guardrail document, not a replacement for the component requirements.

Anything which follows ***(issue) in this document is metadata subject to review 

## Required reading order

Before editing code, read:

1. `README.md` for the project purpose, scope, repository map, and setup.
2. `docs/CURRENT_STATE.md` for where the project is and what the current sprint is doing.
3. `docs/ISSUES.md` for what is known to be broken, missing or undecided.
4. ***(context bloat)`docs/requirements/README.md` for the deliverables, the rule for
   what counts as work, and the shared rules.
5. The assigned deliverable document in full in `docs/requirements/`.
6. The `Inputs` and `Outputs` sections of directly adjacent component documents.
7. The implementation and tests directly relevant to the assigned work.

Do not read every archived investigation by default. `docs/archive/` is historical
evidence, not current implementation authority. Consult a specific archived file only
when an active requirement or a concrete investigation question points to it.

Before making changes, state:

- the assigned component and branch;
- the first `FRxx`, `OUTxx`, and `TESTxx` being implemented;
- the files or interfaces expected from adjacent components; and
- any blocker that prevents the first small implementation slice.

## Authority when documents disagree

Use this order:

1. the project lead's current instruction;
2. the assigned component requirements document;
3. shared rules in `docs/requirements/README.md`;
4. `docs/CURRENT_STATE.md`;
5. the root `README.md`;
6. archived investigation material.

Do not silently choose between conflicting scientific or component-contract
instructions. Record the conflict and ask the project lead or relevant reviewer.

## Scientific and data rules

- Component 01 stores source-faithful data. It must not aggregate rainfall hourly,
  normalise, interpolate, gap-fill, remove suspected outliers, or perform feature
  engineering.
- Component 02 is responsible for scientific preprocessing and produces the standard
  dataset used by every later component.
- After Component 02, rainfall is in millimetres and timestamps are timezone-aware UTC.
- The station key is the Bureau of Meteorology's unique internal `station_id`. Public
  station numbers are display metadata and are not unique identifiers.
- Never turn missing rainfall into zero unless the documented dry-versus-missing rule
  proves that the hour was observed and dry.
- No station, timestamp, raw file, or prediction may disappear silently. Record every
  exclusion or failure with its cause.
- Every method in a comparison must predict the exact target rows defined by Component
  03. Do not create private splits, targets, or random seeds.
- The inherited hand-exported gauge values are regression evidence only. They have
  time- and station-varying distortion and must not support scientific millimetre-based
  claims.
- Numerical fixes to inherited behaviour must be explicitly named and tested. Do not
  silently present a corrected result as an inherited reproduction.

## Component boundaries

Create these packages and configuration directories only when their owner begins work:

| Component | Implementation | Configuration |
|---|---|---|
| 01 — source data | `src/ingestion/` | `config/ingestion/` |
| 02 — standard dataset | `src/canonical/` | `config/canonical/` |
| 03 — study splits | `src/study/`, `src/reproducibility/` | `config/splits/`, `config/seeds.yaml` |
| 04 — graph samples | `src/graph/` | `config/graphs/` |
| 06 — model adapters | `src/models/` | `config/models/` |

An owner may use a clearly labelled stub or small test sample while an earlier component
is incomplete. They must not invent a private production format to bypass that dependency.
Confirm changes to shared Inputs/Outputs with the owner of the component that reads them.

`train_fused.py`, `config.yaml`, `adapters/legacy_sydney_gauge_export.py`,
`database/sydney_gauge_legacy/`, and `database/sydney_split_info.json` are transitional
regression paths. They are not production outputs for Components 01, 02, or 03. Reuse
small, tested functions where appropriate; do not extend the transitional path into the
target architecture.

## Implementation workflow

1. Preserve unrelated changes in the worktree.
2. Map the work to existing requirement identifiers before writing code.
3. Implement the smallest path that produces one prescribed output or failure mode.
4. Add executable tests for the relevant `TESTxx` requirements beside the change.
5. Validate inputs and fail with a concrete message rather than guessing.
6. Make generated outputs traceable to input versions, settings, code revision, and
   exclusions as required by the component contract.
7. Demonstrate the output to the downstream reviewer without private parsing or manual
   repair steps.
8. Update an existing requirements document only when implementation evidence exposes a
   genuine contract gap. Do not create extra planning documents for ordinary work.

Mark genuine uncertainty using the labels prescribed by the requirements:
`BLOCKED`, `TO VALIDATE`, or `UNKNOWN`. These labels do not permit silent placeholders or
invented scientific decisions.

## Keeping the documentation true

Two documents carry live project information, and each has exactly one job. Keeping them
separate is what stops either from going stale.

**`docs/CURRENT_STATE.md` — where the project is.** It is a continuously updated source of
truth, not a milestone report. Its own "How this document is maintained" section is the
authority; the rules in short:

- It names the **latest commit** the description is true of. Never a branch name — a
  branch is stale the moment it merges.
- On a feature branch, your copy is main's latest version plus the lines your feature
  makes untrue. Take main's version when you branch and again when you rebase, and edit
  only your own component's lines. Parallel work then merges as small paragraphs rather
  than competing rewrites.
- Update it in the same commit or pull request as the change it describes. A PR that makes
  it wrong and does not fix it is not finished.
- When a sprint closes, its state description moves to `docs/archive/`.

**`docs/ISSUES.md` — what is broken, missing or undecided.** It is the only place an issue
is recorded.

- Every issue gets a permanent `ISS-xx` id. Never renumber; close in place with the commit
  that closed it. Reference the id in branch names, commit messages and PR descriptions.
- Do not open a new issue instead of making a cheap fix. Defer on purpose, with a reason.
- If you find an issue written up in a component README, a handover note or a code
  comment, move it here and leave the id behind.

Other documents may cite an `ISS-xx` id. None of them — `CURRENT_STATE.md` included — may
restate an issue's severity, owner or status. Two copies means two answers and no way to
tell which is current.

Component READMEs under `docs/` describe how to run the component and what its outputs
mean. Under the Sprint 2 standard, a component does not ship without one.

## Git and repository safety

- Work on the branch assigned by the project lead. Component branches normally use
  `feature/01-source-data`, `feature/02-canonical-data`, and the corresponding numbered
  names from `integration/sydney-gauge-baseline`.
- A slash in a branch name is naming only; it does not establish branch ancestry.
- Do not merge, push, rewrite history, stage unrelated files, or delete another person's
  work unless the project lead explicitly requests it.
- Never use destructive resets or checkouts to clean a dirty worktree.
- Keep one component or shared-contract change per pull request where practical.
- Do not commit raw source data, credentials, checkpoints, experiment directories, or
  generated production datasets. The small per-component samples under `tests/data/` are
  the documented exception.

## Dependency management

- Use uv for project and development dependencies. `pyproject.toml` declares the direct
  dependencies and `uv.lock` records the exact resolved environment.
- Add and remove dependencies with `uv add` and `uv remove`; use `--dev` for development
  tools. Do not install project packages directly with pip.
- Treat `requirements.txt` and `requirements-dev.txt`, when present, as generated exports
  for non-uv consumers. Do not edit them by hand or use them as dependency authority.

## Required verification

Synchronise the locked environment, then run the current repository checks with:

```console
uv sync --locked
uv run --locked python -m pytest tests -q -rx
uv run --locked ruff check .
```

Also run the acceptance tests prescribed by the assigned component. The existing
expected failures document inherited defects; they are not acceptance criteria for new
components. Do not add or weaken an expected failure merely to obtain a green run.

Coverage floors apply: 80 % over the production packages and 90 % over the modules that
write files other work depends on — rule 10 in `docs/requirements/README.md` names them.
Subtask D07.7 wires the check into the commands above; until it lands, run
`uv run --locked python -m pytest tests -q --cov=src --cov-report=term-missing` after
`uv add --dev pytest-cov` and report the numbers for the modules you changed. Do not
lower a floor to pass it.

Before handoff:

- run `git diff --check`;
- check imports or compilation for changed Python modules;
- verify local Markdown links when documentation paths changed;
- report the exact tests run and their results;
- update `docs/CURRENT_STATE.md` if the change makes any line of it untrue, including the
  latest-commit header;
- record every remaining blocker or assumption in `docs/ISSUES.md` with an `ISS-xx` id,
  and state the ids here rather than describing them again; and
- identify the output that the downstream component should review.

Completion means the prescribed output, tests, documentation, and downstream contract
are all usable. A partial implementation or successful smoke run is not component
acceptance unless the component requirements explicitly say so.
