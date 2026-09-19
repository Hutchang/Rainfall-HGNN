# Current state

**Latest commit:** `d337aa6` · **Sprint:** 2 · **Updated:** 2026-09-19

Where the project actually is right now, and what the current sprint is doing. Read this
before picking up work; it is the one place that is kept true.

Open bugs, gaps and unresolved decisions are **not** here. They are in
[ISSUES.md](ISSUES.md) and are referenced below by their `ISS-xx` id only.

## How this document is maintained

This document is a continuously updated source of truth, not a handoff note written once
per milestone. Five rules keep it that way.

1. **It names the latest commit, never a branch.** A branch name is stale the moment the
   work merges; a commit hash stays true forever. Update the header with the commit the
   description is accurate as of.
2. **On a feature branch, your copy is main's latest version plus your own component's
   changes.** Take main's version when you branch and again when you rebase, then edit
   only the lines your feature makes untrue. Two people working in parallel therefore
   merge two small paragraphs, not two competing rewrites.
3. **It records state, not issues.** Anything broken, missing, deferred or undecided goes
   to [ISSUES.md](ISSUES.md) and gets an id. Cite the id here if the state depends on it.
   Never restate an issue's severity, owner or status in this document — there would then
   be two answers and no way to tell which is current.
4. **It records the present, not history.** When a sprint closes, its state description
   moves to `docs/archive/`. Sprint 1's is in
   [archive/sprint1/HANDOFF_HISTORY.md](archive/sprint1/HANDOFF_HISTORY.md).
5. **It is updated in the same commit or pull request as the change it describes.** A PR
   that makes this document wrong and does not fix it is not finished.

## Where the project is

The gauge-only Sydney pipeline runs end to end. Components 01, 02, 03, 04 and 06 are
implemented, tested and connected on `main`: raw BoM responses → standard tables → a
released split → graph samples → one prediction table covering HGNN, IDW and network
mean. The [pipeline guide](graphs/README.md) is the runnable description of that chain.

This is software-integration evidence. It is not a scientific result. Radar and satellite
provenance, units and coverage are unanswered (ISS-02, ISS-03, ISS-04), held-out-gauge
independence is not established, and the inherited hand export still may not be quoted in
millimetres. The top data risk in the standard tables is ISS-01.

No real Sydney build output exists in this checkout — `database/raw/`,
`database/standard/` and `database/graphs/` are gitignored and absent. Everything
verified below runs against committed offline samples and synthetic fixtures.

## Sprint 2

Sprint 1 is closed. Its component contracts are archived in
[archive/sprint1/](archive/sprint1/) and describe what each component implements today,
but they are reference, not current assignments.

In progress:

| Work | Owner | State |
|---|---|---|
| Sprint 2 requirements set | Rick | Being written. `docs/requirements/README.md` is empty in the meantime — ISS-21 |
| Experiment runner | Rick | Being designed. `src/models/run.py` already writes each run's effective config, seeds, versions and input hashes into its own `experiments/` directory; the Sprint 2 runner generalises that across components so any result traces to the settings that produced it |
| Per-component README standard | all owners | No PR until the component has its own README under `docs/`. Component 03 does not meet this yet — ISS-19 |

Carried into Sprint 2 as open work, by owner:

| Component | Open |
|---|---|
| 01 — source data | Radar/satellite registration and index rows. Provenance answers: ISS-02, ISS-03, ISS-04. Region scope: ISS-11, ISS-22 |
| 02 — standard tables | False-zero detection ISS-01, the April–May 2021 gap ISS-06, excluded-download reporting ISS-16, stale README table ISS-20 |
| 03 — study splits | Summary and plot ISS-18, component README ISS-19, spatial blocking ISS-10 |
| 04 — graph samples | Baseline fingerprint test ISS-14, normalization metadata ISS-09, donor policy ISS-07 |
| 06 — model adapters | Donor policy ISS-07, IDW exponent ISS-08, normalization metadata ISS-09 |

Gauge-only is the first milestone, radar the second, satellite blocked on ISS-03.

## Repository boundaries

Each component owns its implementation and configuration directories. Create a package
when its owner starts work.

| Component | Code | Config | Docs |
|---|---|---|---|
| 01 | `src/ingestion/` | `config/ingestion/` | `docs/ingestion/` |
| 02 | `src/canonical/` | `config/canonical/` | `docs/canonical/`, `docs/grids/` |
| 03 | `src/study/`, `src/reproducibility/` | `config/splits/`, `config/seeds.yaml` | — (ISS-19) |
| 04 | `src/graph/` | `config/graphs/` | `docs/graphs/` |
| 06 | `src/models/` | `config/models/` | `docs/models/`, `docs/nci/` |

Two inherited modules have been adopted into a component and are now live code, not
reference: `src/models/models.py` (the retained architectures, Component 06) and
`src/graph/gaugegraphnew.py` (the gauge graph, Component 04). Both also have
`train_fused.py` as a second caller, so a change to either moves the inherited baseline
as well — see ISS-14. `adapters/legacy_sydney_gauge_export.py`, `database/sydney_gauge_legacy/` and
`database/sydney_split_info.json` are transitional regression paths, not production
outputs. See [LEGACY_CODE.md](LEGACY_CODE.md).

## Verification

Run on 2026-09-19 at `d337aa6`, Windows, Python 3.14:

```console
uv run --locked python -m pytest tests -q -rx
uv run --locked ruff check .
```

- `tests/`: **153 passed, 2 skipped, 7 xfailed**. Both skips are the live-API acceptance
  checks; set `BOM_NETWORK=1` to run them. A checkout without
  `database/sydney_gauge_legacy/` skips the legacy-data tests as well, which is expected.
- Repository-wide Ruff: **686** findings (ISS-17), down from 692 when `train_homo.py` and
  `train_st.py` were deleted. Four of them now sit in `src/models/models.py` — they came
  with the inherited architectures when that file moved under Component 06 and are not
  new. Every module a component actually wrote passes focused checks.
- PyTorch and PyG emit Python 3.14 deprecation warnings.
- `scripts/nci/test_guards.py` is excluded from `tests/` and fails on a Windows checkout
  for a line-ending reason (ISS-15). It passes on Linux and macOS.
