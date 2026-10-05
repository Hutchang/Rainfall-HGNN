# Current state

**Latest commit:** `1b786a2` · **Sprint:** 2 · **Updated:** 2026-10-05

Where the project actually is right now, and what the current sprint is doing. Read this
before picking up work; it is the one place that is kept true.

Open bugs and gaps are **not** here. They are in
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

The radar fetcher landed at `45c8dbb` (`src/ingestion/radar/`, NCI Rainfields3
`prcp-m15`, site 71). It answers the product, units and gauge-correction questions for
radar. Production will not download the archive: on Gadi it is a mounted directory and
D13 reads it in place, reusing the fetcher's reader and coordinate code; the fetcher's
download path serves samples and local tries. Nothing downstream can read the archive
files yet (ISS-30), so no radar result is possible until D13.2 lands.

Researchers run the pipeline from a terminal app, `src/experiments/tui.py`, without
editing code. It creates or clones an experiment folder with its own copy of the configs,
edits those configs, runs any range of stages from 01 to 06, and streams the live output.
The Results tab shows each fold's train and validation loss as training runs. Below the
chart, a summary table gives each fold's best and latest losses and its early-stopping
state, followed by the final RMSE, MAE and correlation once the run finishes. **Save graph
PNG** writes the selected fold's chart to the experiment's `results/` folder. To support
this, Component 06's `history.csv` now also records the best validation loss so far, the
count of epochs without improvement, the remaining patience and an early-stopping status,
and prints one line per epoch. The Gadi experiment list on the Experiments tab is a
disabled placeholder; it does not connect to NCI. The
[workbench guide](experiments/README.md) covers its use.

No real Sydney build output exists in this checkout — `database/raw/`,
`database/standard/` and `database/graphs/` are gitignored and absent. Everything
verified below runs against committed offline samples and synthetic fixtures.

## Sprint 2

Sprint 1 is closed. Its component contracts are archived in
[archive/sprint1/](archive/sprint1/) and describe what each component implements today,
but they are reference, not current assignments.

Sprint 2 is one week, six people, organised as seven deliverables, D07 to D13, that run
at the same time; the numbering continues from the Sprint 1 components.
The requirements are in [requirements/README.md](requirements/README.md): that page holds
the rule for what counts as work, the three open paper-versus-code gaps, the rules, and a link to
each deliverable's document. State per deliverable:

| Deliverable | Name | State |
|---|---|---|
| D07 | Experiment runner | Not started. The terminal app above already gives each experiment its own folder, config copies and results; D07's runner, provenance record and registry are still to come. Today only Component 06 writes one folder per run with the fingerprints of its inputs; every component before it writes to a fixed path, and 02 overwrites silently (ISS-28, ISS-29, ISS-32) |
| D08 | Sydney gauge-only baseline | Not started. Model settings already match the paper; the split rule (ISS-10), which gauges each method may read (ISS-07) and input scaling (ISS-09) do not - these are D08's three gaps. Edge direction (ISS-25) is already applied in code behind `both_directions`; it needs only a rebuild on real data. D08 also owns the held-out distance measurement that says whether its own table can be trusted (D08.6, D08.7), and closes ISS-10 with it |
| D09 | Radar fusion | Not started. Now the science only: grid edges (D09.2), then the G+R and G+R+S runs on D13's tables. Builds on gauge-corrected radar, the same kind the paper used, with the caveat stated on every result |
| D10 | Australia-wide | Not started, and deliberately last: it only means something once the Sydney baseline is trustworthy. Nobody has measured the national station count yet. Reads D08.6's distance tool per region |
| D11 | BRAIN paper report | Not started; scope TO CONFIRM |
| D13 | Source grids from NCI | Not started. Reads radar and satellite in place on Gadi and writes the grid tables; resolves ISS-02, ISS-03, ISS-04, ISS-30. Gadi paths and the satellite product are TO CONFIRM |

Open work that belongs to a component rather than a deliverable:

| Component | Open |
|---|---|
| 01 — source data | Satellite registration. Provenance: ISS-03, ISS-04 (ISS-02 now answered for radar by the fetcher; the leak caveat belongs to D09) |
| 02 — standard tables | False-zero detection ISS-01, the April–May 2021 gap ISS-06, excluded-download reporting ISS-16, stale README table ISS-20 |
| 03 — study splits | Summary and plot ISS-18 and component README ISS-19 — both delivered by D08 subtask D08.1 |
| 04 — graph samples | Baseline fingerprint test ISS-14; positional-encoding sign ISS-27, read by hand 2026-09-20 and awaiting an experiment before anything changes |
| 06 — model adapters | IDW exponent ISS-08 |

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
| D07 | `src/experiments/` | — | `docs/experiments/` |

Two inherited modules have been adopted into a component and are now live code, not
reference: `src/models/models.py` (the retained architectures, Component 06) and
`src/graph/gaugegraphnew.py` (the gauge graph, Component 04). Both also have
`train_fused.py` as a second caller, so a change to either moves the inherited baseline
as well — see ISS-14. `adapters/legacy_sydney_gauge_export.py`, `database/sydney_gauge_legacy/` and
`database/sydney_split_info.json` are transitional regression paths, not production
outputs. See [LEGACY_CODE.md](LEGACY_CODE.md).

## Verification

Run on 2026-10-05 on `1b786a2` plus the uncommitted Results-tab and training-history
changes, macOS, Python 3.14.6:

```console
uv run --locked python -m pytest tests -q -rx
uv run --locked ruff check .
```

- `tests/`: **182 passed, 3 skipped, 7 xfailed**. The skips are the live-API acceptance
  checks; set `BOM_NETWORK=1` (two gauge checks) or `NCI_NETWORK=1` (one radar check) to
  run them. A checkout without `database/sydney_gauge_legacy/` skips the legacy-data
  tests as well, which is expected.
- Repository-wide Ruff: **684** findings (ISS-17). The files changed for the Results tab
  add none; the findings in `src/models/run.py` are long lines that were there before.
- Coverage, with `--cov=src`: **50 %** over all of `src/`, which includes the inherited
  review-only code. Modules changed most recently: `src/experiments/results.py` 98 %,
  `workspace.py` 94 %, `chart.py` 96 %, `tui.py` 87 %, `src/models/trainer.py` 94 %,
  `src/models/run.py` 79 %. The floors are not enforced yet (ISS-33).
- PyTorch and PyG emit Python 3.14 deprecation warnings.
- `scripts/nci/test_guards.py` is excluded from `tests/` and fails on a Windows checkout
  for a line-ending reason (ISS-15). It passes on Linux and macOS.
