# Current state

**Latest commit:** `94c6e74` · **Sprint:** 2 · **Updated:** 2026-10-11

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

This is software-integration evidence. It is not a scientific result. The radar and
satellite tables have only been built from the committed offline samples, so real-archive
coverage is unmeasured (ISS-03, ISS-04), held-out-gauge independence is not established,
and the inherited hand export still may not be quoted in millimetres. The top data risk in
the standard tables is ISS-01.

The radar fetcher landed at `45c8dbb` (`src/ingestion/radar/`, NCI Rainfields3
`prcp-m15`, site 71). It answers the product, units and gauge-correction questions for
radar. Production will not download the archive: on Gadi it is a mounted directory and
D13 reads it in place, reusing the fetcher's reader and coordinate code; the fetcher's
download path serves samples and local tries. `src/canonical/radar_archive.py` (D13.2)
now reads site 71's daily zips into hourly node tables (ISS-30); a radar result still
needs Gadi access to run against real data, and D09's edges (D09.2) and runs (D09.4) to
actually produce one.

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
| D08 | Sydney gauge-only baseline | D08.1 and D08.2 merged, closing the code side of D08's three gaps. Split: `config/splits/sydney_5fold_blocked.yaml` hides two whole location clusters per fold (`fold_assignment: whole_clusters`), and its released `.json` over 127 real stations is committed; `sydney_5fold_round_robin.yaml` is a companion that differs only in dealing each cluster's gauges across the folds, as the first author's code does, and its `.json` is built locally, not committed. `src/study/summary.py` writes the split summary and fold map, and `docs/study/README.md` covers Component 03. Every method reads training and validation gauges at test time (ISS-07). `input_rainfall_scale: zscore_train` scales the rainfall input on each fold's training graph and records the numbers in the run's config (ISS-09). `config/graphs/gauge_{blocked,round_robin}.yaml` and `config/models/hgnn_{blocked,round_robin}.yaml` run each split end to end. Not yet done: D08.3's table, D08.5's correlation plot, D08.6's distance tool and D08.7's before/after measurement, which closes ISS-10 |
| D09 | Radar fusion | D09.2's cross-support half merged: each gauge links to its nine nearest sensor nodes, edges run sensor-to-gauge only unless `reverse_cross_support: true`, and the HGNN keeps a sensor layer's values across layers when nothing sends to it. Synthetic graph, training and checkpoint-replay tests pass in both modes. The eight-neighbour grid edges and the coincident-node weight are ISS-38. D09.4 and D09.5 need D08.3's gauge-only table and D13's tables built from the real archive (ISS-04); no real-data fusion result is claimed |
| D10 | Australia-wide | Not started, and deliberately last: it only means something once the Sydney baseline is trustworthy. Nobody has measured the national station count yet. Reads D08.6's distance tool per region |
| D11 | BRAIN paper report | Not started; scope TO CONFIRM |
| D13 | Source grids from NCI | All five subtasks merged, with the integration review's fixes on top. `config/canonical/grids.yaml` points `layers.radar.root` at site 71's folder and `layers.satellite.root` at the CRRPh archive; `src.canonical.grids.build()` reads both in place. Radar: four 15-minute `prcp-m15` files summed per hour across daily zips, one value per cell-hour as in the paper (ISS-36), cut to the same Sydney study box as satellite (857 nodes for site 71). Satellite: six 10-minute `crrph_intensity` samples averaged per hour, cropped from the full disk, with a `valid` column; an image with no value makes the cell-hour invalid (ISS-37). `grid_report.md` states product, units, site, gauge correction, coordinate system and time step from the files themselves, the hourly, crop and subsample rules, every unwritten hour with its cause, and radar's contributing-gauges cross-reference against Component 02's `stations.csv` (ISS-02, ISS-04). `tests/data/ingestion/{radar,satellite}/` hold real, complete-hour samples, and `python -m src.canonical.grids config/canonical/grids.yaml --sample` runs both readers with no Gadi access. Nothing has been run against the real archive yet |

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
| 03 | `src/study/`, `src/reproducibility/` | `config/splits/`, `config/seeds.yaml` | `docs/study/` |
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

Run on 2026-10-11 against the code at `94c6e74`, in a clean checkout, Windows 11,
Python 3.14.2, in the environment `uv sync --locked` builds (the D08 and D09.2 merges
leave `uv.lock` unchanged):

```console
uv run --locked python -m pytest tests -q -rx --cov=src --cov-report=term-missing
uv run --locked ruff check .
```

- `tests/`: **251 passed, 30 skipped, 3 failed**. The skips are the live-API acceptance
  checks (`BOM_NETWORK=1`, `NCI_NETWORK=1`) and the legacy-data tests, which skip in a
  checkout without `database/sydney_gauge_legacy/`. The three failures are all in
  `tests/test_experiment_tui.py` and fail the same way on `main` before these merges on
  this Windows machine; the earlier macOS run of `main` passed them.
- With `database/sydney_gauge_legacy/` present, the same code gives **271 passed,
  3 skipped, 7 xfailed, 3 failed**: the same three failures, and the seven expected
  failures that document the inherited export's defects, unchanged by the merges.
- Repository-wide Ruff: **665** findings (ISS-17), one fewer than `90b91ba` measured the
  same way; the merges add none.
- Coverage, with `--cov=src`: **52 %** over all of `src/`, which includes the inherited
  review-only code. Modules the merges change: `src/models/run.py` 80 %, `scaling.py`
  89 %, `baselines.py` 86 %, `modalities.py` 97 %, `models.py` 36 % (the inherited
  architectures count towards it); `src/study/split.py` 93 %, `summary.py` 100 %,
  `schema.py` 100 %; `src/graph/builder_new.py` 92 %, `gaugegraphnew.py` 84 %,
  `edge_map.py` 97 %. `run.py` is below its 90 % floor and `split.py` meets its own. The
  floors are not enforced yet (ISS-33).
- PyTorch and PyG emit Python 3.14 deprecation warnings. PyG also warns that radar and
  satellite nodes receive no updates; that is expected while edges run sensor-to-gauge
  only and the grid edges are missing (ISS-38).
- `scripts/nci/test_guards.py` is excluded from `tests/` and fails on a Windows checkout
  for a line-ending reason (ISS-15). It passes on Linux and macOS.
