# Implementation handoff

**Branch:** `integration/sydney-gauge-baseline` · **Updated:** 2026-09-01

Sequencing and blockers. The contracts themselves are in
[requirements/README.md](requirements/README.md) — read that first.

## Where the project actually is

- Both inherited codebases have been audited. The graph and model core runs a real
  Sydney gauge-only graph.
- The inherited hand-exported rainfall has time- and station-varying distortion. It is
  regression evidence only. **No result quoted in millimetres may come from it.**
- Everything therefore starts again from BoM source data: Component 01 downloads it,
  Component 02 makes it trustworthy, and the first quotable number comes from a run on
  Component 02 output.
- Radar and satellite files for 2022 now exist locally, so the NCI access blocker is
  gone. What replaces it is a provenance question — see below.

## Order of work

Components 01 and 02 are the critical path; nothing scientific is real until they land.
03, 04 and 06 can all start immediately against small stub inputs of the documented
shape, and should not wait.

| Component | Start now | Blocked until |
|---|---|---|
| 01 | BoM calls, config, disk layout, index | nothing |
| 02 | tips→hourly, UTC, dry-vs-missing, station table | real files from 01 |
| 03 | the whole thing, on a synthetic station table | nothing |
| 04 | point the loader at `database/standard/`; identity through batching; the four named fixes | 02 + 03 for real graphs |
| 06 | prediction schema, checker, idw and network_mean baselines | 04 for real training |

Radar and satellite work sits with whoever has capacity — realistically the person freed
by cutting the golden-fixture component. Gauge-only is the first milestone; radar is the
second; satellite is blocked.

## Open blockers

1. **Satellite node coordinates do not exist.** 726 nodes with no latitude or longitude
   cannot be placed on the map or connected to gauges. Satellite fusion is blocked until
   the file is found or regenerated. Owner: Component 01.
2. **Radar and satellite provenance is unknown.** Which product, which units, and whether
   the radar is gauge-corrected. If it was corrected using gauges that include our test
   stations, radar results leak and are not publishable. Owner: Component 01, by asking
   whoever produced the files.
3. **Radar and satellite cover 2022 only**, and radar is missing 120 hours. Gauges will
   cover 2021–2022. Confirm that fusion experiments are 2022-only.
4. **The paper's exact 75-station Sydney list is unavailable.** Reproducing that table is
   not an acceptance criterion.

## Repository boundaries

Create a package when its owner starts.

| Component | Code | Config |
|---|---|---|
| 01 | `src/ingestion/` | `config/ingestion/` |
| 02 | `src/canonical/` | `config/canonical/` |
| 03 | `src/study/` | `config/splits/`, `config/seeds.yaml` |
| 04 | `src/graphs/` | `config/graphs/` |
| 06 | `src/models/` | `config/models/` |

Useful inherited code lives in `train_fused.py`, `src/graph/gaugegraphnew.py`,
`models/gnn.py` and `training/logic_hetero.py`. Wrap and test it rather than rewriting
it. `adapters/legacy_sydney_gauge_export.py`, `database/sydney_gauge_legacy/` and
`database/sydney_split_info.json` are transitional regression paths, not production
outputs. See [LEGACY_CODE.md](LEGACY_CODE.md).

## What changed on 2026-09-01

The six requirements documents were rewritten from ~3,800 lines to ~850. Cut: acquisition
ids, progress state files, per-file metadata sidecars, hash-verification protocols,
dataset and sample version derivation, manifest files, the separate prediction-target
file, and the paired compatibility/corrected mode matrix in Component 04. Component 05
(golden fixture) was cut entirely; the small committed test sample it was meant to
produce is now a checklist item in each component. Kept: everything that decides what
data enters the study or how a result is defined.
