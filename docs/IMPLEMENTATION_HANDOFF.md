# Implementation handoff

**Current branch:** `integration/sydney-gauge-baseline` (`217748c`) · **Updated:** 2026-09-18

Known open defects and knowingly-deferred issues are tracked in
[ISSUES.md](ISSUES.md), not here. This document records sequencing, repository
boundaries and integration state only.

Sequencing and blockers. The contracts themselves are in
[requirements/README.md](requirements/README.md) — read that first.

## Current integration candidate

- Component 06 model training (`dbdb611`) and grid adapters (`48636ff`, `488a986`)
  are merged into this candidate branch.
- The 02→03→04→06 seam from `1aff5e2` has been reconciled against the newer integration
  implementations rather than replacing them with the stale pipeline snapshots.
- Component 04 retains `config/graphs/`, directed receiving-node-normalized gauge
  edges, finite coincident-coordinate weights, stable identities and UTC timestamps.
- Component 03 retains deterministic station ordering, exclusion reporting and strict
  non-empty roles, while released split files are now protected from replacement.
- The complete synthetic pipeline writes all five graph folds, trains HGNN, IDW and
  network mean on identical targets, and exactly replays checkpoint predictions.
- Verification on 2026-09-18: **133 passed, 29 skipped** in `tests/`; focused Ruff and
  compilation checks pass for the reconciled integration modules. Repository-wide Ruff
  reports **692** pre-existing findings. PyTorch/PyG emit Python 3.14 deprecation
  warnings.
- `scripts/nci/test_guards.py` passes on Linux/macOS but **all 8 fail on a Windows
  checkout**: the repository has no `.gitattributes`, so Git converts the `.sh`/`.pbs`
  scripts to CRLF and Bash reads `set -o pipefail<CR>`. The main `tests/` suite excludes
  these, so this does not show up in the numbers above.
- This is software-integration evidence only. Radar/satellite provenance, units,
  coverage and held-out-gauge independence remain scientific blockers.

## Historical state before this candidate (2026-09-07)

- Both inherited codebases have been audited. The graph and model core runs a real
  Sydney gauge-only graph.
- The inherited hand-exported rainfall has time- and station-varying distortion. It is
  regression evidence only. **No result quoted in millimetres may come from it.**
- Everything therefore starts again from BoM source data: Component 01 downloads it,
  Component 02 makes it trustworthy, and the first quotable number comes from a run on
  Component 02 output.
- Radar and satellite files for 2022 now exist locally, so the NCI access blocker is
  gone. What replaces it is a provenance question — see below.
- **Component 01's gauge path has shipped** and is merged (PR #3, `13ac827`). Re-verified
  on 2026-09-07: offline tests pass, both live-API acceptance checks pass, and a
  two-station end-to-end run fetched 52 files, resumed with zero requests, and produced a
  byte-stable index. Radar/satellite registration and the provenance answers are the
  remaining 01 scope.
- **Component 02 has not started.** `src/canonical/` and `config/canonical/` have never
  existed on any ref. There are no standard tables, so nothing downstream can yet be
  quoted in millimetres.
- **Component 03 exists as a library**, unmerged on `feature/03-study-splits`: spatial
  clustering, test-fold construction and role assignment, with 13 passing tests. It has no
  CLI, no JSON writer, no `stations.csv` reader and no summary, so it does not yet produce
  a split file.

## Order of work

Components 01 and 02 are the critical path; nothing scientific is real until they land.
03, 04 and 06 can all start immediately against small stub inputs of the documented
shape, and should not wait.

| Component | Start now | Blocked until |
|---|---|---|
| 01 | radar/satellite registration and index rows; the provenance answers below | nothing — the gauge fetcher has shipped |
| 02 | tips→hourly, UTC, dry-vs-missing, station table | nothing — real files can be fetched today, and `tests/data/ingestion/` runs offline |
| 03 | the CLI, the JSON writer, the `stations.csv` reader, the summary and plot | nothing |
| 04 | point the loader at `database/standard/`; identity through batching; the four named fixes | 02 + 03 for real graphs |
| 06 | prediction schema, checker, idw and network_mean baselines | 04 for real training |

Radar and satellite work sits with whoever has capacity — realistically the person freed
by cutting the golden-fixture component. Gauge-only is the first milestone; radar is the
second; satellite is blocked.

## Historical integration state (2026-09-07)

Reviewed on `integration/sydney-gauge-baseline`, now at `3c2a450`. Suite: **29 passed,
2 failed, 2 skipped, 7 xfailed** — both failures are deliberate and are described in
defect 1.

Work finished but not merged:

| Branch | Author | Contents |
|---|---|---|
| `feature/01-source-data` | Amos Watson | `740c6ba`, reconciling requirements doc 01 with the shipped fetcher. This is Component 02's owner discharging the reviewer role that doc 01 assigns them. It was filed on 01's branch rather than `feature/02-canonical-data`, which is why 02 looks empty. Its claims were checked against the code and hold |
| `feature/03-study-splits` | Kiran McLernon | 6 commits: `src/study/`, `src/reproducibility/`, `config/seeds.yaml`, `config/splits/sydney_5fold.yaml`, 13 tests. Also adds `pydantic` and `scikit-learn-stubs` to `pyproject.toml` |

`feature/02-canonical-data` points at the integration head with no commits of its own.

### Sprint 1 defects — all three resolved

Kept for the reasoning trail. Nothing below is still open; new issues belong in
[ISSUES.md](ISSUES.md).

1. **`GaugeGraphNew` did not implement the paper's Eq. 1 — fixed.** `gaugegraphnew.py`
   now builds an `nx.DiGraph` (line 144) and guards `1 / dist` with the `1e6` cap
   (lines 156, 358), so both sub-defects below are closed. The inherited-baseline
   fingerprint test is still owed and is tracked in [ISSUES.md](ISSUES.md).
   `65237ba` states the per-node normalisation fix was made "in both niramay and
   integration", but only the integration side was committed. The `gauge_graph_cls`
   fixture imported niramay's unpatched `GaugeGraph` from the `../COMP3888-niramay`
   worktree, so the fix was never exercised. The fixture has been repointed at
   `GaugeGraphNew` in `src/graph/gaugegraphnew.py`, the class
   `04_GRAPH_SAMPLE_REQUIREMENTS.md:26` names as the Component 04 contract, and that
   exposed two real defects:

   - **`1 / dist` is unguarded** (`gaugegraphnew.py:140`). The inherited metadata holds
     3 coincident coordinate pairs, giving `dist == 0` -> `inf` -> `nan` for 6 of 982
     edge weights, under *both* normalisation modes. The inherited class caps this at
     `1e6` (`graph/gauge_graph.py:133`), as do the standalone tests in
     `tests/test_split_integrity.py:136,174`; the rewrite dropped the guard.
   - **Edges are stored in an undirected `nx.Graph`** (`gaugegraphnew.py:123`). For a
     mutual k-NN pair the second `add_edge` overwrites the first's weight, and
     `ToUndirected()` (lines 70-72) then mirrors the survivor, so both directions carry a
     weight normalised by one endpoint's neighbourhood. Measured on the real graph:
     687 of 940 directed k-NN pairs are mutual, and only 161 of 188 nodes (86%) have a
     strongest incoming edge of 1.0, the weakest at 0.629. Eq. 1 requires 188 of 188.

   `test_per_node_normalisation_scales_every_node_by_its_own_neighbourhood` asserts the
   property directly, because a median check cannot detect either defect — a weight
   normalised by the wrong endpoint is still O(1). It fails on `nan` today, will
   fail at 86% once the guard is restored, and passes when the graph is built directed.

   **The zero guard is deliberately not applied here.** Component 04's owner is
   rebuilding this edge construction as a directed graph as of 2026-09-07, which fixes
   both defects, and a one-line change to the same function would only conflict. The two
   failures stand as the specification for that work.

   **Still owed: the inherited-baseline fingerprint.**
   `04_GRAPH_SAMPLE_REQUIREMENTS.md:146-149` requires a regression test pinning node
   counts, edge counts and weights against a known `train_fused.py` run, so that a change
   to shared graph code cannot move the inherited baseline unnoticed. The test that used
   to sit here only asserted `median(weights) < 1e-3`, which the directed rebuild would
   still satisfy while changing the edge count from 598 to 940 — so it gave no real
   protection and was removed on 2026-09-07. The fingerprint doc 04 asks for has not been
   written. Owner: Component 04.

2. **A partial `series.csv` silently dropped stations — fixed 2026-09-10.** The series
   lookup is one request per station with no per-station file to mark it done, so
   `series.csv` was used as a single completion flag for every station. Any station
   missing from it — one transient `HTTP 504` during the lookup was enough — was skipped
   by `run()` with a bare `continue` on every later run: no index row, no log line, no
   report entry, and `exit 0`, because the in-memory failure map was rebuilt empty from
   the cache. `report.md` then recomputed "stations with both series" as
   `len(stations) - len(no_series)` and claimed the full count. Confirmed by experiment
   on 2026-09-07: station `49570554` APPIN (INGHAMS) TBRG was present in `stations.json`
   and absent from everything else, with the run reporting success.

   `series.csv` now caches only stations where every required series was found, and any
   station missing from it is looked up again, so a transient failure self-heals on the
   next run at a cost of one request per unresolved station. A station that still has no
   series ids gets an index row per expected file with `status = "skipped"` and the cause
   in `error`, so it cannot leave `index.csv` between runs; Component 02 selects
   `status == "ok"` and is unaffected. `report.md` separates transient failures ("will
   retry on rerun") from permanent ones ("no required series") so the reader knows
   whether a rerun or a decision is needed, and the run still exits non-zero while any
   station is unresolved. `series.csv` keeps its four documented columns and is only
   rewritten when a lookup resolved something.

   Verified: the five new tests in `tests/test_ingestion_gauge.py` fail against the
   previous implementation and pass against this one, the nine existing tests pass
   unchanged, and a live three-station run now fetches APPIN's 26 files instead of
   skipping it, with a rerun making no requests and leaving `series.csv` byte-identical.

3. **The 02→03 seam rejected every eligible station — fixed.** `read_station_records`
   (`src/study/split.py:47`) reads `stations.csv` with the `csv` module and
   `StationRecord.model_validate_strings`, so pandas never turns an empty
   `exclusion_reason` into `nan` and no numpy scalars reach the strict model. Original
   report: `src/study/schema.py`'s
   `StationRecord` uses pydantic `strict=True`. An empty `exclusion_reason` read from
   `stations.csv` by pandas arrives as `nan`, not `""`, so validation fails for exactly
   the rows where `use_in_study` is true — verified against a synthetic two-row table.
   `strict=True` also rejects numpy scalars, so whoever writes the `stations.csv` reader
   must coerce empty strings and hand over Python types. Owners: Components 02 and 03
   together, since it is a contract question rather than a bug in either side.

### Open defects — deferred (non-essential right now)

Understood, don't break functionality or corrupt output, and are being knowingly left for
later. Revisit if priorities/deliverables are otherwise complete.

1. **`ruff check .` reports 692 findings** (the "120" recorded here previously counted a
   narrower rule set), all in inherited `benchmarks/`, `models/`, `src/graph/`,
   `training/` and the root runners. Identical on every branch; not a regression from any
   component. Two sit in `tests/test_split_integrity.py`.
2. **Failed/unread raw downloads are excluded from `stations.csv`/`days.csv` correctly but
   not reported.** A station whose coverage looks low can't be told apart from "genuinely
   dry" without manually cross-checking `index.csv`. No wrong data results — this is an
   audit-trail gap, not a correctness bug (shared rule 7, "say what you dropped"). Deferred
   under time constraints. Owner: Component 02.

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
| 04 | `src/graph/` | `config/graphs/` |
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
