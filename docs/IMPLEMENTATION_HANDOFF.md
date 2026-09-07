# Implementation handoff

**Branch:** `integration/sydney-gauge-baseline` · **Updated:** 2026-09-07

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

## Integration state (2026-09-07)

Reviewed on `integration/sydney-gauge-baseline` at `13ac827`. Suite: **29 passed,
3 failed, 2 skipped, 7 xfailed** — the three failures are deliberate and are described
in defect 1.

Work finished but not merged:

| Branch | Author | Contents |
|---|---|---|
| `feature/01-source-data` | Amos Watson | `740c6ba`, reconciling requirements doc 01 with the shipped fetcher. This is Component 02's owner discharging the reviewer role that doc 01 assigns them. It was filed on 01's branch rather than `feature/02-canonical-data`, which is why 02 looks empty. Its claims were checked against the code and hold |
| `feature/03-study-splits` | Kiran McLernon | 6 commits: `src/study/`, `src/reproducibility/`, `config/seeds.yaml`, `config/splits/sydney_5fold.yaml`, 13 tests. Also adds `pydantic` and `scikit-learn-stubs` to `pyproject.toml` |

`feature/02-canonical-data` points at the integration head with no commits of its own.

### Open defects

1. **`GaugeGraphNew` does not implement the paper's Eq. 1, and three tests now say so.**
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
   property directly, because the existing median check cannot detect either defect — a
   weight normalised by the wrong endpoint is still O(1). It fails on `nan` today, will
   fail at 86% once the guard is restored, and passes when the graph is built directed.

   **The zero guard is deliberately not applied here.** Component 04's owner is
   rebuilding this edge construction as a directed graph as of 2026-09-07, which fixes
   both defects, and a one-line change to the same function would only conflict. The
   three failures stand as the specification for that work.

2. **A partial `series.csv` silently drops stations from every later run.** Confirmed by
   experiment on 2026-09-07: `stations.json` held 184 stations, `series.csv` held 2 (left
   by an earlier `--limit` run), and station `49570554` APPIN (INGHAMS) TBRG was skipped
   with no index row, no report line, `report.md` claiming "stations with both series: 3"
   and "no usable series: 0", and exit code 0. `load_or_fetch_series` returns early
   whenever `series.csv` exists, with an empty `missing` map, and `run()` then skips
   unknown stations with a bare `continue`. This breaks the rule that no station may
   disappear silently, and it defeats the non-zero exit that `740c6ba`'s revised rule 3
   promises for a station lacking a required series. Owner: Component 01.

3. **The 02→03 seam will reject every eligible station.** `src/study/schema.py`'s
   `StationRecord` uses pydantic `strict=True`. An empty `exclusion_reason` read from
   `stations.csv` by pandas arrives as `nan`, not `""`, so validation fails for exactly
   the rows where `use_in_study` is true — verified against a synthetic two-row table.
   `strict=True` also rejects numpy scalars, so whoever writes the `stations.csv` reader
   must coerce empty strings and hand over Python types. Owners: Components 02 and 03
   together, since it is a contract question rather than a bug in either side.

4. **`ruff check .` reports 120 findings**, all in inherited `benchmarks/`, `models/`,
   `src/graph/`, `training/` and the root runners. Identical on every branch; not a
   regression from 01 or 03. Two sit in `tests/test_split_integrity.py`.

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
