# Sprint 1 handoff history

Historical record, lifted out of `docs/IMPLEMENTATION_HANDOFF.md` on 2026-09-19 when that
document became [CURRENT_STATE.md](../../CURRENT_STATE.md) and was cut back to the present
sprint only.

Nothing here is current. It is kept for the reasoning trail: which defects Sprint 1 found,
how they were diagnosed, and why each fix was made the way it was. Anything still open was
moved to [ISSUES.md](../../ISSUES.md) and carries an `ISS-xx` id there.

---

## Integration candidate as it stood on 2026-09-18

- Component 06 model training (`dbdb611`) and grid adapters (`48636ff`, `488a986`)
  were merged into the candidate branch.
- The 02→03→04→06 seam from `1aff5e2` was reconciled against the newer integration
  implementations rather than replacing them with the stale pipeline snapshots.
- Component 04 retained `config/graphs/`, directed receiving-node-normalized gauge
  edges, finite coincident-coordinate weights, stable identities and UTC timestamps.
- Component 03 retained deterministic station ordering, exclusion reporting and strict
  non-empty roles, while released split files became protected from replacement.
- The complete synthetic pipeline wrote all five graph folds, trained HGNN, IDW and
  network mean on identical targets, and exactly replayed checkpoint predictions.
- Verification on 2026-09-18: **133 passed, 29 skipped** in `tests/`; focused Ruff and
  compilation checks passed for the reconciled integration modules. Repository-wide Ruff
  reported **692** pre-existing findings. PyTorch/PyG emitted Python 3.14 deprecation
  warnings.

## State before that candidate (2026-09-07)

- Both inherited codebases had been audited. The graph and model core ran a real
  Sydney gauge-only graph.
- The inherited hand-exported rainfall has time- and station-varying distortion. It is
  regression evidence only. **No result quoted in millimetres may come from it.**
- Everything therefore started again from BoM source data: Component 01 downloads it,
  Component 02 makes it trustworthy, and the first quotable number comes from a run on
  Component 02 output.
- Radar and satellite files for 2022 existed locally, so the NCI access blocker was
  gone. What replaced it is a provenance question — now ISS-02.
- **Component 01's gauge path shipped** and was merged (PR #3, `13ac827`). Re-verified
  on 2026-09-07: offline tests passed, both live-API acceptance checks passed, and a
  two-station end-to-end run fetched 52 files, resumed with zero requests, and produced a
  byte-stable index.
- **Component 02 had not started.** `src/canonical/` and `config/canonical/` had never
  existed on any ref.
- **Component 03 existed as a library**, unmerged on `feature/03-study-splits`: spatial
  clustering, test-fold construction and role assignment, with 13 passing tests. It had no
  CLI, no JSON writer, no `stations.csv` reader and no summary.

Reviewed on `integration/sydney-gauge-baseline`, then at `3c2a450`. Suite: **29 passed,
2 failed, 2 skipped, 7 xfailed** — both failures were deliberate and are described in
defect 1 below.

Work finished but not merged at that point:

| Branch | Author | Contents |
|---|---|---|
| `feature/01-source-data` | Amos Watson | `740c6ba`, reconciling requirements doc 01 with the shipped fetcher. This was Component 02's owner discharging the reviewer role that doc 01 assigns them. It was filed on 01's branch rather than `feature/02-canonical-data`, which is why 02 looked empty. Its claims were checked against the code and held |
| `feature/03-study-splits` | Kiran McLernon | 6 commits: `src/study/`, `src/reproducibility/`, `config/seeds.yaml`, `config/splits/sydney_5fold.yaml`, 13 tests. Also added `pydantic` and `scikit-learn-stubs` to `pyproject.toml` |

`feature/02-canonical-data` pointed at the integration head with no commits of its own.

## Sprint 1 defects — all three resolved

1. **`GaugeGraphNew` did not implement the paper's Eq. 1 — fixed.** `gaugegraphnew.py`
   now builds an `nx.DiGraph` (line 144) and guards `1 / dist` with the `1e6` cap
   (lines 156, 358), so both sub-defects below are closed. The inherited-baseline
   fingerprint test is still owed; it is ISS-14.
   `65237ba` stated the per-node normalisation fix was made "in both niramay and
   integration", but only the integration side was committed. The `gauge_graph_cls`
   fixture imported niramay's unpatched `GaugeGraph` from the `../COMP3888-niramay`
   worktree, so the fix was never exercised. The fixture was repointed at
   `GaugeGraphNew` in `src/graph/gaugegraphnew.py`, the class
   `04_GRAPH_SAMPLE_REQUIREMENTS.md:26` names as the Component 04 contract, and that
   exposed two real defects:

   - **`1 / dist` was unguarded** (`gaugegraphnew.py:140`). The inherited metadata holds
     3 coincident coordinate pairs, giving `dist == 0` -> `inf` -> `nan` for 6 of 982
     edge weights, under *both* normalisation modes. The inherited class caps this at
     `1e6` (`graph/gauge_graph.py:133`), as do the standalone tests in
     `tests/test_split_integrity.py:136,174`; the rewrite dropped the guard.
   - **Edges were stored in an undirected `nx.Graph`** (`gaugegraphnew.py:123`). For a
     mutual k-NN pair the second `add_edge` overwrote the first's weight, and
     `ToUndirected()` (lines 70-72) then mirrored the survivor, so both directions carried
     a weight normalised by one endpoint's neighbourhood. Measured on the real graph:
     687 of 940 directed k-NN pairs were mutual, and only 161 of 188 nodes (86%) had a
     strongest incoming edge of 1.0, the weakest at 0.629. Eq. 1 requires 188 of 188.

   `test_per_node_normalisation_scales_every_node_by_its_own_neighbourhood` asserts the
   property directly, because a median check cannot detect either defect — a weight
   normalised by the wrong endpoint is still O(1).

   The zero guard was deliberately not applied at the time: Component 04's owner was
   rebuilding the edge construction as a directed graph as of 2026-09-07, which fixed
   both defects, and a one-line change to the same function would only have conflicted.
   The two failures stood as the specification for that work.

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

   Verified: the five new tests in `tests/test_ingestion_gauge.py` failed against the
   previous implementation and passed against this one, the nine existing tests passed
   unchanged, and a live three-station run fetched APPIN's 26 files instead of
   skipping it, with a rerun making no requests and leaving `series.csv` byte-identical.

3. **The 02→03 seam rejected every eligible station — fixed.** `read_station_records`
   (`src/study/split.py:47`) reads `stations.csv` with the `csv` module and
   `StationRecord.model_validate_strings`, so pandas never turns an empty
   `exclusion_reason` into `nan` and no numpy scalars reach the strict model. Original
   report: `src/study/schema.py`'s `StationRecord` uses pydantic `strict=True`. An empty
   `exclusion_reason` read from `stations.csv` by pandas arrives as `nan`, not `""`, so
   validation failed for exactly the rows where `use_in_study` is true — verified against
   a synthetic two-row table. `strict=True` also rejects numpy scalars, so whoever writes
   the `stations.csv` reader must coerce empty strings and hand over Python types.
   Owners: Components 02 and 03 together, since it was a contract question rather than a
   bug in either side.

## Sprint 1 order of work

Components 01 and 02 were the critical path; nothing scientific was real until they
landed. 03, 04 and 06 could all start immediately against small stub inputs of the
documented shape.

| Component | Start now | Blocked until |
|---|---|---|
| 01 | radar/satellite registration and index rows; the provenance answers | nothing — the gauge fetcher has shipped |
| 02 | tips→hourly, UTC, dry-vs-missing, station table | nothing — real files can be fetched today, and `tests/data/ingestion/` runs offline |
| 03 | the CLI, the JSON writer, the `stations.csv` reader, the summary and plot | nothing |
| 04 | point the loader at `database/standard/`; identity through batching; the four named fixes | 02 + 03 for real graphs |
| 06 | prediction schema, checker, idw and network_mean baselines | 04 for real training |

## What changed on 2026-09-01

The six requirements documents were rewritten from ~3,800 lines to ~850. Cut: acquisition
ids, progress state files, per-file metadata sidecars, hash-verification protocols,
dataset and sample version derivation, manifest files, the separate prediction-target
file, and the paired compatibility/corrected mode matrix in Component 04. Component 05
(golden fixture) was cut entirely; the small committed test sample it was meant to
produce became a checklist item in each component. Kept: everything that decides what
data enters the study or how a result is defined.
