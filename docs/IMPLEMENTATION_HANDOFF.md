# Implementation handoff

**Audience:** all component owners and reviewers

**Branch:** `integration/sydney-gauge-baseline`

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive

This is the operational handoff, not another requirements specification. The complete
set of visible work packages and shared rules is in
[requirements/README.md](requirements/README.md). Each component document remains the
authority for its own `FRxx`, `OUTxx`, `TESTxx`, and `DOCSxx` items.

## Current technical position

- Both inherited pipelines have been audited.
- The retained graph/model core can run a genuine Sydney gauge-only graph without
  fabricated non-gauge nodes.
- The transitional Sydney export and imported split prove interface compatibility only.
- The inherited hand-exported rainfall values have time- and station-varying distortion;
  they cannot support scientific millimetre-based results.
- Production data must be regenerated from source-faithful Bureau of Meteorology (BoM)
  records. The canonical station key is BoM's internal `station_id`.
- Both inherited graph paths used graph-wide edge normalisation. Compatibility and
  corrected per-node behaviour must remain explicitly labelled.

The existing smoke runs prove runnability, not baseline validity. Detailed investigation
and run records are retained in the [historical archive](archive/README.md).

## Work order

All requirements are visible to everyone. Owners read their own document fully and the
input/output contracts of adjacent components.

| Component | Work that can start | Release dependency |
|---|---|---|
| [01 Collect source data](requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md) | Gauge settings, raw-data layout, source metadata, resume, and small validation queries | Full gauge retrieval needs approval; radar/satellite need NCI access |
| [02 Prepare standard data](requirements/02_CANONICAL_DATA_REQUIREMENTS.md) | Raw-input checks, required table columns, hourly addition, daily-total checking, station table, and versioning | Production dataset needs 01 gauge output |
| [03 Splits and seeds](requirements/03_STUDY_SPLIT_REQUIREMENTS.md) | Seed loader, schemas, algorithms, audit, and synthetic tests | Production split needs accepted 02 registry |
| [04 Graph samples](requirements/04_GRAPH_SAMPLE_REQUIREMENTS.md) | Interface extraction and compatibility/corrected regression tests | Production samples need accepted 02 and 03 files |
| [06 Train/write predictions](requirements/06_MODEL_ADAPTER_REQUIREMENTS.md) | Common Python interface, run layout, restart, and prediction checks against stubs | Acceptance needs 03 targets and 04 samples |
| [05 Small verified test data](requirements/05_GOLDEN_FIXTURE_REQUIREMENTS.md) | Selection/create tooling and offline system tests | Final truth needs accepted component outputs |

The first implementation target should be a narrow end-to-end gauge slice: acquire one
real source window, write its data and JSON provenance, update progress state, and test
resume and failure reporting.

## Repository boundaries

Create a package only when its owner starts implementation:

| Component | Code | Configuration |
|---|---|---|
| 01 | `src/ingestion/` | `config/ingestion/` |
| 02 | `src/canonical/` | `config/canonical/` |
| 03 | `src/study/` | `config/splits/`, `config/seeds.yaml` |
| 04 | `src/graphs/` | `config/graphs/` |
| 05 | `src/fixtures/` or `tools/fixtures/` | `config/fixtures/` |
| 06 | `src/models/adapters/` | `config/models/` |

Useful inherited core remains in `train_fused.py`, `src/graph/gaugegraphnew.py`,
`models/gnn.py`, and `training/logic_hetero.py`. Wrap and regression-test it; its current
interfaces are not the target architecture.

`adapters/legacy_sydney_gauge_export.py`, `database/sydney_gauge_legacy/`, and
`database/sydney_split_info.json` are transitional regression files. They must not
be accepted as component 01, 02, or 03 production outputs. See
[LEGACY_CODE.md](LEGACY_CODE.md) for the retained and removed inherited entry points.

## Owner workflow

1. Record the owner and reviewer at the top of the assigned component document.
2. Confirm required input/output files and fields with adjacent component owners.
3. Run the repository tests and preserve unrelated worktree changes.
4. Map the implementation slice to existing `FRxx`, `OUTxx`, and `TESTxx` identifiers.
5. Implement the smallest path that writes one real prescribed output.
6. Test provenance, resume/failure behaviour, and rejection of invalid inputs.
7. Have the next component's reviewer read the output without private parsing logic.
8. Revise a requirement only when implementation evidence exposes a real contract gap.

Use one component or contract change per branch or pull request. Keep generated data,
credentials, checkpoints, and experiment results out of Git, except for the explicitly
approved small golden fixture. Preserve labelled compatibility behaviour when correcting
an inherited numerical defect.

Singapore/NEA data handling and commercial microwave link support are not compatibility
behaviours. They are permanently outside project scope and must not be reintroduced.

Environment setup and the repository test command are in the root
[README.md](../README.md). No additional onboarding document is required.

## Decisions and blockers

Decided:

- Baseline period is 2021–2022.
- Canonical rainfall is millimetres and time is timezone-aware UTC.
- Station identity uses BoM internal `station_id`.
- Every method predicts the same row-level target index.
- Official file sets carry input identifiers or hashes, complete settings,
  code revision, exclusions, and failure information.

External blockers:

- Complete BoM baseline retrieval awaits approval.
- Radar and satellite product confirmation and access await National Computational
  Infrastructure (NCI) access.
- The paper's exact 75-station Sydney list is unavailable, so reproducing that table is
  not an acceptance criterion for the regenerated dataset.

These blockers do not prevent gauge ingestion design, schemas, split/graph algorithms,
the model boundary, or gauge-only fixture tooling.

## Component handoff is complete when

- required inputs, outputs, paths, and schemas are implemented;
- required tests pass, with external blockers explicitly recorded;
- outputs contain the prescribed source information, settings, identifiers, and exclusions;
- the runbook works from a fresh supported environment; and
- the next component's reviewer reads a real output without undocumented fixes.

No official result may depend on the distorted inherited rainfall export.
