# 05 — Small verified test dataset and offline system test

**Owner:** _unassigned_ · **Reviewers:** one owner from every component
**Status:** selection tooling and the offline test harness can start now; final expected
values wait for accepted gauge outputs from components 01–04 and the Component 06 interface

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive
**Scope:** small 2021–2022 Greater Sydney evidence; never a scientific benchmark

## 1. Summary

This component creates a small, committed, externally verified dataset and the offline
tests that exercise the whole system. It lets a teammate understand and run the system
on a laptop in minutes without BoM, NCI, the untracked legacy database, or a GPU.

“Golden fixture” is testing terminology for a small input with reviewed expected
answers. Here, its rainfall values are checked against independent BoM records and saved
with metadata showing where they came from. It does not mean its stations or metric
ranges estimate model quality. The fixture proves file formats, transformations, target
hiding, identity, and component handoff.

Responsibility starts with accepted files from the other components and ends with a
versioned fixture directory, expected results, deliberate-error tests, and one
documented continuous-integration (CI) command. CI means the automated tests run when
the team submits a code change. This component does not redefine earlier algorithms or
approve scientific performance claims.

## 2. System Position

```text
accepted outputs from 01, 02, 03, 04 and 06
                      │
                      ▼
           GOLDEN FIXTURE + EXPECTATIONS
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      every component test      CI / teammate onboarding
```

### Worked example — one offline end-to-end test

1. Load the committed source-faithful BoM sample and source/download metadata.
2. Recreate or validate its standard station-hour table.
3. Load the frozen fixture split and exact targets.
4. Build compatibility and corrected gauge graphs.
5. Train the HGNN for a few seeded CPU epochs and write row-level predictions.
6. Validate the prediction table's required columns and exact target coverage.
7. Compare transformations, graph summaries, and smoke metrics with reviewed
   expectations.
8. Complete in under ten minutes with network disabled.

## 3. Inputs

### 3.1 Settings owned by Component 05

#### IN01 — Test-dataset selection settings

The Component 05 owner creates and maintains
`config/fixtures/sydney_gauge_v1.yaml`. It names the accepted Component 01–04/06
versions, selection rules in FR02/03, fixture seed, output directory, and size limit.
The complete settings used are committed inside the fixture directory.

#### IN02 — Expectation update reason

Updating an existing fixture requires a short change identifier and explanation. The
update tool refuses an unexplained change.

### 3.2 Files and interfaces from the other components

#### IN10 — Accepted raw and standard data

Component 01 gauge files and Component 02 standard data for 2021–2022, including
source/download metadata, independent daily totals, station table, exclusions, and
dataset records.

#### IN11 — Split and graph interfaces

Component 03 split/target file structures and Component 04 sample-creation interfaces.
The fixture
uses a fixture-only split (FR05), not a production performance split.

#### IN12 — Model adapter interface

Component 06 training, prediction, validation, and comparison-method interfaces.

#### IN13 — Project seeds

`config/seeds.yaml`; only `fixture_seed` controls deterministic selection. Its hash is
recorded.

A hash in this document means a SHA-256 content fingerprint used to detect a changed or
corrupted file.

## 4. Functionality Requirements

### FR01 — Use one committed bundle layout

```text
tests/fixtures/sydney_gauge_v1/
  README.md
  manifest.json
  settings-used.yaml
  source/
    station-snapshot.json
    chunks/
    metadata/
    daily-totals/
  standard/
    observations.parquet
    stations.parquet
    exclusions.parquet
    manifest.json
  study/
    split.json
    targets.parquet
    split-check.json
  graph/
    compatibility-summary.json
    corrected-summary.json
  truth/
    daily-totals.parquet
    expected-values.json
  model/
    expected-bands.json
```

The full bundle must remain below 25 MiB compressed in Git. If source exemplars would
exceed the limit, reduce time windows—not source traceability or test coverage.

Parquet is the typed table format explained in Component 02; it preserves UTC timestamp
and numeric column types. In this directory, each `manifest.json` is a machine-readable
record listing the included files, versions, counts, and content fingerprints.

### FR02 — Select stations by a fixed rule

**Required**

Build a source fixture of at most 25 stations:

1. include every internal station returned for public numbers `563083` and `567163`;
2. include Greaves Creek internal id `49565602`;
3. include the station with public number `568163` as the rainfall-event anchor;
4. fill remaining places with the nearest baseline-eligible stations to that anchor by
   haversine distance, tie-breaking by internal station id;
5. retain any selected but ineligible station row in the raw/standard station lists and
   exclusion table, but exclude it from model graphs.

Save the resulting internal-id list in `manifest.json`. Recreating the fixture replays
the rule
and reports additions/removals; it does not silently choose a new list.

**Why**

- The forced cases exercise non-unique public identity and corrected source metadata.
- A compact geographic neighbourhood still produces meaningful KNN graph structure.

### FR03 — Select three deterministic 2021–2022 windows

**Required**

- **Wet window:** the earliest 72-hour window in 2022 where the anchor's verified
  daily-total series contains at least 25 mm in one rain day.
- **Dry window:** the earliest 48-hour window in 2021 where at least 80% of eligible
  fixture stations have verified zero daily totals.
- **Boundary window:** 48 hours centred on the 2022-04-03 Sydney daylight-saving
  transition, used to validate offset-aware parsing and unique UTC output.

If a rule cannot select a window, fixture construction fails and the requirement is
reviewed; the owner does not substitute an arbitrary date. Freeze selected UTC bounds
and source evidence in `manifest.json`.

### FR04 — Preserve source evidence and external truth

- Include only the raw gauge files/records needed for FR03 plus their original metadata.
- Freeze the corresponding BoM `DailyTotal.09HR` values in
  `truth/daily-totals.parquet` with station id, rain-day end, total mm, source series id,
  source-metadata hash, and retrieval time.
- Test code performs no network calls.
- Every standard fixture value traces to included source evidence.

### FR05 — Create a fixture-only split with exact targets

**Required**

- Use one deterministic three-way station split named `fixture_spatial_v1`:
  60% train, 20% validation, 20% test after station-id sort and seeded permutation.
- Require at least 10 train, 3 validation, and 3 test eligible stations.
- Write exact fixture target rows using Component 03's required columns and check them.
- Label it `fixture_only_not_for_science`; do not publish its metrics as cross-validation.

### FR06 — Save expected standard-data results

Store content hashes and explicit expected assertions for station identity, UTC times,
wet/dry/missing semantics, hourly aggregation, exclusions, row counts, and daily-total
verification. Numeric rainfall truth uses a 0.5 mm tolerance only where source tip
granularity requires it; deterministic transformations use exact comparisons.

### FR07 — Freeze graph expectations for two named modes

Build component 04's exact compatibility and corrected configurations. Store node/edge
counts, station order, target count, edge-weight summaries, validity values, LPE
consistency checks, target-hiding assertions, and output-file hashes/tolerances.

Do not require every cross-product of individual switches. The two named configurations
are the stable acceptance targets; focused unit tests cover invalid/partial combinations.

### FR08 — Define model smoke expectations conservatively

- Run component 06's inherited HGNN and reference method for a small fixed CPU budget.
- Store broad reviewed bands for finite loss, prediction count, non-negativity, RMSE,
  and Pearson correlation where defined.
- Do not require the HGNN to beat the reference method on this tiny fixture.
- Exact checkpoints are not required to match across platforms; required fields,
  repeatability under the
  same supported environment, and broad numeric bands are tested.

### FR09 — Provide one offline system acceptance test

One pytest marker or command checks source metadata → standard data → the fixture
split → both graph modes → tiny model/comparison predictions → the prediction table.
It must finish within ten minutes on a two-core CPU with network disabled.

### FR10 — Make incomplete components visible

Before an earlier component exists, its tests may skip only with a specific component
and output named. Once its output is present, the skip becomes a failure. Fake data must
not satisfy a source-verification acceptance test.

### FR11 — Control expectation updates

Provide a command that rebuilds candidate expectations into a temporary directory,
compares every metadata record/table/summary, and writes `EXPECTATION_DIFF.md`. Updating the
committed bundle requires reviewer approval and an explanation in the fixture changelog.

### FR12 — Keep the fixture self-validating

`manifest.json` contains the fixture version, selection rules and selected ids/windows,
input component versions/hashes, source-metadata hashes, file-format versions, seed-file
hash, code revision, output hashes, total size, and expected test command. A small
startup check validates this record before any component test runs.

## 5. Non-negotiable Rules

1. Fixture raw and standard data come only from the 2021–2022 baseline.
2. All tests run offline; source truth and source/download metadata are included.
3. The bundle uses BoM internal station ids and preserves public-number collisions.
4. Selection rules are deterministic and frozen; convenient hand-picking is prohibited.
5. Fixture metrics prove execution, not scientific performance.
6. Expected changes require a human-readable diff and reviewer approval.
7. The fixture contains no secrets, credentials, personal paths, or full source archive.
8. Skips are explicit temporary blockers, never replacements for acceptance.
9. Bundle size remains small enough for a fresh clone and CI.

## 6. Outputs

### OUT01 — Versioned fixture bundle

`tests/fixtures/sydney_gauge_v1/` using FR01, committed to Git and independently
validated using its `manifest.json` record.

### OUT02 — External truth tables

`truth/daily-totals.parquet` and source metadata from FR04, read by Component 02 and
source-faithfulness tests.

### OUT03 — Frozen fixture split and targets

`study/split.json`, `targets.parquet`, and `split-check.json`, read by graph/model tests.

### OUT04 — Graph and model expectations

Named compatibility/corrected graph summaries and conservative model smoke bands.

### OUT05 — Offline test suite and CI command

Tests live under `tests/` with one documented command that returns non-zero on any
unexpected skip or failure in a completed component.

### OUT06 — Fixture create/update commands

`src/fixtures/` or `tools/fixtures/` provides documented create, validate, compare, and
update commands with Python type hints. Candidate versions never overwrite the committed
fixture directly.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Fresh-clone offline run

A clean supported environment with network disabled validates the bundle and runs the
complete fixture suite in under ten minutes on CPU.

#### TEST02 — Source traceability and file integrity

Every file hash, source-metadata link, station id, time window, required file structure,
and input component version in `manifest.json` resolves and validates.

#### TEST03 — Deterministic selection

Replaying FR02/03 against the same accepted input versions selects the same ids and
windows. A provider/data revision appears in the generated diff.

#### TEST04 — Acceptance mutation drills

At minimum, changing one source value, standard-data value, split target, edge weight,
station mapping, and seed is caught by the correct layer with a specific message.

This is a **deliberate-error test** (often called a mutation test): the test intentionally
changes one known-good value and proves that the correct validation check notices it.

#### TEST05 — No-network proof

Block network access during the suite; no test attempts BoM/NCI access or depends on an
adjacent worktree/database.

### 7.2 Data-specific tests

#### TEST10 — Source-to-daily-total truth

Included tip evidence reproduces frozen BoM rain-day totals within one tip increment.

#### TEST11 — Wet, dry, missing, and boundary cases

Wet-window nonzero rainfall, dry verified zeros, deliberately absent/unverifiable hours,
and the boundary window all satisfy component 02 semantics.

#### TEST12 — Station identity

Repeated public numbers remain separate internal ids through raw data, standard data,
split/targets, graph mappings, and prediction rows.

### 7.3 Pipeline-specific tests

#### TEST20 — Both graph modes

Compatibility and corrected graph builds meet FR07 expectations and leakage guards.

#### TEST21 — End-to-end prediction contract

HGNN and comparison method each write exactly the fixture held-out row set and pass the
component 06 validator.

#### TEST22 — Update workflow

A candidate expectation change produces a complete readable diff and cannot update the
committed version without the required reason.

## 8. Documentation Requirements

Documentation lives inside the fixture bundle or under `docs/testing/`.

### DOCS01 — Fixture contents

Explain selected stations/windows, forced hard cases, eligibility, and why each exists.
Include a small map only if it improves understanding.

### DOCS02 — Running and reading tests

Provide commands, supported environment, runtime, offline guarantee, and the meaning of
pass/fail/skip.

### DOCS03 — Expectation update runbook

Explain candidate rebuild, diff review, approval, versioning, and rollback.

### DOCS04 — Fixture changelog

One entry per fixture version with input component versions, selected cases, reason, reviewer,
and important expectation changes.

### DOCS05 — End-to-end walkthrough

Trace one source tip through standard hourly rainfall, split target, graph sample,
prediction row, and validation in plain English.

## 9. Dependencies / Blockers

- Requires a small accepted 2021–2022 BoM raw-data acquisition and Component 02 dataset.
- Graph and model expectation sections require accepted component 04/06 interfaces.
- NCI radar/satellite fixtures are a later revision after real products are validated;
  their absence does not block the gauge-only golden fixture.
- The known 2023 storm is useful for ingestion API regression but is deliberately not
  part of this baseline fixture.

## 10. Current Technical Findings

- Existing tests rely on the adjacent Niramay worktree or legacy database and therefore
  do not provide a true fresh-clone fixture.
- The repository has 20 passing and 7 expected-failure tests in the last recorded
  environment, but several expected failures encode inherited defects rather than final
  component acceptance.
- A 2023 station `568163` storm is externally verified, but the required baseline is
  2021–2022. FR03 selects a verified in-period wet case instead of silently widening it.
- Current transitional smoke metrics use distorted legacy values and are runnability evidence,
  not valid millimetre-based expectation targets.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines golden fixture, standard/canonical data, rain day,
  held-out, LPE, and prediction table.
- Related component requirements: [01](01_DATA_SOURCE_INGESTION_REQUIREMENTS.md),
  [02](02_CANONICAL_DATA_REQUIREMENTS.md),
  [03](03_STUDY_SPLIT_REQUIREMENTS.md),
  [04](04_GRAPH_SAMPLE_REQUIREMENTS.md), and
  [06](06_MODEL_ADAPTER_REQUIREMENTS.md).
- Existing test patterns: `tests/conftest.py`, `tests/test_sydney_gauge_loader.py`, and
  `tests/test_split_integrity.py`.
- Evidence:
  [../archive/baseline-reconciliation/AUDIT_NIRAMAY.md](../archive/baseline-reconciliation/AUDIT_NIRAMAY.md)
  §8–9.
