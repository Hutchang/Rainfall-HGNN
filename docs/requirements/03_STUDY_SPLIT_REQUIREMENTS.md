# 03 — Choose train, validation, and test stations

**Owner:** _unassigned_ · **Reviewers:** graph/sample and model-adapter owners
**Status:** starts by creating split/seed settings and synthetic tests; official station
assignments wait for an accepted Component 02 dataset

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive

## 1. Summary

This component fixes two experiment-wide decisions:

1. which stations are training, validation, and test stations in every fold; and
2. which random seeds every component uses.

It produces read-only split files, the exact held-out station-hour list, and a check
report. Neural models and statistical comparison methods use the same files, so a
comparison cannot quietly use different targets or station availability.

Here, a **split** is one assignment of stations to training, validation, or testing. A
**fold** is one complete version of that assignment; five-fold evaluation repeats the
study five times with different test stations. A **random seed** is a saved number that
makes controlled random choices repeatable.

Responsibility begins with one accepted Component 02 standard dataset and ends with
checked, versioned split files. This component does not construct graphs, calculate
feature normalisation, train models, or calculate metrics.

## 2. System Position

```text
02 standard station list + hourly observations
                     │
                     ▼
          STUDY SPLITS + PROJECT SEEDS
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
    04 graph samples      06 models / future baselines
```

### Worked example — corrected five-fold study

1. Load eligible stations for Component 02 dataset `abc123`.
2. Project their coordinates to an Australian metre-based coordinate system.
3. Form eight spatial strata, deterministically shuffle each, and distribute stations
   round-robin across five test folds.
4. For fold 0, use fold 1's test group as validation and the remaining groups as train.
5. Write every observed 2021–2022 `(station_id, timestamp)` belonging to fold 0's test
   stations into `targets.parquet`, a table file that preserves column types.
6. Check disjointness, coverage, station identity, quality eligibility, target counts,
   and saved file fingerprints (hashes).
7. Components 04 and 06 read the resulting `split_id`; neither chooses stations again.

## 3. Inputs

### 3.1 Settings owned by Component 03

#### IN01 — Station-splitting settings

The Component 03 owner creates and maintains
`config/splits/baseline_2021_2022.yaml`. It supplies the Component 02 dataset version,
splitting method, fold count, station eligibility rule, number of spatial groups,
temporal-holdout setting, and output directory. The complete settings used are copied
into the output.

#### IN02 — Project seed settings

The Component 03 owner also creates and maintains `config/seeds.yaml`. Its required
fields and initial values are fixed in FR10.

### 3.2 Data from Component 02

#### IN10 — Standard station table

Component 02 OUT02. Only rows with `eligible_baseline=true` and no disqualifying flag
may enter a production split.

#### IN11 — Standard hourly observations

Component 02 OUT01. Used only to write the exact held-out target list and coverage
counts; rainfall values do not affect station assignment.

#### IN12 — Component 02 dataset record and exclusions

Component 02 OUT04/OUT05. The dataset version and recorded file fingerprints must match
the loaded tables.

## 4. Functionality Requirements

### FR01 — Select the eligible study population

**Required**

- Select only stations with `eligible_baseline=true` for the quoted dataset version.
- Use only accepted Component 02 observations from 2021–2022.
- Write an eligibility table listing every station-table row as included or excluded and
  the exact reason.
- Sort by internal `station_id` before any seeded operation.
- Refuse fewer stations than required by the selected splitting method rather than shrinking
  cluster/fold counts silently.

### FR02 — Use one prescribed split-file structure

Write `split.json` with this minimum structure. `schema_version` is a version number for
the file's required fields; it lets later code reject a file whose structure it does
not understand.

```json
{
  "schema_version": "1",
  "split_id": "<content-derived id>",
  "dataset_version": "<component 02 id>",
  "method": {"name": "full_coverage_spatial_v1", "params": {}},
  "seeds_file_hash": "<sha256>",
  "station_registry_hash": "<sha256>",
  "temporal_holdout": null,
  "folds": [
    {
      "fold_index": 0,
      "train": ["<internal station_id>"],
      "validation": ["<internal station_id>"],
      "test": ["<internal station_id>"],
      "target_list_hash": "<sha256>"
    }
  ]
}
```

- Station ids are strings because BoM identifiers are opaque keys, not quantities.
- Arrays are sorted in the final file.
- Calculate `split_id` from all file content except the `split_id` field itself using
  SHA-256. SHA-256 is a content fingerprint: changing a station, seed, or setting changes
  the id.
- Identical inputs and seeds produce the same ordered JSON content and `split_id`.

### FR03 — Preserve the inherited splitting method as `reference_kmeans_v1`

**Required**

Implement the inherited spatial method for controlled compatibility. **KMeans** is an
algorithm that groups coordinate points into a requested number of nearby clusters.

1. Run KMeans with eight clusters on raw `(longitude, latitude)` degrees using
   `split_seed`.
2. Within each cluster, deterministically shuffle its station rows using the same seed.
3. For five folds, assign shuffled positions 0–4 as that cluster's fold-specific test
   stations.
4. Treat remaining stations as the always-train pool.
5. For each fold, select 20% of the non-test stations per cluster for validation using
   the seeded inherited procedure; the remainder is train.

- Preserve the known partial-test-coverage behaviour and report its percentage.
- Run on unique internal station ids; do not reproduce duplicate-metadata leakage.
- Label these split files `reference_kmeans_v1`; never call them corrected or full
  coverage.

**Why**

- Existing smoke results use this study shape, so it remains necessary for controlled
  regression comparisons.
- Reproducing its limitation explicitly is safer than hiding it in a generic “k-fold”
  label.

### FR04 — Provide the corrected splitting method `full_coverage_spatial_v1`

**Required**

Use this fixed five-fold design:

1. Transform WGS84 station coordinates to EPSG:3577 (Australian Albers, metres).
2. Run KMeans with eight spatial strata using `split_seed`.
3. Within each stratum, sort by station id, apply a seeded permutation, and assign each
   station to test fold `position mod 5`.
4. For fold `f`, validation is the test group of fold `(f + 1) mod 5`.
5. Training is every remaining eligible station.

Properties:

- every eligible station appears in exactly one test set and one validation set;
- each fold's train, validation, and test sets are pairwise disjoint;
- test sets are pairwise disjoint and their union is the eligible population;
- save cluster labels and station-to-fold assignments for inspection.

**Why**

- It fixes the inherited scheme in which most stations are never tested.
- Projected metres avoid latitude-dependent distortion from clustering raw degrees.

### FR05 — Write the exact held-out station-hour list

**Required**

Write one `targets.parquet` for the split with these columns. Parquet is the typed table
format explained in Component 02; unlike CSV, it preserves UTC timestamp and numeric
column types.

```text
split_id: string
fold: int16
station_id: string
timestamp: timestamp[UTC]
```

- Include one row for every accepted Component 02 observation at every fold's test station.
- Sort by `(fold, station_id, timestamp)` and require uniqueness.
- Store each fold's subset hash in `split.json`.
- Do not include rainfall values; methods join truth from the quoted dataset version.

**Why**

- A station-only split is insufficient when some hourly observations are missing.
- This table is the exact list reused by models, inverse-distance weighting (IDW),
  Kriging, and evaluation.

### FR06 — Keep temporal holdout explicit

`temporal_holdout` is `null` for both current protocols: stations are split spatially,
while all eligible 2021–2022 timestamps may appear in train, validation, and test
graphs. The field remains in the file structure for a future, separately specified temporal
study. No owner may add a private temporal split inside another component.

### FR07 — Check every split before release

This is an automated completeness and consistency check, not a manual code audit.

**Required**

The check validates:

- dataset and station-table hashes;
- station existence, uniqueness, and eligibility;
- within-fold train/validation/test disjointness;
- splitting-method-specific test coverage across folds;
- target-list uniqueness, UTC type, coverage, and exact station membership;
- fold/cluster size tables and geographic coverage;
- no unreported exclusions;
- split id and every content hash.

Write `split-check.json` and `SPLIT_CHECK.md`. Components 04 and 06 must reject a split whose check
status is not `passed`.

### FR08 — Do not edit a released split

A released split directory is never edited. A different dataset, seed, eligibility
rule, station assignment, target list, or splitting-method parameter creates a new `split_id`
and changelog entry.

### FR10 — Own `config/seeds.yaml`

Create exactly:

```yaml
schema_version: 1
split_seed: 123
model_init_seed: 20260831
dataloader_seed: 20260831
torch_global_seed: 20260831
numpy_global_seed: 20260831
python_global_seed: 20260831
fixture_seed: 4242
```

- Provide `src/reproducibility/seeds.py` with type-hinted `load_seeds(path)` and
  `seed_everything(seeds)`.
- `seed_everything` applies Python `random`, NumPy, and PyTorch seeds and configures
  deterministic algorithms where supported.
- Every entry point calls it before creating a sampler, model, or data loader.
- Every generated file set records the seed-file hash.
- There is no silent fallback when the file or a required key is missing.

### FR11 — Enforce the single-seed source

Provide a repository check that rejects literal production seeds outside
`config/seeds.yaml` and the seed loader. Tests may contain clearly labelled fixed seeds
for local synthetic cases. Existing hardcoded seeds in transitional drivers must be
replaced when those paths are integrated.

## 5. Non-negotiable Rules

1. Only unique BoM internal station ids appear in splits.
2. Only stations and 2021–2022 rows accepted by Component 02 are eligible.
3. Split generation never reads rainfall magnitudes.
4. Released split directories are read-only; any content change creates a new `split_id`.
5. The exact prediction target list is part of every split.
6. Every component using a split rejects a failed or mismatched check report.
7. `reference_kmeans_v1` and `full_coverage_spatial_v1` remain distinct labels.
8. All production randomness comes from `config/seeds.yaml`.
9. Spatial-only protocols do not imply temporal generalisation.

## 6. Outputs

Each release lives under:

```text
database/splits/<split_id>/
  split.json
  targets.parquet
  eligibility.parquet
  station-assignments.parquet
  split-check.json
  SPLIT_CHECK.md
  settings-used.yaml
```

### OUT01 — Accepted split definition

`split.json` using FR02, read by components 04 and 06 and future comparison methods.

### OUT02 — Exact prediction target list

`targets.parquet` using FR05, read by model adapters, comparison methods, prediction
validators, and evaluation.

### OUT03 — Eligibility and station assignments

`eligibility.parquet` records inclusion/exclusion. `station-assignments.parquet`
contains station id, coordinates, projected coordinates, spatial stratum, test fold,
and validation fold.

### OUT04 — Split check result

Machine-readable `split-check.json` and teammate-readable `SPLIT_CHECK.md` using FR07.

### OUT05 — Project seeds and loader

Tracked `config/seeds.yaml` and `src/reproducibility/seeds.py` using FR10.

### OUT06 — Split creation/check command

An importable package under `src/study/` and documented commands to generate, check,
inspect, and export a split. Commands return non-zero when the split check fails.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Repeatable split creation

Two supported machines generate the same station assignments, target rows, ordering,
and ids from the same Component 02 data, settings, and seeds. Parquet compression bytes
do not need to match across different library versions.

#### TEST02 — Exact target handoff

Every target row belongs to its fold's test station list and Component 02 dataset; every
available standard observation for those stations appears exactly once.

#### TEST03 — Split check catches violations

Planted overlap, unknown station, ineligible station, duplicate target, wrong timestamp
type, missing target, bad hash, and unreported exclusion each fail with the responsible
identifier named.

#### TEST04 — Changed content creates a new id

Changing any station assignment, target row, seed, or splitting-method parameter changes
`split_id`; attempting to overwrite an existing different release fails.

### 7.2 Splitting-method tests

#### TEST10 — Reference fidelity

On fixed station-table test data, `reference_kmeans_v1` matches the inherited algorithm's
station assignments after removing duplicate-id leakage and reports partial test
coverage as a known property.

#### TEST11 — Full-coverage properties

For `full_coverage_spatial_v1`, every eligible station appears once in test, once in
validation, and three times in train across five folds; all required disjointness checks
pass.

#### TEST12 — Coordinate-system decision

A Sydney fixture demonstrates that corrected clustering uses EPSG:3577 metre
coordinates while the reference method preserves raw-degree coordinates.

### 7.3 Seed tests

#### TEST20 — Seed application

Repeated Python, NumPy, PyTorch initialisation, and data-loader sequences match after
`seed_everything`; changing the relevant seed changes the sequence.

#### TEST21 — Missing seed failure

Missing file/key and hash mismatch fail clearly. The literal-seed repository check
finds a planted violation.

## 8. Documentation Requirements

All documents are Markdown under `docs/study/`.

### DOCS01 — Split-file guide

Explain every field and table with a complete filled example and state which component
uses it.

### DOCS02 — Splitting-method guide

Explain spatial cross-validation, both protocols, projected coordinates, the inherited
partial-coverage limitation, and the corrected round-robin design with a small diagram.

### DOCS03 — Seeds guide

Explain each seed, loader usage, limits to repeatability across hardware, and which
generated files record the seed hash.

### DOCS04 — Split changelog

One entry per split id with dataset version, splitting method, coverage, reason, and previous version.

### DOCS05 — Operator runbook

Commands to generate, check, inspect fold membership/coverage, and hand the split to
components 04/06.

## 9. Dependencies / Blockers

- Requires an accepted Component 02 dataset version and its type-hinted Python loader.
- Production full-coverage splits must wait for regenerated BoM data; distorted legacy
  data may be used only for regression tests.
- A splitting-method parameter change requires reviewer agreement because it changes evaluation
  targets.
- Future temporal studies require a new splitting method, not an edit to these
  spatial releases.

## 10. Current Technical Findings

- Both inherited implementations split spatially and reuse all timestamps across roles.
- Their eight-cluster/five-fold algorithm tests only a small fixed subset of stations.
- Duplicate metadata previously put the same public-id series in conflicting roles;
  Component 02's internal ids remove that identity defect.
- The transitional driver can import a precomputed split JSON, but it drops unavailable
  ids rather than enforcing the exact target list. New components must use the stricter
  contract.
- Model initialisation and dropout are currently unseeded, and a driver contains a
  hardcoded `123`. FR10/11 retire these behaviours.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines split, fold, spatial cross-validation, leakage,
  held-out targets, and dataset/split versions.
- Previous component: [02_CANONICAL_DATA_REQUIREMENTS.md](02_CANONICAL_DATA_REQUIREMENTS.md).
- Inherited samplers: `src/sampling/main.py` and
  `../COMP3888-niramay/data/sampling.py`.
- Investigation evidence:
  [../archive/baseline-reconciliation/AUDIT_NIRAMAY.md](../archive/baseline-reconciliation/AUDIT_NIRAMAY.md)
  §8.1/8.3.
