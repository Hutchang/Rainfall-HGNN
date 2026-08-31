# Component requirements

**Status:** implementation handoff · **Baseline period:** 2021-01-01 through 2022-12-31

This directory contains the current implementation requirements. Each document is
intended to be owned end-to-end by one teammate and reviewed by the owner of the
component that reads its main outputs.

All six work packages remain visible to the whole team in the map below. Team members
do not need to read all six documents end to end. They must:

1. understand the complete component map and shared rules on this page;
2. read their assigned component document in full;
3. read the `Inputs` and `Outputs` sections of directly adjacent components; and
4. use the [implementation handoff](../IMPLEMENTATION_HANDOFF.md) for sequencing,
   repository boundaries, and blockers.

## Component map

```text
External BoM/NCI sources
          |
          v
01 Collect/store source data
          | raw files + source/download metadata
          v
02 Prepare standard dataset
          |\
          | +--> 03 Choose stations/seeds
          |             |
          +-----------> 04 Create graph samples
                         |
                         v
                  06 Train/write predictions
                         |
                         v
                 row-level predictions

05 Small verified test dataset and CI spans every component.
```

| Document | Component | Primary inputs | Primary outputs | Readiness | Reviewer |
|---|---|---|---|---|---|
| [01](01_DATA_SOURCE_INGESTION_REQUIREMENTS.md) | Collect source data | BoM WaterData; NCI files | source-faithful raw data, metadata, and complete file list | Gauge work ready; NCI sources blocked | 02 owner |
| [02](02_CANONICAL_DATA_REQUIREMENTS.md) | Prepare standard dataset | 01 raw data and file list | checked hourly rainfall and station table | Starts with settings/sample; full data waits for 01 | 04 owner |
| [03](03_STUDY_SPLIT_REQUIREMENTS.md) | Choose stations and seeds | 02 station table | checked split files and `seeds.yaml` | Algorithm work ready; release waits for 02 | 04/06 owners |
| [04](04_GRAPH_SAMPLE_REQUIREMENTS.md) | Graph/sample construction | 02 data; 03 split | validated fold samples and sample-set record | Compatibility work ready; release waits for 02/03 | 06 owner |
| [05](05_GOLDEN_FIXTURE_REQUIREMENTS.md) | Small verified test dataset and CI | accepted outputs from all components | committed test dataset and offline acceptance suite | Tooling ready; final truth waits for component outputs | one owner from every component |
| [06](06_MODEL_ADAPTER_REQUIREMENTS.md) | Train models/write predictions | 04 samples; 03 split | checkpoints, run metadata, row-level predictions | Interface work ready; acceptance waits for 03/04 | future evaluation owner |

Statistical baselines, common evaluation, experiment orchestration, and production
National Computational Infrastructure (NCI) execution are the next specification wave.
Do not invent private versions of those systems inside current components. Documents
03 and 06 define the split and prediction files they must use.

## Shared rules

### RULE00 — Australian scope only

The project supports Australian rainfall data only. Singapore/NEA formats, timezone
assumptions, and commercial microwave link inputs are permanently out of scope. No
component may retain them behind an optional flag or describe them as future work.

### RULE01 — One seeds file

All randomness comes from `config/seeds.yaml` through the shared loader defined by
component 03. No production module contains a literal random seed.

### RULE02 — Label corrections

An inherited defect that changes numerical results is retained as an explicitly named
compatibility mode when practical. A corrected mode receives different named settings
and is recorded in the component's machine-readable record and run metadata.

### RULE03 — Standard units and time

After component 02, rainfall is millimetres and timestamps are timezone-aware UTC.
Later components must reject other units or timestamps without timezone information.

### RULE04 — Station identity

The station key is BoM's unique internal `station_id`. Public station numbers remain
display metadata because they are not unique.

### RULE05 — Trace every output to its inputs

Every generated file set records the exact input dataset/file versions or hashes,
complete settings, code revision, and relevant seed-file hash.

### RULE06 — Identical targets

Every method in an experiment predicts the exact held-out `(station_id, timestamp)`
records defined by the same split file. Components do not create private splits.

### RULE07 — Exclude with cause

No station, timestamp, expected raw file, or prediction is silently omitted. Exclusions
and failures appear in a machine-readable record and a human-readable summary.

### RULE08 — Baseline period

The baseline study period is 2021-01-01 through 2022-12-31 inclusive. Expanding it is a
new project decision, not a convenient settings change in a baseline result.

## Ownership and review workflow

1. Record the owner and reviewer at the top of the component document.
2. Implement against the documented files from earlier components. A small fixture or
   stub may be used while those files are incomplete, but its status must be explicit.
3. Keep acceptance tests beside the implementation and provide the required Markdown
   documentation.
4. The owner demonstrates every `TESTxx`, output path, and failure mode to the reviewer.
5. The reviewer signs off only when the next component can read the outputs without
   private parsing rules or undocumented assumptions.

## Requirements maintenance references

- [GLOSSARY.md](GLOSSARY.md) defines shared terminology.
- [AUTHORING_STANDARD.md](AUTHORING_STANDARD.md) applies when component requirements are
  created or materially revised. It is not required daily reading for implementers.
- Historical investigation and smoke-run evidence is indexed under
  [../archive/](../archive/README.md).
