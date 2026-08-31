# 02 — Prepare the standard rainfall dataset

**Owner:** _unassigned_ · **Reviewer:** graph/sample owner (component 04)

**Status:** starts by creating the preprocessing settings and working against a small,
accepted Component 01 sample; the full gauge dataset waits for Component 01, and radar
and satellite work waits for National Computational Infrastructure (NCI) access

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive

## 0. Branch/Git Usage

```
  main
  └── integration/data-pipeline
      ├── feature/01-source-data
      └── feature/02-canonical-data <--- you are here
```

Recommended workflow:

1. Create integration/data-pipeline from main (if it doesn't already exist).
2. Create feature/02-canonical-data from integration/data-pipeline.
3. Implement Component 02 and pass its documented TESTxx requirements.
4. Have its reviewer confirm that Components 03/04 can read the outputs without private fixes.
5. Merge feature/02-canonical-data into integration/data-pipeline.
6. Merge integration/data-pipeline into main only when its included components work together.
7. Delete completed feature branches.

## 1. Summary

This component turns the raw source data saved by Component 01 into the single cleaned,
checked dataset used by the rest of the project. We call that dataset the **canonical
dataset**: "canonical" simply means it is the one authoritative version that every
downstream component must use.

For rain gauges, this component:

- turns individual tipping-bucket events into hourly rainfall totals;
- converts all timestamps to Coordinated Universal Time (UTC);
- distinguishes a verified dry hour from an hour with missing data;
- creates one trustworthy row for every physical station; and
- records why any station, day, or raw file was excluded.

The component receives raw files and download records from Component 01. It produces
hourly rainfall tables for components 03, 04, and 05. After this point, teammates should
not need to parse Bureau of Meteorology (BoM) API responses, guess rainfall units,
resolve station identities, or repair timestamps.

This component does **not** choose train/test splits, calculate training statistics,
build graphs, hide target values, train models, or evaluate predictions.

## 2. System Position

```text
01 raw source data + download records
                 |
                 v
      02 PREPARE STANDARD DATASET
                 |
                 +--> hourly rainfall + station list --> 03 splits
                 +--> hourly rainfall + station list --> 04 graphs
                 +--> small verified data -------------> 05 fixture
                 +--> radar/satellite file lists ------> future fusion work
```

### Worked example — what a gauge "tip" means

A tipping-bucket rain gauge contains a small bucket mechanism. Rain fills the bucket;
when a fixed amount has accumulated, the bucket tips and the gauge records an event.
Each event therefore represents a small rainfall increment rather than an hourly total.

For illustration, if the source metadata confirms that one tip represents 0.5 mm:

```text
10:07  tip  -> 0.5 mm
10:12  tip  -> 0.5 mm
10:41  tip  -> 0.5 mm
```

Component 02 writes `1.5 mm` for the hour beginning at 10:00. The 0.5 mm value is only
an example: Component 01 must record the actual increment reported by the selected BoM
series, and Component 02 must use that recorded value rather than hard-coding 0.5 mm.

Component 01 may split a long download into smaller files, for example one station and
one month per file. Where this document says **raw gauge file**, it means one such file
containing tip events for a named station and time period.

For each BoM rain day, Component 02 also reads the independent daily rainfall total. It
uses that total to check the sum of the tips and to decide whether an hour with no tips
was genuinely dry or was simply not observed.

## 3. Inputs

### 3.1 Settings owned by Component 02

#### IN01 — Dataset preprocessing settings

The Component 02 owner creates and maintains:

```text
config/canonical/baseline_2021_2022.yaml
```

This YAML file contains the settings needed to turn one Component 01 download into the
standard dataset. A YAML file is used so that important choices are reviewed and saved,
not hidden as values inside Python code.

The first baseline file is:

```yaml
source_acquisition_id: sydney_2021_2022_v1
start_date: 2021-01-01
end_date: 2022-12-31
minimum_station_coverage: 0.90
prolonged_zero_detection:
  consecutive_zero_days: 90
  neighbour_distance_km: 50
  minimum_wet_neighbours: 3
  minimum_neighbour_rain_days: 5
output_directory: database/canonical
```

The required period and initial quality thresholds are prescribed in this document.
The Component 02 owner implements the file and proposes any later changes; the component
04 reviewer and project lead must approve changes that alter which scientific data are
included. The complete settings file used for a run is copied into that dataset's
output directory.

#### IN02 — Raw-data acquisition to process

The settings name exactly one accepted Component 01 `acquisition_id`. An acquisition id
is the unique name Component 01 gives one completed collection of raw source data.
The first implementation must not silently combine files from separate acquisitions.

### 3.2 Raw data inputs from Component 01

#### IN10 — Raw-file list and download summary

Component 01 OUT08 and OUT07 provide:

- a machine-readable list of every raw file expected in the acquisition; and
- a summary of completed, failed, missing, and deliberately excluded downloads.

Component 02 uses these files to know what should exist. It must not discover files by
scanning a directory and assuming that whatever happens to be present is complete.

#### IN11 — Gauge source data and download records

Component 01 OUT01, OUT04, and OUT05 provide:

- raw tip-event files for each station and time period;
- independent BoM daily rainfall totals;
- the station list and rainfall-series information captured at download time; and
- one JSON download record beside each raw file, stating its source, station, time
  coverage, units, status, and original request or NCI path.

#### IN12 — Radar and satellite raw-file lists — BLOCKED

Component 01 OUT02 and OUT03 will list the selected radar and satellite files. Component
02 cannot process them until Component 01 has confirmed the product, units, coverage,
and ability to read a real sample.

## 4. Functionality Requirements

### 4.1 Check inputs and create a repeatable dataset

#### FR01 — Check the raw data before preprocessing

**Required**

- Confirm that Component 01 marked the requested acquisition complete.
- Confirm that every expected raw gauge file exists and has a successful download
  record.
- Confirm that the station, time range, units, and source recorded for each file agree
  between the raw-file list and its download record.
- Recalculate each file's SHA-256 hash and compare it with Component 01's value. A hash
  is a content fingerprint: if a file is corrupted or silently replaced, its hash will
  change.
- Reject raw files marked `pending`, `failed`, incomplete, corrupt, or not yet safe for
  scientific use.
- If an expected file is missing, continue only when the approved exclusions list names
  that exact file and explains why it is absent.
- Write `input-validation.json` before creating the standard tables. It records the
  expected/found file counts, failures, exclusions, coverage checks, and final result:
  `ready` or `blocked`.

**Why**

- Incomplete downloads must be caught before they silently become missing rainfall.
- The saved input-check result lets another teammate see exactly why processing did or
  did not proceed.

**Done when**

A corrupt file, missing download record, hidden time gap, or unreported omission stops
processing and names the affected station, period, and file.

#### FR02 — Use one prescribed output layout

```text
database/canonical/<dataset_version>/
  observations.parquet
  stations.parquet
  radar-index.parquet
  satellite-index.parquet
  exclusions.parquet
  manifest.json
  input-validation.json
  settings-used.yaml
```

**What is Parquet?**

Apache Parquet is a file format for tables. It serves the same broad purpose as CSV,
but it preserves column types such as UTC timestamps, numbers, and lists; is normally
smaller; and is faster for Python tools such as pandas to read selectively. Use Parquet
for the standard internal tables. Do not also write a region-specific legacy format.

`manifest.json` is the dataset's machine-readable record: it lists the files in this
directory, what produced them, their versions and fingerprints, and important counts.
OUT05 gives its complete required contents and explains why later components need it.

The radar and satellite index tables may be empty while those sources are blocked. The
dataset record must say why they are empty.

#### FR03 — Make settings and dataset versions traceable

`dataset_version` is the unique name of one complete Component 02 output directory. It
changes whenever the raw input, preprocessing settings, required table format, or code
revision changes.

**Required**

- Save the complete preprocessing settings actually used, including default values, as
  `settings-used.yaml` beside the output.
- Calculate and record a SHA-256 hash of those settings. Even a small settings change,
  such as coverage changing from 90% to 80%, produces a different fingerprint.
- Generate `dataset_version` from the raw-data acquisition id and hashes, settings hash,
  table-format version, and code revision.
- Sort each table before writing: observations by `station_id,timestamp`; stations by
  `station_id`; radar/satellite file tables by source file id and start time; exclusions by source,
  station, and time.
- Running the same code with the same raw data and settings must produce the same rows,
  values, ordering, and dataset version. The compressed Parquet bytes need not be
  identical across different library versions.
- Never overwrite an existing dataset version with different data.

**Why use a settings hash?**

- It proves which exact settings produced a dataset without comparing YAML files by
  hand.
- It exposes a quiet threshold change that would otherwise make two scientifically
  different datasets look like the same version.

#### FR04 — Record every exclusion and quality decision

Write `exclusions.parquet` with one row for every excluded raw file, station, or time
period. It contains at least:

```text
source
raw_file_id
station_id (when applicable)
start_time (when applicable)
end_time (when applicable)
reason_code
plain_english_reason
download_record_path
```

`reason_code` values are fixed names defined in the data dictionary, for example
`missing_raw_file` or `daily_total_mismatch`. The dataset record reports counts for each
reason. A filter printed only to the console is not acceptable.

### 4.2 Prepare gauge data

#### FR10 — Build the station table

Create one `stations.parquet` row for each BoM internal `station_id` with at least:

```text
station_id
station_no
station_name
latitude
longitude
coordinate_system
owner_or_network
source_first_time
source_last_time
usable_first_time
usable_last_time
quality_flags
eligible_baseline
exclusion_reasons
download_record_path
```

- Use World Geodetic System 1984 (WGS84) coordinates from Component 01's accepted BoM
  station list.
- Preserve stations that share a public station number as separate rows when their BoM
  internal ids differ.
- Do not merge stations because their names, public numbers, or coordinates look alike.
- If a station fails required coverage or verification, keep its row and set
  `eligible_baseline=false` with the reason.

#### FR11 — Convert timestamps to UTC

**Required**

- Read the timezone offset supplied with every source timestamp.
- Convert the timestamp to timezone-aware UTC before grouping or comparing records.
- Store UTC explicitly in the Parquet timestamp column; timestamps without timezone
  information are not allowed.
- Interpret BoM `DailyTotal.09HR` using BoM's 09:00-to-09:00 rain day, not a normal
  midnight-to-midnight calendar day.

**Why**

- Sydney local-clock labels become ambiguous when daylight saving changes.
- UTC gives gauges, radar, and satellite one timeline for later comparison.

#### FR12 — Add tip events into hourly rainfall

**Required**

- A row stamped `t` contains the sum of accepted tip increments from `t` up to, but not
  including, `t + 1 hour`.
- Read the rainfall increment per tip from Component 01's series metadata. Do not
  hard-code 0.5 mm.
- Store the result as a non-negative `rainfall_mm` value in millimetres.
- Record `tip_count` and the ids of the raw files contributing to each hourly row.
- Reject exact duplicate source events unless their download record explains the
  duplication and this document contains an approved, repeatable de-duplication rule.

#### FR13 — Distinguish a dry hour from missing data

**Required**

- Treat a station's rain day as fully observed only when its daily-total record exists
  and the sum of its tips agrees within one confirmed tip increment.
- On a fully observed rain day, write an hour with no tips as `0.0` and
  `quality_flag=verified_dry`.
- Write an hour containing tips with `quality_flag=observed_tip`.
- On a missing or unverifiable rain day, do not create assumed zero rows. Leave the
  affected hours absent and record the reason in the exclusion and coverage tables.
- Never fill missing rainfall by interpolation or another estimate.

**Why**

- A tip-event feed is silent both when no rain fell and when the gauge or data feed was
  unavailable.
- The separate BoM daily total provides an independent check that the day was observed.

#### FR14 — Cross-check hourly rainfall against BoM daily totals

For every station and BoM rain day:

- add the standard hourly rows from 09:00 to 09:00;
- compare the result with the downloaded `DailyTotal.09HR` value;
- allow a maximum difference of one confirmed source tip increment; and
- if the difference is larger, stop dataset creation and report the station, date,
  hourly sum, BoM total, tip increment, and raw download records.

Save the number of checked, passed, and failed station-days in `manifest.json`.

#### FR15 — Detect and label unusable station behaviour

**Required**

- Check for prolonged all-zero behaviour, insufficient verified coverage, impossible
  negative rainfall, invalid coordinates, and unresolved station identity.
- The initial prolonged-zero rule is 90 consecutive verified days with zero rainfall
  while at least three stations within 50 km record rain on at least five days.
- Do not delete the station row. Add a quality flag, set `eligible_baseline=false` when
  the rule disqualifies it, and write an exclusion row.
- Record any reviewed false positive. Changing a threshold creates a new settings hash
  and therefore a new dataset version.

#### FR16 — Enforce the baseline period and coverage

Write only timestamps from 2021-01-01 through 2022-12-31 inclusive. Report verified and
missing hours for every station and year. A station is baseline-eligible only when at
least 90% of its verifiable baseline hours are present, unless an approved revision to
this requirement changes that threshold.

### 4.3 Prepare radar and satellite file lists

#### FR20 — Standardise radar and satellite file information — BLOCKED

After Component 01 validates a product, write one row per radar or satellite file with:

- UTC start/end time;
- spatial bounds that can be related to WGS84 coordinates;
- product name and version;
- physical field and units;
- array dimensions, grid/geolocation description, and missing-value encoding;
- path to the raw file and its download record; and
- whether radar rainfall is gauge-corrected, when relevant.

#### FR21 — Do not invent unapproved scientific transformations — BLOCKED

This component may standardise file information and access, but it must not regrid,
aggregate in time, convert radar reflectivity to rainfall, or rescale a radar/satellite
field until a product-specific rule is approved here. Those rules remain `TO SPECIFY`
until the team can inspect representative NCI files.

## 5. Non-negotiable Rules

1. Standard rainfall values are millimetres and timestamps are timezone-aware UTC.
2. Every station is identified by its unique BoM internal `station_id`.
3. Zero means verified dry; missing data means no observation row exists.
4. This component contains no train/test, model, graph, target-masking, or learned
   normalisation logic.
5. Every exclusion names the affected file, station, or time and explains why.
6. The same inputs and settings produce the same ordered data content.
7. The baseline contains only 2021–2022.
8. Radar or satellite files with unknown product, units, or scientific meaning remain
   blocked.
9. New components use the provided standard-data loader, not legacy CSV layouts.

## 6. Outputs

### OUT00 — Raw-data validation result

`input-validation.json` records what was checked before preprocessing, including file
counts, missing/failed files, exclusions, coverage, hash failures, and `ready|blocked`.
It is both readable by Python and understandable when opened by a teammate.

### OUT01 — Hourly gauge observations

`database/canonical/<dataset_version>/observations.parquet` contains one row per
verified station-hour:

```text
station_id: text
timestamp: UTC timestamp
rainfall_mm: decimal number
quality_flag: text
tip_count: whole number
source_file_ids: list of text ids
```

Components 04 and 05 and future comparison methods read this table.

### OUT02 — Station table

`stations.parquet` contains the fields in FR10 and exactly one row for each internal
`station_id`. Components 03, 04, and 05 read it.

### OUT03 — Radar and satellite file tables

`radar-index.parquet` and `satellite-index.parquet` use FR20. They may be empty for the
gauge-only milestone, but `manifest.json` must record the blocker.

### OUT04 — Exclusion table

`exclusions.parquet` contains the fields in FR04. Later components use it to explain
coverage and prevent silently reintroducing rejected data.

### OUT05 — Dataset record (`manifest.json`)

A **manifest** is the dataset's machine-readable record. It prevents teammates from
having to open every table to determine what the directory contains or how it was made.
`manifest.json` records at least:

```text
dataset_version
source acquisition id and hashes
preprocessing settings hash
code revision
expected filenames and table-format versions
row and station counts
time and geographic coverage
daily-total verification counts
exclusion counts by reason
SHA-256 hash of every output file
```

**Why it is required**

- A downstream component can check that it received a complete, internally consistent
  dataset rather than a mixture of files from different runs.
- File hashes expose corruption or accidental replacement.
- Counts and versions make a dataset inspectable without reading millions of rows.

### OUT07 — Python loader and preprocessing command

Provide `src/canonical/` with:

```text
prepare(config_path)
load_observations(dataset_version)
load_stations(dataset_version)
load_source_file_table(dataset_version, source_type)
```

The public functions use Python type hints and validate required columns and data types.
A missing dataset version or a table with the wrong columns/types must fail with a clear
message.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Repeatable dataset creation

Run the same code twice with the same accepted raw data and settings. Confirm identical
rows, column values, ordering, and dataset version. Do not require compressed Parquet
files to have identical bytes across different library versions.

#### TEST02 — Required columns and relationships

Confirm required columns/types, unique station ids, prescribed ordering, explicit UTC,
non-negative millimetres, valid station references from observation rows, and correct
file hashes in `manifest.json`.

#### TEST03 — Trace an hourly row back to raw data

Select hourly rows and follow `source_file_ids` through Component 01's raw-file list and
download records to the original tip values and source request/path.

#### TEST04 — Visible failure behaviour

Test a corrupted file, missing download record, incomplete station-day, and unreported
missing station. Each must stop processing or appear in the approved exclusions with
its exact cause; none may silently disappear.

### 7.2 Gauge-specific tests

#### TEST10 — Hour-boundary addition

Tip events immediately before, exactly at, and immediately after an hour boundary must
enter the correct hour. The total remains in millimetres.

#### TEST11 — UTC and daylight-saving independence

Source records around the 2021 and 2022 daylight-saving changes must map to unique,
ordered UTC times. Ambiguous local-clock labels must not duplicate or lose a UTC hour.

#### TEST12 — Missing versus zero

A verified rain day produces dry zero hours; a day without a daily-total record produces
no assumed zero hours; and a wet hour contains its summed tips and correct quality flag.

#### TEST13 — Daily-total cross-check

At least 20 representative station-months must pass FR14. Changing one hourly total by
more than one confirmed tip increment must stop processing and name the station-day.

#### TEST14 — Station identity

The two stations sharing public number `563083` and the two sharing `567163` must remain
four separate internal-id rows. Internal `station_id` values must be unique.

#### TEST15 — Dead or constant station detection

Use synthetic known-good and all-zero sequences to exercise FR15. Legacy public numbers
`211010`, `212063`, `568057`, `568232`, and `568295` may be investigation examples only
after mapping to internal ids; do not assume current BoM data for them are dead.

### 7.3 Radar and satellite tests — BLOCKED

#### TEST20 — Convert one real file of each type

One accepted radar and one accepted satellite sample must produce correct UTC coverage,
spatial bounds, product/version, field, units, dimensions, and raw-file linkage.

#### TEST21 — Block unknown scientific meaning

A file with unknown units/product, or radar data with unknown gauge-correction status,
must not be marked ready for the standard dataset.

## 8. Documentation Requirements

All documents are Markdown under `docs/canonical/`.

### DOCS01 — Data dictionary

Explain every field in the observations, stations, radar/satellite file lists,
exclusions, validation result, and dataset record. Include units, allowed values, and a
small real example.

### DOCS02 — Quality and coverage rules

Explain dry versus missing, daily-total cross-checking, station eligibility,
prolonged-zero detection, thresholds, and worked examples.

### DOCS03 — Preprocessing and loading runbook

From a fresh environment, show how to create/review the preprocessing YAML, check raw
inputs, create the dataset, validate it, load its tables, and inspect exclusions.

### DOCS04 — Dataset change history

For each dataset version, record its raw-data acquisition, settings, coverage, format or
rule changes, counts, reason for change, and superseded version when applicable.

### DOCS05 — Radar and satellite decisions

After NCI access, explain the approved radar/satellite fields and list transformations
that remain outside this component.

## 9. Dependencies / Blockers

- The Component 02 owner can create the settings, code, and tests immediately using a
  small test input with the documented Component 01 format.
- The full gauge dataset requires a completed and accepted Component 01 gauge download.
- Radar and satellite work waits for Component 01 to validate real NCI products/files.
- A daily total that disagrees by more than one confirmed tip increment is a source-data
  blocker, not permission to quietly increase the tolerance.
- Component 03 must not create the official study split from a provisional dataset.

## 10. Current Technical Findings

- The inherited hand export cannot be repaired with one scale factor and is not valid
  input for the standard scientific dataset.
- Current BoM investigation suggests that the selected series contains quality-controlled
  tip events commonly representing 0.5 mm and a separate 09:00 daily total. Component 01
  must confirm the increment and meaning across representative stations before the value
  becomes an accepted input.
- The inherited loader represents missing data by an absent row. FR13 preserves that
  behaviour while making verified zero rainfall explicit.
- No radar or satellite preprocessing rule is approved until real NCI files are read.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines shared terms including UTC, WGS84, rain day, hash,
  acquisition id, and canonical data.
- Previous component:
  [01_DATA_SOURCE_INGESTION_REQUIREMENTS.md](01_DATA_SOURCE_INGESTION_REQUIREMENTS.md).
- Investigation evidence:
  [../archive/baseline-reconciliation/AUDIT_NIRAMAY.md](../archive/baseline-reconciliation/AUDIT_NIRAMAY.md)
  §8 and [../CLIENT_BRIEFING.md](../CLIENT_BRIEFING.md) Part 1.
