# 01 — Collect and store source data

**Owner:** _unassigned_ · **Reviewer:** standard-dataset owner (Component 02)
**Status:** ready for implementation; radar and satellite acceptance are blocked on NCI access
**Baseline period:** 2021-01-01 through 2022-12-31 inclusive

## 1. Summary

This component finds and collects the external data used by the project. It creates a
restartable raw-data directory containing Australian Bureau of Meteorology (BoM)
rain-gauge records and exact paths to the required radar and satellite files on the
National Computational Infrastructure (NCI).

Its output is **source-faithful data**: records may be parsed, compressed, or packaged
for reliable storage, but their scientific values and meaning are unchanged. The
component exists because the inherited Sydney gauge data was downloaded manually,
without reproducible query metadata, and was later proven to contain inconsistent
value scaling.

Responsibility starts when the Component 01 owner creates a valid source-data settings
file and has access to the external sources. It ends when every requested raw file has
either been downloaded or recorded at its existing NCI path, has metadata explaining
where it came from, and is explicitly marked complete or failed.

This component does **not**:

- aggregate gauge tips into hourly rainfall;
- convert timestamps to the project's standard UTC timeline;
- gap-fill, interpolate, normalise, or remove suspected outliers;
- build model-ready tensors;
- choose train, validation, or test records;
- construct graphs; or
- run any model or statistical baseline.

Those responsibilities begin with component 02.

## 2. System Position

```text
BoM WaterData              NCI radar/satellite holdings
       │                               │
       └──────────────┬────────────────┘
                      ▼
             SOURCE-DATA INGESTION
                      │
       raw source data + source/download metadata
                      ▼
             02 STANDARD DATASET
```

Component 02 is the only direct reader. Other components must use Component 02's
standard dataset rather than parse source formats themselves.

### Worked example — Greater Sydney, January 2022

Requested settings:

```yaml
acquisition_id: sydney_2021_2022_v1
region: greater_sydney
start: 2021-01-01
end: 2022-12-31
sources: [gauge, radar, satellite]
```

For a January 2022 slice, the component:

A **tip event** is recorded when a tipping-bucket rain gauge has collected one fixed
increment of rainfall and its internal bucket tips. Component 01 stores those individual
events and the source metadata stating the rainfall increment; Component 02 later adds
them into hourly totals.

The component then:

1. discovers the BoM stations within the configured Greater Sydney boundary;
2. finds and validates the required quality-controlled tip series for each station;
3. retrieves source-faithful gauge records in restartable time batches, normally one
   station-month per file;
4. locates and indexes the NCI radar files intersecting the same region and period;
5. locates and indexes the matching Himawari satellite files;
6. writes one JSON metadata record for every gauge file and external-file reference;
7. updates persistent progress only after the associated data or index is safely stored;
8. writes a machine-readable complete raw-file list and a human-readable summary.

The Component 02 owner receives the raw-data paths, file list, station snapshot,
source/download metadata, and complete failure/exclusion list.

## 3. Inputs

### 3.1 Settings owned by Component 01

The Component 01 owner creates and maintains
`config/ingestion/sydney_2021_2022_v1.yaml`. It contains IN01–IN05 below. The file makes
important download choices visible and reviewable instead of hiding them in Python.

#### IN01 — Acquisition identifier

A stable lowercase identifier supplied in the settings file. For the first baseline use
`sydney_2021_2022_v1`. An **acquisition** means one identified attempt to collect a
specified region, period, and source set. Its id names the raw-data directory and is
recorded in every metadata and file-list entry.

#### IN02 — Geographic region

The settings supply a named region and a World Geodetic System 1984 (WGS84)
longitude/latitude bounding box. The first region is Greater Sydney. Source queries and
spatial file intersection use this boundary; code must not contain a private station
list or Sydney coordinate constants.

#### IN03 — Study period

The required baseline period is `2021-01-01T00:00:00` through
`2022-12-31T23:59:59`. Source-specific requests may include a boundary record needed to
close the final aggregation interval, but the index must distinguish that from study
coverage. More years require a new acquisition id and project decision.

#### IN04 — Requested source set

One or more of `gauge`, `radar`, and `satellite`. A run may complete a subset while an
external dependency is blocked, but the acquisition summary must show the status of
every requested source.

#### IN05 — Runtime policy

The settings include output directory, gauge time-batch size, request delay, maximum
retries, request timeout, and whether data already stored on NCI should be recorded at
its current path or copied into the project's raw-data directory. Defaults are specified
in FR01 and FR07.

### 3.2 External source inputs and dependencies

#### IN10 — BoM WaterData service — confirmed

BoM WaterData exposes station, time-series, and value queries through a KiWIS service
interface. KiWIS is the web API behind the portal. Gauge ingestion uses this service
for station discovery, series discovery, tip records, and independent daily totals.

Required coverage: configured Greater Sydney region, 2021–2022.

#### IN11 — NCI radar holdings — BLOCKED pending access

The candidate source is BoM Rainfields3, an operational gridded radar-rainfall product
believed to be held in NCI project `rq0`. The exact product version, directory, file
layout, units, spatial grid, revision policy, and 2021–2022 availability must be
confirmed from the holdings before full ingestion.

#### IN12 — NCI satellite holdings — BLOCKED pending access

The candidate source is Himawari-8/9 Convective Rain Rate, Physically based (CRRPH),
believed to be held in NCI project `rv74`. The exact collection, path, variables,
units, file format, resolution, revisions, and 2021–2022 coverage must be confirmed.

#### IN13 — Legacy Sydney metadata — cross-check only

`../COMP3888-niramay/database/australia/station_metadata.csv` may be used to explain
differences from inherited experiments. It is not an authoritative discovery input and
must never override current BoM identity or coordinates.

## 4. Functionality Requirements

### 4.1 Functionality shared by all data sources

#### FR01 — Validate one source-data settings file

**Required**

- Provide a command that accepts exactly one YAML settings-file path.
- Validate the acquisition id, WGS84 bounds, inclusive study period, requested sources,
  and output root before contacting any source.
- Default gauge files to one station and calendar month, request delay to 1 second, request
  timeout to 60 seconds, and maximum attempts to 5.
- Write the complete resolved settings to the raw-data directory before work begins.
- Refuse an existing acquisition id when its resolved settings differ.

**Done when**

Invalid or conflicting settings fail before any network or NCI read, while
valid settings create the raw-data directory structure in FR02.

#### FR02 — Use one prescribed raw-data layout

**Required**

```text
database/raw_archive/<acquisition_id>/
  settings-used.yaml
  acquisition-index.jsonl
  progress.json
  summary.json
  SUMMARY.md
  gauge/
    station-snapshot.json
    series-snapshot.jsonl
    tip-events/<station_id>/<YYYY-MM>.json.gz
    tip-events/<station_id>/<YYYY-MM>.metadata.json
    daily-totals/<station_id>/<YYYY>.json.gz
  radar/
    source-index.jsonl
    metadata/<source-file-id>.metadata.json
    samples/
  satellite/
    source-index.jsonl
    metadata/<source-file-id>.metadata.json
    samples/
```

- Create only requested source directories.
- Paths stored inside file lists are relative to the raw-data directory when a file is copied,
  and exact absolute NCI paths plus project id when it remains externally resident.
- Do not edit completed raw files or recorded NCI references. A source revision
  requires a new acquisition id.

`*.jsonl` means **JSON Lines**: each line is one complete JSON record. It is used for
file lists because a long acquisition can safely write and read one record at a time.

**Why**

- A fixed layout lets Component 02 read any acquisition without guessing paths.
- Separate gauge, radar, and satellite directories keep source metadata and failures
  understandable.

#### FR03 — Preserve source-faithful records

**Required**

- Preserve every source field needed to reconstruct and review the measurement.
- Compression and lossless packaging are allowed; scientific transformation is not.
- Do not aggregate, normalise, interpolate, gap-fill, quality-reclassify, clip,
  de-duplicate measurements, or replace source values.
- Record any lossless packaging step in that file's JSON metadata.

**Done when**

A reviewer can compare representative stored records with the source response or file
and account for every field and value without invoking downstream scientific logic.

#### FR04 — Write one JSON metadata record per raw file or file reference

**Required**

Each downloaded gauge file, NCI file reference, or copied subset has a sibling
`*.metadata.json` containing at minimum:

```text
acquisition_id
source
source_product_or_series
station_id or source_file_id
request_url_and_parameters or NCI_project_and_path
retrieved_or_indexed_at_utc
coverage_start
coverage_end
source_units
record_count or dimensions
content_size_bytes
sha256 when bytes are locally available
status: complete | failed
attempt_count
last_error
local_or_external_location
```

SHA-256 is a content fingerprint used to detect corruption or silent replacement. When
the project has local bytes, calculate it after writing the file. Unknown source
metadata is stored as `null` with an explanation in a dedicated note; it is never
guessed.

**Why**

- Each raw file or NCI reference can be traced back to its source independently.
- JSON is easy for teammates to inspect and for Python to validate.

#### FR05 — Maintain persistent progress state

**Required**

- Maintain `progress.json`, written atomically through a temporary file and rename.
- Store one work item for every gauge station/time window and every radar/satellite
  file with: source, identifier, time window, status (`pending`, `complete`,
  `failed`), output location, attempts, and last error.
- Mark an item complete only after its data/file-list entry, metadata record, and checks are on
  disk.
- Failed items remain visible and retryable.

**Why**

- Australia-scale acquisition may be interrupted by network failures or batch limits.
- Persistent state prevents expensive re-downloads and silent omissions.

#### FR06 — Make retries and repeated runs safe

**Required**

- Restarting with the same settings resumes pending or failed work and validates completed
  files rather than downloading them again.
- Write downloads to a temporary file, validate them, then atomically move them to the
  final path.
- Never append duplicate records to a completed monthly gauge file.
- If a source revision changes bytes for the same request, fail the existing acquisition
  and require a new acquisition id.

**Done when**

An interrupted acquisition and an uninterrupted acquisition produce equivalent indexes
and source content, with retries visible in metadata.

#### FR07 — Pace external requests and report failures

**Required**

- Document any provider-published usage limit before implementation.
- Until a stronger provider policy is confirmed, make BoM calls sequentially with at
  least 1 second between request starts.
- Retry timeouts and transient server errors with exponential backoff and bounded jitter.
- Do not retry invalid requests, authentication errors, or confirmed missing products.
- After the configured maximum attempts, record `failed` and continue independent work.
- Return a non-zero exit status when any requested work item remains failed.

#### FR08 — Produce the complete raw-file list and summaries

Here, an **index** means the complete machine-readable list of expected raw files or NCI
file references. The settings hash is a SHA-256 fingerprint of the complete settings
used; it proves that two acquisitions requested the same region, dates, sources, and
download rules.

**Required**

- `acquisition-index.jsonl` contains one current row per expected raw file or external
  file reference and points to its metadata and local/external location.
- `summary.json` records the resolved settings hash, code revision, counts by source/status,
  coverage, total bytes, failures, unavailable files, and completion status.
- `SUMMARY.md` presents the same outcome for teammates, including exact rerun command
  and blockers.
- No omission is represented only in console output.

### 4.2 Gauge ingestion

#### FR10 — Discover stations from the configured region

**Required**

- Call `getStationList` using the configured WGS84 bounds.
- Request at minimum the BoM internal `station_id`, public `station_no`, station name,
  latitude, longitude, and available owner/network fields.
- Save the complete source response as `gauge/station-snapshot.json`.
- Key all following work on internal `station_id`.
- Preserve multiple internal stations sharing one public station number.

**Done when**

The snapshot contains one record per returned internal id, coordinates are finite and
inside or explicitly adjacent to the requested boundary, and public-number collisions
remain separate.

#### FR11 — Discover and validate rainfall series

**Required**

- For every station, call `getTimeseriesList` for the Rainfall parameter.
- Select `DMQaQc.Merged.AsStored.1` as the quality-controlled, tip-level acquisition
  series and record its `ts_id`.
- Also select `DMQaQc.Merged.DailyTotal.09HR` for independent rain-day verification.
- Before region-wide download, validate both series on at least 20 stations spanning
  different networks, wet/dry periods, and the known public-number collision cases.
- Record every candidate series and selection result in `series-snapshot.jsonl`.
- If the required series is missing or ambiguous, do not substitute another series;
  mark that station failed with the candidate list.

**Why**

- The `AsStored` series retains individual tipping-bucket events needed to construct
  hourly rainfall without inheriting an unknown aggregation.
- The separate daily-total series provides an independent correctness check.

#### FR12 — Retrieve tip events in monthly station files

**Required**

- Call `getTimeseriesValues` by internal station series id and calendar month over the
  baseline period.
- Store the complete source response as compressed JSON in the FR02 path.
- Preserve irregular timestamps, 0.5 mm tips, zero marker rows, flags, and any source
  metadata returned.
- Do not infer dry hours or construct a regular time grid.
- Record empty but successful responses as complete, with zero records.

#### FR13 — Preserve station identity and source snapshots

**Required**

- Store internal id, public number, name, coordinates, owner/network, and source
  response fields without merging stations.
- Produce a collision report grouping repeated public station numbers.
- Never repair legacy coordinates inside ingestion; current source coordinates are
  preserved and later standard-dataset decisions are made in Component 02.

#### FR14 — Acquire and cross-check daily totals

**Required**

- Acquire the selected `DailyTotal.09HR` series for the same station and coverage.
- As an ingestion verification only, sum tip records over BoM rain days ending at 09:00
  local standard time and compare with the source daily totals.
- Allow a tolerance of one 0.5 mm tip per station-day.
- Record discrepancies in the file's metadata and summary; do not modify source records.
- A station with unexplained discrepancies beyond tolerance is marked unsafe for use in
  the standard dataset.

**Done when**

At least 20 representative station-months pass, and a deliberately corrupted local
copy is detected.

#### FR15 — Report gauge coverage explicitly

For every discovered station, report series availability, requested and returned time
coverage, monthly-file counts, empty files, failures, and verification status. The summary
must distinguish “no rain records returned” from “request failed” and “series missing”.

### 4.3 Radar ingestion

#### FR20 — Confirm the authoritative radar product — BLOCKED

**Required after NCI access**

- Inspect candidate Rainfields holdings in project `rq0` with the client or data owner.
- Confirm product name and version, exact project/path, file naming, revision policy,
  gauge-correction status, physical variable and units, temporal resolution, grid/CRS,
  spatial coverage, and complete 2021–2022 availability.
- Record the selected product and rejected candidates in DOCS03.
- Do not index or stage the full baseline until these facts are confirmed.

**Done when**

The selected product is approved in writing, its identity is machine-readable in the
resolved settings, and one real file has passed FR24.

#### FR21 — List the required radar files

**Required**

- Enumerate files intersecting the configured time period without reading full arrays.
- Read headers/coordinates to determine whether each file intersects the region.
- Write one `radar/source-index.jsonl` row per required source file with NCI project,
  path, coverage time, variable, units, dimensions, CRS/grid, size, revision metadata,
  gauge-correction status, and availability.
- Record missing intervals rather than silently shortening coverage.

#### FR22 — Access or stage the required radar data

**Required**

- Default to indexing existing NCI files in place; do not duplicate the full product.
- If a portable sample or downstream performance requires staging, copy only the
  configured spatial/time subset into `radar/samples/` without changing values.
- Hash copied bytes and preserve the exact parent NCI path and subsetting bounds.
- Never download or copy the full 2021–2022 product without a storage estimate and
  reviewer approval.

#### FR23 — Record leakage-relevant product semantics

Every radar file records whether it is gauge-corrected, which gauge correction is
applied, and whether the gauges overlap experimental targets. `unknown` is permitted
only while the product is blocked; it prevents downstream scientific use.

#### FR24 — Demonstrate one real radar sample

Read one real file and a small Greater Sydney subset. Verify that time, dimensions,
coordinates, CRS, values, missing-value encoding, units, and gauge-correction metadata
match the product documentation. Store the small source-faithful sample and metadata for
offline tests.

### 4.4 Satellite ingestion

#### FR30 — Confirm the authoritative satellite product — BLOCKED

**Required after NCI access**

- Inspect Himawari-8/9 CRRPH holdings in candidate project `rv74`.
- Confirm collection/version, exact project/path, file format, rainfall field name and
  units, temporal and spatial resolution, geolocation/CRS, missing-value encoding,
  revision policy, and complete 2021–2022 availability.
- Record the selected product and rejected candidates in DOCS03.
- Do not silently substitute another Himawari rainfall or cloud product.

#### FR31 — List the required satellite files

Write `satellite/source-index.jsonl` with the same traceability as FR21: NCI location,
coverage, variable, units, dimensions, geolocation, size, version/revision, and status.
Explicitly report missing time intervals and satellite/platform transitions.

#### FR32 — Access or stage the required satellite data

Index resident NCI files by default. Stage only approved Greater Sydney/time samples,
preserving values and exact parent paths. Record all spatial/time subsetting and hashes
in metadata records; do not resample or convert the rainfall field.

#### FR33 — Demonstrate one real satellite sample

Read one real file and small Greater Sydney subset. Verify time, dimensions,
geolocation, physical field, units, missing-value encoding, and documented resolution.
Store the source-faithful sample and metadata record for offline tests.

#### FR34 — Record platform and product continuity

The index must identify which Himawari platform and product version produced every
file and expose any change during 2021–2022. A platform or product transition is
metadata, not a reason to silently transform records.

## 5. Non-negotiable Rules

1. Baseline coverage is 2021-01-01 through 2022-12-31. Do not silently expand it.
2. Ingestion is source-faithful. Scientific preprocessing belongs to component 02.
3. Internal BoM `station_id` is the gauge identity key.
4. Every completed raw file or NCI file reference has a JSON metadata record and file-list entry.
5. Progress is persistent and updated only after durable writes and validation.
6. No expected file or failure is silently omitted.
7. Source data already on NCI is indexed in place unless copying is approved.
8. Radar gauge-correction status and all source units must be known before scientific use.
9. Credentials, tokens, and user-specific absolute paths are never committed.
10. Do not edit completed raw data; revisions create a new acquisition id.

## 6. Outputs

### OUT01 — Gauge source-faithful data

`database/raw_archive/<acquisition_id>/gauge/tip-events/` contains compressed monthly
BoM responses keyed by internal station id. Component 02 reads these records.

### OUT02 — Radar file list and samples

`radar/source-index.jsonl` contains one row per required NCI file. Approved small
source-faithful samples live under `radar/samples/`; full holdings remain referenced in
place. Component 02 uses the index to open the selected source.

### OUT03 — Satellite file list and samples

The satellite equivalent of OUT02, including platform/product version metadata.

### OUT04 — Station and series snapshots

`gauge/station-snapshot.json` and `gauge/series-snapshot.jsonl` preserve discovery
responses and selections keyed by internal station id.

### OUT05 — Per-file source/download metadata

One `*.metadata.json` per monthly gauge file, radar file, satellite file, or copied subset,
with the fields in FR04.

### OUT06 — Persistent acquisition progress

`progress.json` contains every planned work item and the fields defined in FR05.
It is both the resume input and an auditable completeness record.

### OUT07 — Acquisition summaries

`summary.json` is the machine-readable completion result; `SUMMARY.md` is the teammate
handoff with coverage, failures, warnings, storage, settings hash, and rerun command.

### OUT08 — Machine-readable complete raw-file list

`acquisition-index.jsonl` is Component 02's complete raw-file list. Every row contains
the acquisition id, source type (`gauge|radar|satellite`), file id, coverage, status,
metadata path, and local/external location.

### OUT09 — Importable ingestion package and command

Provide an importable package under `src/ingestion/` and one documented command that
runs a YAML source-data settings file. Public functions use Python type hints. The command
returns non-zero when requested work remains failed or blocked.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Complete requested acquisition

Given a region, period, and requested source set, every planned input is complete,
failed with cause, or explicitly blocked. The index and progress state contain the same
work-item set; there are no silent omissions.

#### TEST02 — Source faithfulness

Compare representative stored records with source responses/files and prove no
aggregation, normalisation, gap-filling, interpolation, or value replacement occurred.

#### TEST03 — Trace stored data back to its source

Select at least one file per source type and reconstruct its source query or NCI path,
coverage, units, version, retrieval/access time, and local hash from its metadata record.

#### TEST04 — Resume and atomic-write drill

Interrupt a multi-part run during a write. Restart it and confirm completed files are
not re-downloaded, the partial file is not accepted, and the final result matches an
uninterrupted acquisition.

#### TEST05 — Repeatability and revision visibility

Identical settings produce equivalent source content and file-list ordering. If the
provider revises content, the mismatch is visible and requires a new acquisition id.

#### TEST06 — Failure reporting

Plant one timeout, one missing file, and one malformed response. Each appears with a
distinct status/error, independent work completes, and the process exits non-zero.

### 7.2 Gauge-specific tests

#### TEST10 — Known storm API regression

As a small non-baseline API test, retrieve public station `568163` for
2023-01-04–05. The verified event contains 50 tips between 19:00 and 20:00 AEST on
January 4 (25.0 mm); daily totals are 4.0 mm and 30.0 mm for the adjacent 09:00 rain
days. This 2023 window validates source semantics and does not expand the 2021–2022
baseline raw-data acquisition.

#### TEST11 — Public-number collision

A station query for public number `563083` retains both BLACKHEATH G.C. (internal
`7730155`) and NORTH RICHMOND WFP (`49645132`) as separate station records.

#### TEST12 — Daily-total verification

FR14 passes for at least 20 representative station-months. Changing one local tip
causes the correct station-day to fail.

#### TEST13 — Request pacing

Request logs prove spacing and retry backoff comply with FR07.

### 7.3 Radar-specific tests

#### TEST20 — Read one real NCI radar file — BLOCKED

Open one selected product file and verify time, dimensions, coordinates, CRS, values,
units, missing-value encoding, and gauge-correction metadata against documentation.

#### TEST21 — Radar window inventory — BLOCKED

Index a known one-day Greater Sydney window. Every expected file is present or named as
missing, and a staged subset retains parent paths and hashes.

#### TEST22 — Gauge-correction status gate — BLOCKED

A radar file with `gauge_correction_status: unknown` is rejected by the acquisition
completion validator.

### 7.4 Satellite-specific tests

#### TEST30 — Read one real NCI satellite file — BLOCKED

Open one selected Himawari product file and verify dimensions, time, geolocation,
rainfall field, units, missing-value encoding, and platform/version.

#### TEST31 — Satellite window inventory — BLOCKED

Index a known one-day Greater Sydney window with no silent time gaps and stage one
source-faithful subset with parent path and hash.

#### TEST32 — Product identity gate — BLOCKED

A satellite file with unknown CRRPH collection/version or field units is rejected for
scientific use.

## 8. Documentation Requirements

All documents are Markdown under `docs/ingestion/`.

### DOCS01 — Source and access guide

Explain BoM, KiWIS, NCI, each selected project/product, access setup, endpoints or path
patterns, parameters, request pacing, and one annotated real example per data source.

### DOCS02 — Source-data exemplars

Show one real monthly gauge file, radar file/sample, and satellite file/sample. Explain every
important field/dimension, units, coordinates, timestamps, and missing-value encoding
in plain English.

### DOCS03 — Data-product decisions

Record the selected BoM series, radar product, and satellite product; concise reasons;
validation evidence; rejected alternatives; and remaining uncertainty.

### DOCS04 — Runbook

From a fresh environment, provide copy-pasteable commands to validate the settings, run
or resume an acquisition, inspect status, retry failures, and hand the raw data to
Component 02. Include local BoM and NCI/PBS examples where applicable.

### DOCS05 — Known issues and limitations

Document unavailable stations/files, API oddities, source revisions, coverage gaps,
NCI access constraints, and any unresolved unit or product metadata.

### DOCS06 — Raw-data layout and file fields

Document every file in FR02 and every field in metadata records, progress, file lists,
and summaries, with a small filled example.

## 9. Dependencies / Blockers

- **Gauge implementation:** unblocked; BoM service calls were verified on 2026-08-31.
- **Radar implementation:** `BLOCKED — NCI access and product confirmation required`.
- **Satellite implementation:** `BLOCKED — NCI access and product confirmation required`.
- The reviewer must approve the radar and satellite product decisions before listing
  or copying files for the full period.
- Full acquisition must not be committed to Git; `database/` remains ignored. Small
  approved fixture samples belong to component 05.

## 10. Current Technical Findings

These findings guide implementation but remain distinct from the component contract.

- BoM's working query types are `getStationList`, `getTimeseriesList`, and
  `getTimeseriesValues`.
- Investigation indicates `DMQaQc.Merged.AsStored.1` is the quality-controlled tip-level
  series and `DMQaQc.Merged.DailyTotal.09HR` is the independent daily-total series.
- Tip values observed in the verified series are 0.5 mm; timestamps are served using a
  fixed UTC+10 offset. Component 02 defines the project's standard UTC timeline.
- Public numbers `563083` and `567163` each identify two real stations; internal ids
  disambiguate them.
- The inherited 2021–2025 hand export contains time- and station-varying distortion and
  five false all-zero series. It is not an acceptable ingestion source.
- Rainfields3 in `rq0` and Himawari CRRPH in `rv74` are candidates, not yet validated
  product decisions. Exact paths and semantics remain blocked.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines BoM, KiWIS, NCI, Rainfields3, CRRPH, source-faithful
  data, tips, and station identity.
- BoM service base:
  `http://www.bom.gov.au/waterdata/services?service=kisters&type=queryServices`
- Verified station example: add
  `request=getStationList&format=json&station_no=563083` and explicit return fields.
- Verified series example: add
  `request=getTimeseriesList&format=json&station_no=211010&parametertype_name=Rainfall`.
- Verified values example: add
  `request=getTimeseriesValues&format=json&ts_id=782800010&from=2023-01-04&to=2023-01-06`.
- Investigation evidence:
  [../archive/baseline-reconciliation/AUDIT_NIRAMAY.md](../archive/baseline-reconciliation/AUDIT_NIRAMAY.md)
  §8 and [../CLIENT_BRIEFING.md](../CLIENT_BRIEFING.md) Part 1.
