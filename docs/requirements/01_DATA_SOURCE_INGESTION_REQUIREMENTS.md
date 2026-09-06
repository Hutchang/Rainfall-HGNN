# 01 — Get the raw data onto disk

**Owner:** _unassigned_ · **Reviewer:** Component 02 owner

## What this component is for

Three jobs:

1. Ask the Bureau of Meteorology (BoM) API for Sydney rain-gauge records, with the query
   parameters coming from a YAML config file rather than from hard-coded values.
2. Write what comes back to disk in a fixed, predictable layout, plus register the radar
   and satellite files we already have.
3. Produce one `index.csv` that tells Component 02 exactly what exists.

You store what the source gave you. You do **not** convert timezones, add tips into
hourly totals, fill gaps, or drop outliers. That is Component 02's job. The reason for
the split is that the inherited Sydney data was hand-downloaded with no record of the
query, and it turned out to be silently rescaled. If raw stays raw, any later mistake is
found by re-reading these files.

## What you receive

### A config file you write yourself

`config/ingestion/sydney.yaml` — one file per region, one section per source:

```yaml
region: sydney
start: 2021-01-01
end: 2022-12-31
output_dir: database/raw

gauge:
  bounds: {min_lat: -34.3, max_lat: -33.4, min_lon: 150.4, max_lon: 151.4}
  api:
    base_url: https://www.bom.gov.au/waterdata/services
    request_delay_seconds: 1
    timeout_seconds: 120
    user_agent: COMP3888-P23-ingestion/0.1 (university capstone)
  series:
    tips: DMQaQc.Merged.AsStored.1
    daily: DMQaQc.Merged.DailyTotal.09HR

radar: {}        # filled in when the radar/satellite owner starts
satellite: {}
```

Everything that decides *which data we get* lives here, so a reviewer can see it without
reading Python. `src/ingestion/config.py` validates the file and fails with a concrete
message on a missing key. The box above is the current Greater Sydney box; the comments
in the real file record where it came from and the open question about whether the upper
Blue Mountains belong in it.

### The BoM WaterData API

Base URL: `https://www.bom.gov.au/waterdata/services` — every request also carries
`service=kisters&type=queryServices&datasource=0&format=json`. Use **https**: the old
`http://` address now redirects, and the client does not follow redirects, so a moved
API fails loudly instead of silently.

Three calls, in this order. All three were confirmed working on 2026-08-31 and
re-verified against the live API on 2026-09-05:

| Call | Gives you | Why you need it |
|---|---|---|
| `getStationList` | every station in the bounding box: internal `station_id`, public `station_no`, name, lat, lon, owner | You cannot request data without `ts_id`, and you cannot get `ts_id` without `station_id`. Coordinates are also what Component 04 uses to build graph edges |
| `getTimeseriesList` | the available series per station, each with a `ts_id` | Picks the right feed. Use `DMQaQc.Merged.AsStored.1` (individual tips) and `DMQaQc.Merged.DailyTotal.09HR` (BoM's own daily total) |
| `getTimeseriesValues` | the actual records for one `ts_id` and date range | The data |

Request one station and one calendar month of tips at a time, plus one request per
station-year for the daily totals and one per station to list its series. For Greater
Sydney that is roughly 180 stations × (1 + 24 + 2) ≈ 4,800 requests; at one per second,
about 90 minutes.

Why two series and not one: the tip series is silent both when it is dry and when the
gauge is broken. The daily total is an independent record of what BoM says fell that
day. Component 02 compares them to tell those two cases apart. Without the daily
series, "missing" and "dry" are indistinguishable and the dataset is not trustworthy.

**Note on the API.** The [SOS2 guide](https://www.bom.gov.au/waterdata/wiski-web-public/Guide%20to%20Sensor%20Observation%20Services%20%28SOS2%29%20for%20Water%20Data%20%20Online%20v1.0.1.pdf)
documents a second interface to the same BoM server, using OGC-style requests
(`GetObservation`, `GetFeatureOfInterest`). Either works. This document specifies the
KiWIS interface above because those exact calls are already verified. If you switch to
SOS2, re-verify with the storm case in **Done when** and update this section.

### Radar and satellite files, already on disk

These exist; nothing needs downloading. Copy them under `database/raw/` and add index
rows. Do not modify them.

| File | Contents |
|---|---|
| `radar_2022_preprocessed.pkl` | pandas DataFrame, 8,640 rows, columns `timestamp` (UTC) and `data`; each `data` cell is a float32 array of 952 values |
| `radar_node_coords.csv` | 952 rows, columns `latitude`, `longitude` — one row per radar value, same order |
| `satellite_2022_preprocessed.pkl` | pandas DataFrame, 8,760 rows, columns `timestamp` (UTC) and `data`; each `data` cell is a float32 array shaped (726, 2) |

Component 02 needs to know what these numbers mean, and right now we do not. Your job is
to find out and write it down — see **Open questions**.

## What you produce

```text
database/raw/
  gauge/
    stations.json                          getStationList response, unchanged
    series.csv                             station_id, ts_id, series_name, parameter
    tips/<station_id>/<YYYY-MM>.json.gz    getTimeseriesValues response, unchanged
    daily/<station_id>/<YYYY>.json.gz      getTimeseriesValues response, unchanged
  radar/
    radar_2022_preprocessed.pkl
    radar_node_coords.csv
  satellite/
    satellite_2022_preprocessed.pkl
  index.csv
  report.md
```

`index.csv`, one row per file written:

| Column | Example | Purpose |
|---|---|---|
| `source` | `gauge_tips` | the kind of file: `gauge_tips`, `gauge_daily`, or a radar/satellite row |
| `station_id` | `7730155` | blank for radar/satellite rows |
| `path` | `gauge/tips/7730155/2022-01.json.gz` | relative to `database/raw/` |
| `start`, `end` | `2022-01-01`, `2022-01-31` | what the file is supposed to cover |
| `n_records` | `412` | `0` is fine and means a dry month; it is not a failure |
| `status` | `ok` or `failed` | |
| `error` | `HTTP 504` | blank when `ok` |
| `retrieved_at` | `2026-09-02T10:14:00Z` | |
| `sha256` | `a3f9…` | one line: `hashlib.sha256(p.read_bytes()).hexdigest()` |

The `sha256` column is written but nothing downstream is required to verify it. It exists
so that when a result changes, you can tell in one command whether the data changed.

`report.md` is for humans: how many stations, how many files, which ones failed and why,
and the command to re-run.

## How your output is used

Component 02 reads `index.csv` first, then opens each `ok` gauge file, sums the tips into
hourly totals, and converts to UTC. It reads `index.csv` rather than listing the
directory, because a directory listing cannot tell the difference between "this month
had no rain" and "this month failed to download".

## Rules

1. Store the API response as it came back. No aggregating, rounding, gap-filling, or
   unit conversion.
2. `station_id` is the key. Never merge two stations because their `station_no` matches.
3. A run that finishes with failures — or with a station that has neither required
   series — still writes `index.csv` and `report.md`, and exits non-zero.
4. Never edit a file already on disk. Delete and re-fetch instead.

## Done when

- [ ] `uv run python -m src.ingestion.gauge.fetch config/ingestion/sydney.yaml` runs end
      to end and, run twice, does not re-fetch files it already wrote (a file already on
      disk is kept in the index but not downloaded again — there is no state file).
- [ ] `index.csv` has a row per expected station-month, with failures marked.
- [ ] **Storm check.** Fetching public station `568163` for 2023-01-04 to 2023-01-05
      returns 50 tips between 19:00 and 20:00 AEST on 4 January (25.0 mm), with adjacent
      09:00 daily totals of 4.0 mm and 30.0 mm. This is an API sanity check only; it does
      not go in the 2021–2022 dataset.
- [ ] **Collision check.** Querying `station_no=563083` keeps BLACKHEATH G.C.
      (`7730155`) and NORTH RICHMOND WFP (`49645132`) as two separate rows.
- [ ] One real station-month of tips and one real station-year of daily totals are
      committed under `tests/data/` (a few KB, gzipped and unmodified) so Component 02's
      tests run on a fresh clone with no network.
- [ ] `docs/ingestion/README.md` gives the copy-pasteable commands and the meaning of
      every `index.csv` column.

## Open questions

- **Radar and satellite provenance — you own this.** For each file we need: which
  product, which units, and how it was made. Specifically: are the 952 radar values
  mm/h or mm accumulated over the hour? Values run 0 to ~72 with a mean near 0.28, which
  is consistent with mm/h but is not proof. Is the radar gauge-corrected, and if so with
  which gauges? If those gauges include our test stations, radar results leak and the
  comparison is invalid. Ask whoever produced the files and write the answers into
  `docs/ingestion/README.md`.
- **Satellite node coordinates are missing.** Radar has `radar_node_coords.csv`;
  satellite has no equivalent, so its 726 nodes cannot be placed on the map and cannot be
  connected to gauges. Component 04 is blocked on satellite until this file exists. Find
  it or regenerate it.
- **Radar is missing 120 hours** of 2022 (8,640 rows where 8,760 were expected). Which
  hours, and why? Component 02 needs to know whether they are absent or corrupt.
- **Radar and satellite are 2022 only.** Gauges will cover 2021–2022. Confirm with the
  project lead that fusion experiments are 2022-only, or source 2021.
