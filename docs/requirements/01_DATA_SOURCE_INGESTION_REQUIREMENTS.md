# 01 — Get the raw data onto disk

**Owner:** _unassigned_ · **Reviewer:** Component 02 owner

## What this component is for

Three jobs:

1. Ask the Bureau of Meteorology (BoM) API for Sydney rain-gauge records, and NCI's
   THREDDS archive for Sydney radar, with the query parameters coming from a YAML config
   file rather than from hard-coded values. Register the satellite file we already have
   (still unowned — see **Open questions**).
2. Write what comes back to disk in a fixed, predictable layout.
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

radar:
  site_id: "71"          # T_Hills (Terrey Hills) — the Sydney radar
  product: prcp-m15      # Merged Gauge/Radar Accumulation, 15 min — gauge-corrected
  api:
    base_url: https://thredds.nci.org.au/thredds/fileServer/rq0/rainfields3
    request_delay_seconds: 1
    timeout_seconds: 120
    user_agent: COMP3888-P23-ingestion/0.1 (university capstone)

satellite: {}        # filled in when the satellite owner starts
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

### Radar: NCI THREDDS, BoM Rainfields 3

**Superseded finding, 2026-09-13:** a preprocessed `radar_2022_preprocessed.pkl` was
handed to the team, but it carries no coordinates, no companion `radar_node_coords.csv`,
and no way to recover either — it is a flat `[timestamp, data]` array with no embedded
lat/lon, bounds, or CRS. Nobody on the team has access to whoever produced it. Rather
than guess a grid to match its 952-value arrays, Component 01 now fetches the real,
georeferenced source directly and derives coordinates from its genuine projection
metadata. Do not manufacture coordinates from a node count or an assumed grid.

The source is NCI's public `rq0` THREDDS archive (BoM/Monash's Rainfields 3 volumetric
radar collection): `https://thredds.nci.org.au/thredds/catalog/rq0/rainfields3/`.
One file per radar site per day:

```text
https://thredds.nci.org.au/thredds/fileServer/rq0/rainfields3/<site>/<year>/<product>/<site>_<YYYYMMDD>.<product>.zip
```

Site **71** is `T_Hills` (Terrey Hills), the Sydney radar — confirmed from the
`station_name` attribute inside a downloaded file, not assumed from the site number.
Three products exist per site-year; use **`prcp-m15`**, not the others:

| Product | Contents | Size |
|---|---|---|
| `gndrefl` | raw reflectivity volumes | ~100–200 MB/day, ~40 GB/year — too large to fetch whole |
| `prcp-crate` | instantaneous precipitation rate | ~2–20 MB/day |
| **`prcp-m15`** | **"Merged Gauge/Radar Accumulation (15 min)"** | ~2 MB/day, **~1.4 GB for the whole 2021–2022 Sydney period** |

`prcp-m15` is the gauge-corrected Rainfields 3 product: each file's `contributing_gauges`
global attribute lists the BoM `station_no` values folded into that correction. Cross-
check this list against the study's gauge stations before quoting a radar result — if a
held-out test station's number appears, that comparison leaks (see **Rules**).

Each daily zip contains one NetCDF4 file per ~15 minutes
(`<site>_<YYYYMMDD>_<HHMMSS>.<product>.nc`), verified structure:

- grid: 512×512 cells, 0.5 km spacing, fixed and radar-centred (not the gauge bounding
  box)
- projection: `albers_conical_equal_area`, a complete CF grid-mapping definition
  (standard parallels, projection origin, WGS84 ellipsoid) — real, transformable to
  WGS84 lat/lon with `pyproj`, not assumed
- `precipitation`: kg/m² (≈ mm) accumulated over the 15-minute window — **not mm/h**,
  and not hourly; Component 02 sums four files per hour, the same way it sums gauge tips
- global attributes also give `station_id`, `station_name`, `source` (Rainfields
  version), and `contributing_gauges`

Both 2021 and 2022 are available from this source, resolving the "radar/satellite are
2022-only" open question below for radar specifically (satellite is unaffected).

### Satellite: preprocessed file already on disk, still unowned

`satellite_2022_preprocessed.pkl` exists (pandas DataFrame, 8,760 rows, columns
`timestamp` and `data`; each `data` cell a float32 array shaped (726, 2)) but has the
same missing-coordinates problem as radar did, with no known equivalent public archive
yet identified. Copy it under `database/raw/` and add an index row; do not modify it.
Component 02 needs to know what these numbers mean, and right now nobody does — see
**Open questions**.

## What you produce

```text
database/raw/
  gauge/
    stations.json                          getStationList response, unchanged
    series.csv                             station_id, ts_id, series_name, parameter
    tips/<station_id>/<YYYY-MM>.json.gz    getTimeseriesValues response, unchanged
    daily/<station_id>/<YYYY>.json.gz      getTimeseriesValues response, unchanged
  radar/
    <YYYY>/<YYYY-MM-DD>.prcp-m15.zip        NCI daily zip of ~15-min NetCDF files, unchanged
    radar_node_coords.csv                   node_id, latitude, longitude — 512x512 grid, real projection
  satellite/
    satellite_2022_preprocessed.pkl
  index.csv
  report.md
```

`index.csv`, one row per file written:

| Column | Example | Purpose |
|---|---|---|
| `source` | `gauge_tips` | the kind of file: `gauge_tips`, `gauge_daily`, `radar_precip`, or a satellite row |
| `station_id` | `7730155` | blank for radar/satellite rows |
| `path` | `gauge/tips/7730155/2022-01.json.gz` | relative to `database/raw/` |
| `start`, `end` | `2022-01-01`, `2022-01-31` | what the file is supposed to cover; for radar, `start` and `end` are the same day |
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
- [ ] `uv run python -m src.ingestion.radar.fetch config/ingestion/sydney.yaml` runs end
      to end and, run twice, does not re-fetch days it already wrote.
- [ ] `uv run python -m src.ingestion.radar.fetch config/ingestion/sydney.yaml --sample`
      fetches a handful of days (5 by default; `--sample N` for a different count) into
      `local/radar-sample/` rather than `database/raw/`, so a fresh clone can be tried out
      without committing to the full ~1.4 GB, ~20-minute, 2021–2022 download.
- [ ] `radar_node_coords.csv` has one row per grid cell (`node_id, latitude, longitude`),
      derived from a downloaded file's own CF projection metadata, not assumed or
      reverse-engineered from an array length.
- [ ] One real day's radar zip (trimmed to a couple of its ~15-minute files) is committed
      under `tests/data/ingestion/radar/` so Component 02's tests run on a fresh clone
      with no network.
- [ ] `docs/ingestion/README.md` gives the copy-pasteable commands and the meaning of
      every `index.csv` column, for both gauge and radar.

## Open questions

- **Radar provenance — resolved for the `prcp-m15` product.** It is Rainfields 3
  "Merged Gauge/Radar Accumulation (15 min)", units kg/m² (≈ mm) per 15-minute window,
  gauge-corrected with the contributing station numbers listed in each file's
  `contributing_gauges` attribute. `TO CONFIRM`: whether any contributing gauge is one of
  our held-out test stations (would leak); cross-check the list against
  `standard/stations.csv` once Component 02 produces it.
- **Satellite provenance and node coordinates are still unknown and still unowned.**
  `satellite_2022_preprocessed.pkl` has the same missing-coordinates problem radar did,
  and no equivalent public archive has been identified yet. Its 726 nodes cannot be
  placed on the map or connected to gauges until this is resolved. Component 04 is
  blocked on satellite until it is. Do not manufacture coordinates from the node count.
- **Which specific radar hours are missing or corrupt, and why** — `TO CONFIRM` once a
  full 2021–2022 fetch has run; check `index.csv` for `failed` days and gaps in the
  15-minute file count per day against the expected 96.
- **Radar is no longer 2022-only.** Both 2021 and 2022 are available from NCI for the
  `prcp-m15` product, matching the gauge baseline period. Satellite provenance and
  coverage remain unresolved.
