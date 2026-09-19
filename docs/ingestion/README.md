# Component 01: source data ingestion

Downloads raw source data to `database/raw/` and writes an index of every file.
Sprint 1 contract: [01_DATA_SOURCE_INGESTION_REQUIREMENTS.md](../archive/sprint1/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md)
(archived; the Sprint 2 contract is being written).

Sources:

| Source | Status | Code |
|---|---|---|
| Gauges (BoM Water Data Online) | implemented | `src/ingestion/gauge/` |
| Radar (2022 preprocessed file) | not started, owner TBC | |
| Satellite (2022 preprocessed file) | not started, owner TBC | |

## Commands

All commands run from the repository root inside the uv environment.

Preview the bounding box before downloading anything. Draws every BoM rainfall gauge in
Australia, coloured by whether it advertises coverage of the study period, with the
config's box overlaid. Writes to `database/survey/` (gitignored).

```
uv run python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml
uv run python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml --show     # also open a window
uv run python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml --no-box   # Australia only
```

Download the gauge data. About 4,800 requests at one per second for Greater Sydney,
roughly 90 minutes. Exit code is non-zero if any file failed.

> The box in `sydney.yaml` currently covers the whole continent, so this estimate is
> wrong for the file as it stands — see ISS-11.

```
uv run python -m src.ingestion.gauge.fetch config/ingestion/sydney.yaml
```

Run it again to resume: files already on disk are skipped and only missing or failed
files are fetched. That is the whole resume mechanism; there is no state file. To
refetch a file, delete it. To redo the station or series lookup, delete
`database/raw/gauge/stations.json` or `database/raw/gauge/series.csv`.

For a quick test, limit the stations and write somewhere disposable:

```
uv run python -m src.ingestion.gauge.fetch config/ingestion/sydney.yaml --limit 2 --output-dir local/test
```

## Configuration

One YAML per region under `config/ingestion/`. Everything that decides which data is
fetched lives there, so a reviewer can see the query without reading Python. The
`gauge:` section holds the bounding box, API settings and the two series names; the
comments in `sydney.yaml` record where the box came from.

## What the gauge fetcher writes

```
database/raw/
  gauge/
    stations.json                          getStationList response, unchanged
    series.csv                             station_id, ts_id, series_name, parameter
    tips/<station_id>/<YYYY-MM>.json.gz    getTimeseriesValues response, unchanged
    daily/<station_id>/<YYYY>.json.gz      getTimeseriesValues response, unchanged
  index.csv
  report.md
```

Responses are stored as the bytes the server returned, gzipped. Nothing is aggregated,
converted, rounded or gap-filled here; that is Component 02's job. The gzip header's
timestamp and filename fields are zeroed so identical responses give identical files
and checksums.

### `index.csv` columns

One row per expected file, whether or not it was fetched successfully.

| Column | Meaning |
|---|---|
| `source` | `gauge_tips` or `gauge_daily` (radar and satellite rows to be added by their owners) |
| `station_id` | BoM internal station id. The key everywhere in this project. Blank for non-gauge rows |
| `path` | relative to `database/raw/` |
| `start`, `end` | the dates the file is supposed to cover, inclusive |
| `n_records` | records in the response. `0` is a dry month, not a failure |
| `status` | `ok` or `failed` |
| `error` | the error message when `failed`, else blank |
| `retrieved_at` | UTC time the file was written (file modification time for files found on disk) |
| `sha256` | hash of the gzip file. Compare two runs to see whether the data changed |

### `report.md`

Human summary: counts of stations and files, stations excluded because a series was
missing (with cause), failed files (with cause), and the rerun command.

## The BoM API

Base URL `https://www.bom.gov.au/waterdata/services`, KiWIS query services. **Use
https**: the http address in the requirements doc now redirects, and the client does
not follow redirects so a moved API fails loudly. Every request carries
`service=kisters&type=queryServices&datasource=0&format=json`.

| Request | Used for | Notes |
|---|---|---|
| `getStationList` | stations in the bounding box | `bbox=min_lon,min_lat,max_lon,max_lat`; also accepts `station_no` |
| `getTimeseriesList` | series ids per station | asking for `coverage` returns `from` and `to` columns |
| `getTimeseriesValues` | the records | `from` and `to` are inclusive dates; a month window loses nothing at either edge |

Series fetched per station:

- `DMQaQc.Merged.AsStored.1`: quality-controlled tips, irregular timestamps. The data.
- `DMQaQc.Merged.DailyTotal.09HR`: BoM's total for each rain day ending 09:00 local.
  The independent check Component 02 uses to tell a dry day from a dead gauge.

List requests return `[[header...], [row...], ...]`. Value requests return a list with
one object per series: `{"ts_id", "rows", "columns", "data"}` where `data` is a list of
`[timestamp, value, quality_code]`.

### Facts about the data that Component 02 needs

Found while verifying the fetcher on 2026-09-05. None are in the requirements docs.

- Timestamps carry a fixed `+10:00` offset year-round. Converting to UTC is a constant
  subtraction; no daylight-saving logic.
- The tip series contains marker records with value `0.0`, typically at fixed times each
  day (09:00, or 03:00 and 12:00, varying by station). They are not rain and are not
  evidence that the gauge observed that hour.
- Millimetres per tip is **not** in the series metadata (`ts_unitname` is just
  `millimeter`). The 02 doc's instruction to read it from metadata cannot be followed;
  it must be inferred from the data (0.5 at every station seen so far) and recorded.
- Quality codes seen: 10, 110, 140, 210, and -1 with a null value. BoM's `getQualityCodes`
  table and what Component 02 does with each are in
  [canonical/README.md](../canonical/README.md).
- Public `station_no` is not unique: 14 numbers in the Sydney box map to two
  `station_id` rows each. One (567163) is a genuine collision of two distant sites; the
  rest are believed to be dual registrations of one gauge. All rows are fetched.
- Station 568163 (the doc's storm check) is JOADJA, in the Southern Highlands, outside
  the Sydney box. The check queries it by `station_no` directly.

## Verifying a run

1. The log ends with a summary line and the process exits 0.
2. `database/raw/report.md` shows failed files: none, and stations with no usable
   series: none (or lists them with cause).
3. `database/raw/index.csv` has 26 rows per station (24 tip months, 2 daily years) and
   every `status` is `ok`.
4. Open one file and look at it:

   ```
   uv run python -c "import gzip, json; d = json.load(gzip.open('database/raw/gauge/tips/31531755/2022-01.json.gz'))[0]; print(d['ts_id'], d['rows'], d['columns']); print(d['data'][:5])"
   ```

5. Run the fetch command a second time. It should finish in seconds with
   `0 fetched this run` and the same index row count.
6. Run the acceptance tests against the live API. Both the storm check (station 568163,
   50 tips and 25.0 mm between 19:00 and 20:00 on 2023-01-04, daily totals 4.0 and
   30.0) and the collision check (`station_no` 563083 returns two stations) must pass:

   ```
   BOM_NETWORK=1 uv run python -m pytest tests/test_ingestion_gauge.py -v
   ```

Without `BOM_NETWORK=1` the same file runs offline against the sample in
`tests/data/ingestion/`, which is what CI and a fresh clone use.

## Open questions

Tracked in [ISSUES.md](../ISSUES.md), not here:

- **ISS-11** — `config/ingestion/sydney.yaml` says `region: sydney` but its bounds cover
  the continent. The request count and runtime above are the Sydney estimates and no
  longer describe what the documented command does.
- **ISS-22** — whether the upper Blue Mountains (Katoomba, Blackheath; longitude below
  150.4) are in scope for Greater Sydney.
- **ISS-02**, **ISS-03**, **ISS-04** — radar and satellite provenance, units, the missing
  satellite coordinates file, and 2022-only coverage.
