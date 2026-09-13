# Component 01: source data ingestion

Downloads raw source data to `database/raw/` and writes an index of every file.
Contract: [01_DATA_SOURCE_INGESTION_REQUIREMENTS.md](../requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md).

Sources:

| Source | Status | Code |
|---|---|---|
| Gauges (BoM Water Data Online) | implemented | `src/ingestion/gauge/` |
| Radar (NCI THREDDS, BoM Rainfields 3) | implemented | `src/ingestion/radar/` |
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
- Quality codes seen: 10, 110, 140, 210, and -1 with a null value. Meanings not yet
  looked up.
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

## Radar

Downloads BoM Rainfields 3 "Merged Gauge/Radar Accumulation (15 min)" (product
`prcp-m15`) for the Sydney radar site from NCI's public THREDDS archive
(`rq0/rainfields3`), one day's zip of ~15-minute NetCDF files at a time.

```
uv run python -m src.ingestion.radar.fetch config/ingestion/sydney.yaml
```

Run it again to resume: days already on disk are skipped. To refetch a day, delete its
zip. `radar_node_coords.csv` is written once, the first time the command has at least
one successfully-downloaded day to read a real grid definition from; delete it to
regenerate from whatever is on disk then.

The full run is both years for one site at `prcp-m15` — roughly 719 requests at one per
second, ~1.4 GB, about 20 minutes. **Try it small first**, with `--sample`:

```
uv run python -m src.ingestion.radar.fetch config/ingestion/sydney.yaml --sample
```

This is a single flag that does two things at once, so you don't touch `database/raw/`
while trying the fetcher out: it limits the run to the first 5 days of the configured
period, and writes to `local/radar-sample/` instead of `output_dir`. Pass a number to
change the day count (`--sample 20`); `local/radar-sample/` is gitignored, safe to
delete, and independent of any real data you already have under `database/raw/`.

`--limit N` and `--output-dir DIR` are the same two knobs spelled out separately, if you
want only one of them (e.g. a custom count written to the real `database/raw/`) — either
one, if given, overrides what `--sample` would otherwise pick:

```
uv run python -m src.ingestion.radar.fetch config/ingestion/sydney.yaml --output-dir local/test --limit 5
```

### What the radar fetcher writes

```
database/raw/
  radar/
    <YYYY>/<YYYY-MM-DD>.prcp-m15.zip   NCI's daily zip, unchanged
    radar_node_coords.csv              node_id, latitude, longitude
  index.csv
  report.md
```

Each `index.csv` row has `source=radar_precip`, a blank `station_id`, and `start`/`end`
both set to that day. `n_records` is the count of `.nc` files inside the zip (normally
96 — one per 15 minutes; the server's own `_file_list` manifest entry doesn't count).

`radar_node_coords.csv` has one row per grid cell — 512 × 512 = 262,144 rows, ordered to
match `precipitation.values.flatten()` (the file's own y axis outer, x axis inner) —
computed from a downloaded file's real CF projection metadata (`albers_conical_equal_area`,
via `pyproj`), never assumed. See `src/ingestion/radar/coords.py`.

### The NCI source

Base URL `https://thredds.nci.org.au/thredds/fileServer/rq0/rainfields3`. Site `71` is
`T_Hills` (Terrey Hills), the Sydney radar — confirmed from a downloaded file's own
`station_name` attribute. Files are plain zips at a predictable URL; no catalog listing
or authentication is needed to download one.

Three products exist per site-year; this fetcher uses **`prcp-m15`** only:

| Product | Contents | Size |
|---|---|---|
| `gndrefl` | raw reflectivity volumes | ~100–200 MB/day — too large for routine use |
| `prcp-crate` | instantaneous precipitation rate | ~2–20 MB/day |
| **`prcp-m15`** | Merged Gauge/Radar Accumulation, 15 min | ~2 MB/day, ~1.4 GB for 2021–2022 |

`prcp-m15` is gauge-corrected: each file's `contributing_gauges` attribute lists the BoM
`station_no` values used in the correction. Cross-check this list against the study's
gauge stations before quoting a radar comparison — a held-out test station in that list
means the comparison leaks.

Each `.nc` file's `precipitation` variable is in kg/m² (≈ mm) accumulated over its own
15-minute window — not mm/h, and not hourly. Component 02 sums four files per hour.

### Changing what gets downloaded

Everything that decides *which* radar data is fetched lives in `config/ingestion/sydney.yaml`'s
`radar:` section — nothing is hard-coded in Python, the same principle as the gauge config.

```yaml
radar:
  site_id: "71"          # which BoM radar site
  product: prcp-m15      # which product from that site
  api:
    base_url: https://thredds.nci.org.au/thredds/fileServer/rq0/rainfields3
    request_delay_seconds: 1
    timeout_seconds: 120
    user_agent: COMP3888-P23-ingestion/0.1 (university capstone)
```

| To change... | Edit... | Notes |
|---|---|---|
| **Which radar site** | `radar.site_id` | Any BoM site number from the `rq0` collection, e.g. `https://thredds.nci.org.au/thredds/catalog/rq0/rainfields3/catalog.html` lists them all. Changing this changes both the URL *and* the grid: coordinates are re-derived from that site's own files, never assumed, so `radar_node_coords.csv` will be centred on the new site. Delete the old `radar/` directory (or point `output_dir` somewhere new) rather than mixing sites in one run — the fetcher's consistency check (`assert_consistent_grid`) will refuse to build coordinates from files with different projections, exactly to catch this. |
| **Which product** | `radar.product` | Stick to `prcp-m15` for real work — see the size/gauge-correction table above for why the other two are unsuitable. If you do change it, the daily-file naming pattern (`<site>_<YYYYMMDD>.<product>.zip`) is otherwise the same for all three, so no code changes are needed — only the config value. |
| **Date range** | the top-level `start` / `end` (not inside `radar:`) | Shared with the gauge fetch — one study period for the whole config file. Both 2021 and 2022 are available on NCI for `prcp-m15`; going earlier than 2019 or into a year not yet listed under the site's catalog will 404 per-day, which the fetcher records as a `failed` row rather than crashing. |
| **How polite to be to the server** | `radar.api.request_delay_seconds` | Seconds between requests. Raise it if you see repeated failures that look like rate-limiting; the fetcher already retries transient errors with backoff regardless. |
| **A quick local try before a full run** | nothing in the config — use the `--sample` CLI flag instead (see above) | Keeps experiments out of both `database/raw/` and the YAML. |

A minimal worked example — say you wanted only January 2022, to check the fetcher against
one month without waiting for the full two years:

```yaml
# config/ingestion/sydney-jan2022.yaml — a copy of sydney.yaml with:
start: 2022-01-01
end: 2022-01-31
```

```
uv run python -m src.ingestion.radar.fetch config/ingestion/sydney-jan2022.yaml
```

(Making a separate file rather than editing `sydney.yaml` in place keeps the real
2021–2022 config intact — `load_config` validates whichever file you point it at, so
any config under `config/ingestion/` works the same way.)

### Verifying a run

1. The log ends with a summary line and the process exits 0.
2. `database/raw/report.md` shows failed days: none (or lists them with cause).
3. `database/raw/index.csv` has one row per day in the configured period, `n_records`
   normally 96, every `status` is `ok`.
4. `database/raw/radar/radar_node_coords.csv` has 262,144 rows; spot-check that a
   coordinate near the Sydney bounding box centre looks plausible on a map.
5. Run the fetch command a second time. It should finish in seconds with
   `0 fetched this run` and the same index row count.
6. Run the acceptance tests against the live archive:

   ```
   NCI_NETWORK=1 uv run python -m pytest tests/test_ingestion_radar.py -v
   ```

Without `NCI_NETWORK=1` the same file runs offline against the trimmed real sample in
`tests/data/ingestion/radar/`.

## Open questions

- Whether the upper Blue Mountains (Katoomba, Blackheath; longitude below 150.4) are in
  scope for Greater Sydney. Blackheath G.C. is one of the 563083 collision pair and was
  in the inherited dataset. Project lead to decide; changing the box is one line in the
  YAML.
- Whether any of radar's `contributing_gauges` station numbers belong to a held-out test
  station — check once Component 02/03 produce `standard/stations.csv` and the split.
- Which specific radar days/hours are missing or corrupt in the full 2021–2022 fetch —
  check `index.csv` for `failed` rows and `n_records` below 96 once a full run has been
  done.
- Satellite provenance, units, and the missing satellite coordinates file. See the
  requirements doc's open questions. Owner TBC.
