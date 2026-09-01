# 02 — Build the standard tables

**Owner:** _unassigned_ · **Reviewer:** Component 04 owner
**May be split between two people:** gauges (the hard part) and radar/satellite.

## What this component is for

Turn Component 01's raw files into the small set of tables that everyone else uses.
After this component, nobody parses a BoM API response, guesses units, or worries about
timezones again.

Four real jobs. Everything else in this document is packaging:

1. **Tips into hourly totals.** A tipping-bucket gauge records one event each time it
   collects a fixed increment of rain. Three tips at 10:07, 10:12 and 10:41 with a
   0.5 mm increment is 1.5 mm for the hour starting 10:00.
2. **Local time into UTC.** BoM serves timestamps at a fixed UTC+10 offset. Sydney local
   clock time is ambiguous across daylight-saving changes; UTC is not, and it is the
   only way gauge, radar and satellite land on one timeline.
3. **Dry apart from missing.** This is the one that matters.
4. **One row per station**, with coordinates, so Component 04 can build a graph.

## What you receive

From Component 01, `database/raw/`:

| Input | Shape | Why you need it |
|---|---|---|
| `index.csv` | one row per raw file, with `status` | Tells you what should exist. Read this *instead of* listing the directory, so a failed download cannot be mistaken for a dry month |
| `gauge/tips/<station_id>/<YYYY-MM>.json.gz` | BoM response: irregular timestamps, one row per tip | The rainfall |
| `gauge/daily/<station_id>/<YYYY>.json.gz` | BoM response: one total per 09:00-to-09:00 rain day | Your independent check that the gauge was working. Job 3 is impossible without it |
| `gauge/stations.json` | `station_id`, `station_no`, name, lat, lon, owner | Coordinates and identity |
| `radar/radar_2022_preprocessed.pkl` | DataFrame, 8,640 rows: `timestamp` (UTC) + `data` = float32 array of 952 values | Radar rainfall |
| `radar/radar_node_coords.csv` | 952 rows: `latitude`, `longitude`, in the same order as the array | Places each radar value on the map |
| `satellite/satellite_2022_preprocessed.pkl` | DataFrame, 8,760 rows: `timestamp` (UTC) + `data` = float32 array shaped (726, 2). Column 0 is a rain rate (0 to ~4.2), column 1 is a 0/1 availability flag (about 86% ones) | Satellite rainfall |

You also need one fact from Component 01 that is not in the files: **the millimetres per
tip** for each series. Read it from the series metadata. Do not hard-code 0.5 mm — that
is the value observed so far, not a guaranteed constant.

### Your own config

`config/canonical/standard.yaml`:

```yaml
raw_dir: database/raw
output_dir: database/standard
min_coverage: 0.90          # station needs this fraction of verified hours to be usable
dead_gauge_days: 90         # consecutive verified-dry days that make a gauge suspect
```

## What you produce

```text
database/standard/
  gauge_hourly.csv
  stations.csv
  radar_hourly.csv
  radar_nodes.csv
  satellite_hourly.csv
  satellite_nodes.csv      (blocked — see Open questions)
  report.md
```

Plain CSV, so anyone can `head` the file, grep it, or open it in Excel. CSV carries no
column types, so two conventions are mandatory and everything downstream depends on them:

- **Write timestamps as ISO8601 with the UTC offset**: `2022-01-01 00:00:00+00:00`, never
  `2022-01-01 00:00:00`. A naive timestamp does not say whether it is UTC or Sydney local
  time, and nobody can recover the answer later. The legacy export the team is using
  right now has exactly this bug.
- **Read `station_id` as a string**: `pd.read_csv(path, dtype={"station_id": str})`.
  Otherwise pandas makes it an `int64`, it stops matching the ids in the split file and
  the station table, and rows silently join to the wrong station.

Put both in the loader in `src/canonical/` so no teammate has to remember them.

**`gauge_hourly.csv`** — one row per observed station-hour:

| Column | Type | Notes |
|---|---|---|
| `station_id` | string | BoM internal id |
| `timestamp` | timestamp, UTC | start of the hour; the row covers `[t, t+1h)` |
| `rainfall_mm` | float | `0.0` means the gauge was working and no rain fell |

**A missing hour is an absent row.** There is no `NaN` and no flag column. If an hour is
not in this table, it was not observed. Component 04 relies on exactly this: it pivots
the table wide and reads `notna()` as the model's validity channel.

**`stations.csv`** — one row per `station_id`, including the ones you rejected:

| Column | Type | Notes |
|---|---|---|
| `station_id`, `station_no`, `name` | string | `station_no` is display only and is not unique |
| `latitude`, `longitude` | float | WGS84, straight from BoM |
| `hours_observed` | int | rows in `gauge_hourly` for this station |
| `use_in_study` | bool | passed coverage and quality checks |
| `exclusion_reason` | string | empty when `use_in_study` is true |

Keep rejected stations as rows with `use_in_study = false`. Deleting them makes it
impossible to answer "why is this station not in the results?".

**`radar_hourly.csv`**: `node_id` (int, 0–951), `timestamp` (UTC), `rain_rate`.
**`radar_nodes.csv`**: `node_id`, `latitude`, `longitude`.
**`satellite_hourly.csv`**: `node_id` (int, 0–725), `timestamp`, `rain_rate`, `valid`
(0/1, straight from source column 1 — the model reads it as a feature).
**`satellite_nodes.csv`**: `node_id`, `latitude`, `longitude`.

The two `*_hourly` grid files are big in this shape — radar is 952 nodes × 8,640 hours =
8.2M rows, about 270MB. See **Open questions** before you commit to it.

For radar and satellite the work is only reshaping: explode the array into rows, join the
coordinates, keep UTC. Do not regrid, resample, smooth, or rescale. Nobody has confirmed
what the values mean yet, so any transformation now is a guess.

**`report.md`** — for humans: stations found, stations used, stations rejected and why,
hours written per year, daily-total check pass rate, and the list of station-days that
failed.

## The three rules that carry the science

**Dry versus missing.** For each station and each BoM rain day (09:00 to 09:00 local
standard time):

- Sum the tips in that day and compare with the downloaded daily total.
- If they agree within one tip increment, the day is **verified**. Write all 24 hours:
  hours with tips get the summed value, hours without tips get `0.0`.
- Otherwise the day is **not verified**. Write no rows for it and record it in
  `report.md`.

So every row in `gauge_hourly` is backed by an independent BoM daily total. This is the
check that would have caught the inherited dataset being wrong, and it is about thirty
lines of code.

**Dead gauges.** A gauge that has failed shut still reports zero tips and a zero daily
total, so it passes the check above while being useless. Flag any station with
`dead_gauge_days` consecutive verified-dry days while at least three stations within
50 km recorded rain. Set `use_in_study = false` with the reason; do not delete the row.

**Coverage.** `use_in_study = true` requires at least `min_coverage` of the period's
hours present for that station.

## How your output is used

Component 03 reads `stations.csv`, filters `use_in_study`, and clusters on
`latitude`/`longitude` to choose train/validation/test stations.

Component 04 reads both gauge tables and does essentially this:

```python
wide = gauge_hourly.pivot(index="timestamp", columns="station_id", values="rainfall_mm")
x = torch.stack(
    [
        torch.tensor(wide.fillna(0).values.T),  # channel 0: rainfall
        torch.tensor(wide.notna().values.T),  # channel 1: validity  <- your absent rows
    ],
    dim=2,
)
```

That is why the absent-row rule is not cosmetic: it is literally the model's validity
channel. If you write `0.0` for a missing hour, the model is told a broken gauge measured
no rain, and the results are wrong in a way no test will catch.

## Done when

- [ ] `python -m canonical.build config/canonical/standard.yaml` produces every table.
- [ ] Rerunning with the same input produces identical rows and ordering.
- [ ] A tip at 10:59:59 lands in the 10:00 hour and a tip at 11:00:00 lands in the 11:00
      hour.
- [ ] Timestamps around the 2021 and 2022 daylight-saving changes map to distinct,
      ordered UTC hours — no duplicated or lost hour.
- [ ] The two stations sharing `station_no` `563083`, and the two sharing `567163`,
      remain four separate rows.
- [ ] A verified day yields 24 rows; an unverified day yields none; both appear in
      `report.md`.
- [ ] Tests run offline against the sample Component 01 committed under `tests/data/`.
- [ ] `docs/canonical/README.md` documents every column and the dry-versus-missing rule.

## Open questions

- **Satellite has no coordinates file.** Without it, `satellite_nodes.csv` cannot be
  written and Component 04 cannot connect satellite nodes to gauges. Tracked in
  Component 01's open questions; satellite work is blocked until it appears.
- **Radar units are unconfirmed.** If the 952 values turn out to be an accumulation
  rather than a rate, the column name `rain_rate` is wrong and Component 04's feature
  scaling changes. Do not convert anything until Component 01 answers.
- **Long-format CSV is probably the wrong shape for the grids.** They are natively 2D
  arrays, and Component 04 pivots them straight back into `[nodes, time, features]`, so
  the long form costs roughly 320MB and a pivot to gain nothing. Saving
  `radar_hourly.npy` (float32, `[8640, 952]`, 33MB) beside `radar_nodes.csv` is one line
  to load and already matches the shape Component 04 wants. Gauge data stays CSV either
  way — the grids alone are affected. Agree this with the Component 04 owner before
  building it.
- **Radar is missing 120 hours of 2022** and both grids cover 2022 only, while gauges
  cover 2021–2022. Expect `radar_hourly` and `satellite_hourly` to be shorter than
  `gauge_hourly`; say so in `report.md` rather than padding them.
