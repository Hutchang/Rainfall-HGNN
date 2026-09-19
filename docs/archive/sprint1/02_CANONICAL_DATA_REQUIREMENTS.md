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
| `radar/<YYYY>/<YYYY-MM-DD>.prcp-m15.zip` | NCI daily zip, ~96 NetCDF files/day, each a 512x512 grid, 15-min precipitation accumulation (kg/m², i.e. mm) | Radar rainfall — **shape changed 2026-09-13, see note below** |
| `radar/radar_node_coords.csv` | 262,144 rows: `node_id`, `latitude`, `longitude`, in the same row-major order as `precipitation.values.flatten()` | Places each radar cell on the map |
| `satellite/satellite_2022_preprocessed.pkl` | DataFrame, 8,760 rows: `timestamp` (UTC) + `data` = float32 array shaped (726, 2). Column 0 is a rain rate (0 to ~4.2), column 1 is a 0/1 availability flag (about 86% ones) | Satellite rainfall |

You also need one fact that is in neither the raw files nor the API: **the millimetres
per tip** for each series. BoM's series metadata does not carry it — the KiWIS
`getTimeseriesList` response has no such field (checked 2026-09-05). So infer it per
station as the smallest positive counted record, and write the inferred value to
`stations.csv` so a reviewer can see what was assumed. A station that recorded no
positive tip in the whole period has nothing to infer from; its fallback tolerance comes
from `default_tip_increment_mm` in your config. Do not hard-code 0.5 mm anywhere — that
is the increment at most Sydney stations, not a guaranteed constant.

### Your own config

`config/canonical/standard.yaml`:

```yaml
raw_dir: database/raw
output_dir: database/standard

# Station rules
min_coverage: 0.90            # fraction of the period's judgeable hours to be usable
dead_gauge_days: 90           # consecutive verified-dry days that make a gauge suspect
dead_gauge_radius_km: 50      # ...while this many neighbours inside this radius saw rain
dead_gauge_neighbours: 3
colocated_m: 100              # closer than this is flagged, not excluded; 03 decides
default_tip_increment_mm: 0.5 # tolerance for a station with no positive tip to infer from

# Physical plausibility caps: each rejects the whole rain day
plausibility:
  max_record_mm: 2.0          # a single record above this is not one tip
  max_hourly_mm: 150          # a tip burst, not rain
  max_daily_mm: 400           # a malfunction, whatever BoM's total says

# BoM quality codes. Codes decide which records are counted at all; the daily-total
# check is still the arbiter for a day.
quality:
  tips:
    count: [10, 110, 140]     # A best, C estimate, E unknown
    ignore: [210]             # F missing or incorrect
    gap: [-1]                 # undocumented, always a null value: BoM marking a hole
  daily:
    trusted: [10, 110, 140]   # a total worth comparing against
    dry: [130]                # BoM "no rainfall": a verified total of 0.0
    unverifiable: [-1, 210, 254]
```

Every value that decides what enters the standard dataset belongs here, not in Python.
The client may want the caps, the tolerance or the coverage threshold set differently.

## What you produce

```text
database/standard/
  gauge_hourly.csv
  stations.csv
  days.csv                 the per-rain-day verdicts behind gauge_hourly
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

Two extra columns are worth carrying beside the required ones, because Component 03 and
any reviewer will ask: `coverage` (the fraction below), `tip_increment_mm` (what was
inferred for this station), and `colocated_with` (ids of other active stations within
`colocated_m`, so 03 can keep such a pair in one fold instead of splitting it).

**`days.csv`** — one row per station rain-day that could be judged: `station_id`,
`day_end` (UTC timestamp of the 09:00-local boundary), `n_tips`, `tip_mm`, `daily_mm`,
`daily_quality`, `max_record_mm`, `max_hourly_mm`, `verdict`, `reason`,
`tip_increment_mm`.

This is the evidence behind every hour written or withheld, so it must carry the values
the rules actually fired on — including the two maxima, or an `implausible` verdict
cannot be checked without rerunning the build.

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
hours written per year, daily-total check pass rate, and the station-days that failed
(counts per station are enough, with `days.csv` holding the per-day detail).

## The rules that carry the science

**Which records count.** Every BoM record carries a quality code. Count only the codes
in `quality.tips.count`; never count `quality.tips.ignore`. Compare against a daily total
only when its code is in `quality.daily.trusted`, or treat it as a verified `0.0` when the
code is `quality.daily.dry`. A day whose daily total is unverifiable gets no rows. Codes
are BoM's opinion of a single record; the daily-total check below is the arbiter for a
whole day.

The tip series also contains marker records with value `0.0` at fixed times each day.
They contribute nothing to a sum, and they are **not** evidence that the gauge observed
that hour.

**Dry versus missing.** For each station and each BoM rain day (09:00 to 09:00 local
standard time):

- Sum the counted tips in that day and compare with the downloaded daily total.
- If they agree within one tip increment, the day is **verified**. Write all 24 hours:
  hours with tips get the summed value, hours without tips get `0.0`.
- Otherwise the day is **not verified**. Write no rows for it and record it in
  `report.md`.

So every row in `gauge_hourly` is backed by an independent BoM daily total. This is the
check that would have caught the inherited dataset being wrong, and it is about thirty
lines of code.

Treat the rain day as half-open, `[start, end)`: a tip at exactly 09:00:00 local opens
the day that ends 24 hours later. It has to. The hourly grid is 24 buckets labelled by
their start and covering `[t, t+1h)`, and a `(start, end]` day spans 25 of them, so a
boundary tip would be verified against one day and then written into the next day's rows
— or dropped, if that next day did not verify. The invariant to hold onto: **the 24 hours
written for a verified day sum to the total that was checked against BoM.**

**Plausibility caps.** BoM's daily total is computed from the same tips, so a
malfunctioning gauge fools both sides of the check above. Reject the whole rain day when
any of `plausibility.max_record_mm` (a single record too large to be one tip, so the
series is not tip-resolution that day), `plausibility.max_hourly_mm` (a tip burst too
fast to be rain) or `plausibility.max_daily_mm` (a total no Sydney day reaches) is
exceeded. Record which cap fired. Reject the day rather than the hour: a gauge that
produced one impossible hour was malfunctioning, and the rest of its day is not evidence.

**Dead gauges.** A gauge that has failed shut still reports zero tips and a zero daily
total, so it passes the daily check while being useless. Flag any station with
`dead_gauge_days` consecutive verified-dry days while at least `dead_gauge_neighbours`
stations within `dead_gauge_radius_km` recorded rain. Set `use_in_study = false` with the
reason; do not delete the row.

**Coverage.** `use_in_study = true` requires at least `min_coverage` of the period's
judgeable hours present for that station. Derive that denominator from the period —
whole rain days inside `[start, end]`, so 17,496 hours for 2021–2022, not 17,520 — and
never from the hours that happened to verify. A data-dependent denominator lets one
network-wide outage inflate every station's coverage at once and makes two runs over the
same period incomparable.

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

- [ ] `uv run python -m src.canonical.gauge.build config/canonical/standard.yaml`
      produces the gauge tables, and the radar/satellite half produces the rest.
- [ ] Rerunning with the same input produces identical rows and ordering. (`report.md`
      records its own build time, so it is not byte-identical; the tables are.)
- [ ] A tip at 10:59:59 lands in the 10:00 hour and a tip at 11:00:00 lands in the 11:00
      hour.
- [ ] The 24 hours written for a verified day sum to the tip total that was checked
      against BoM — including when a tip falls on the 09:00:00 rain-day boundary.
- [ ] Timestamps around the 2021 and 2022 daylight-saving changes map to distinct,
      ordered UTC hours — no duplicated or lost hour.
- [ ] The two stations sharing `station_no` `563083`, and the two sharing `567163`,
      remain four separate rows.
- [ ] A verified day yields 24 rows; an unverified day yields none; both appear in
      `report.md`.
- [ ] `coverage` is unchanged by which hours happened to verify: the denominator comes
      from the period.
- [ ] The loader in `src/canonical/` returns `station_id` as a string, timestamps as
      timezone-aware UTC, `use_in_study` as a bool, and the numeric columns as numbers
      even when a station has a blank cell.
- [ ] Tests run offline against the sample Component 01 committed under `tests/data/`.
- [ ] `docs/canonical/README.md` documents every column and the dry-versus-missing rule.

## Open questions

- **Short blockages still produce confident false zeros.** `TO VALIDATE`. A gauge
  blocked for a few days reports no tips, and BoM's daily total — computed from the same
  tips — reports `0.0`, so the day verifies and 24 hours of observed-dry are written. The
  dead-gauge rule only catches runs of `dead_gauge_days`. A day-level check (verified-dry
  while most near neighbours recorded rain) would close it, but the threshold is a
  scientific choice for the client. This is the largest known risk in the gauge tables
  and Component 04 inherits it through the validity channel.
- **Satellite has no coordinates file.** Without it, `satellite_nodes.csv` cannot be
  written and Component 04 cannot connect satellite nodes to gauges. Tracked in
  Component 01's open questions; satellite work is blocked until it appears.
- **Radar's real shape is far bigger than this document assumed, and the row/size
  estimates below (952 nodes, 8.2M rows, 33MB `.npy`) are stale — flagged 2026-09-13,
  not yet resolved with this component's owner.** Component 01 now fetches BoM
  Rainfields 3 `prcp-m15` ("Merged Gauge/Radar Accumulation, 15 min") directly from NCI
  instead of registering a pre-supplied 952-node pkl. Confirmed facts, not guesses:
  units are kg/m² (≈ mm) accumulated over each file's own 15-minute window, not an
  hourly rate — the column name `rain_rate` and the "explode into hourly rows" plan both
  need revisiting; the native grid is 512×512 = 262,144 cells at 0.5 km spacing, not
  952 — whether Component 02 subsamples this down (the source paper describes
  Rainfields subsampled to ~5 km) is a scientific decision for this component's owner to
  make and record, not something Component 01 should decide; and gauge-correction is
  confirmed, with the exact `contributing_gauges` station numbers available per file for
  a leakage check against Component 03's held-out stations. See
  [01's doc](01_DATA_SOURCE_INGESTION_REQUIREMENTS.md) and
  `docs/ingestion/README.md` for the full detail. Agree the resulting shape with the
  Component 04 owner before building it, per the "Long-format CSV" question below.
- **Long-format CSV is probably the wrong shape for the grids.** They are natively 2D
  arrays, and Component 04 pivots them straight back into `[nodes, time, features]`. The
  row/size estimate here was computed for the old 952-node assumption and must be
  redone against the real 262,144-node grid (and whatever subsampling Component 02
  decides on) before choosing a format. Gauge data stays CSV either way — the grids
  alone are affected. Agree this with the Component 04 owner before building it.
- **Radar is no longer 2022-only** — both 2021 and 2022 are available from NCI for
  `prcp-m15`, matching the gauge baseline period. Satellite still covers 2022 only.
  Whether any radar days/hours are missing within that range needs checking against a
  full fetch's `index.csv` once one has been run.
