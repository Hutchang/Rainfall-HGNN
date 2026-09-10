# Component 02: the standard gauge dataset

Turns Component 01's raw BoM responses into the two tables every later component reads.
Contract: [02_CANONICAL_DATA_REQUIREMENTS.md](../requirements/02_CANONICAL_DATA_REQUIREMENTS.md).
Radar and satellite reshaping is not included here (blocked on Component 01's provenance
answers; owner TBC).

## Commands

```
uv run python -m src.canonical.gauge.build   config/canonical/standard.yaml   # the tables, ~15 s
uv run python -m src.canonical.gauge.profile config/canonical/standard.yaml   # figures and profile.md, ~1 min
```

Both read `database/raw/` and write to `database/standard/` (gitignored). Rerunning the
build produces byte-identical CSVs. Tests run offline against the Component 01 sample:

```
uv run python -m pytest tests/test_canonical_gauge.py -v
```

## What the build writes

```
database/standard/
  gauge_hourly.csv   one row per verified station-hour
  stations.csv       one row per station in the box, rejected ones included
  days.csv           one row per station rain-day with its verdict (for review)
  report.md          counts, exclusions with cause, rejected days, run settings
  profile/           figures and profile.md from the profile command
```

### `gauge_hourly.csv`

| Column | Type | Meaning |
|---|---|---|
| `station_id` | string | BoM internal id. **Read with `dtype={"station_id": str}`** or pandas makes it an integer and it stops matching other tables |
| `timestamp` | timestamp, UTC | start of the hour, written as `2022-01-01 00:00:00+00:00`. The row covers `[t, t+1h)` |
| `rainfall_mm` | float | millimetres in that hour. `0.0` means the gauge was working and no rain fell |

**A missing hour is an absent row.** There is no NaN and no flag column. Component 04
pivots this table wide and reads `notna()` as the model's validity channel, so writing a
zero for a missing hour would tell the model a broken gauge measured no rain.

Rows exist for every station that had at least one verified day (152 stations), not only
the `use_in_study` ones. Filter by `stations.csv` for the study set.

Use `src.canonical.gauge.build.load_gauge_hourly(path)` and `load_stations(path)` rather
than `read_csv` directly. They apply the string-id and UTC conventions, and
`load_stations` also undoes a trap: `keep_default_na=False` is what keeps an empty
`exclusion_reason` as `""` instead of `NaN`, but it suppresses numeric inference on any
column with a blank cell, so a plain `read_csv` hands back `tip_increment_mm` as strings
as soon as one station has no tips. The loader coerces the numeric columns back and
leaves `use_in_study` a real bool for Component 03 to filter on.

### `stations.csv`

| Column | Meaning |
|---|---|
| `station_id`, `station_no`, `name` | BoM internal id (the key), public number (display only, not unique), name |
| `latitude`, `longitude` | WGS84, from BoM's station list |
| `hours_observed` | rows in `gauge_hourly.csv` for this station |
| `coverage` | `hours_observed` divided by the judgeable window (17,496 hours for 2021-2022) |
| `days_verified`, `days_rejected` | rain days that passed, and that failed as mismatch or implausible |
| `tip_increment_mm` | millimetres per tip, inferred as the smallest positive counted record (BoM's metadata does not state it). 0.5 at 144 stations, 0.2 at 4, 0.1 at 3, 1.0 at 1; blank where the station recorded no positive tip |
| `colocated_with` | ids of other active stations within `colocated_m` (100 m). Not an exclusion: Component 03 keeps such pairs in the same fold |
| `use_in_study` | passed every station rule |
| `exclusion_reason` | why not, else empty |

### `days.csv`

One row per station rain-day inside the period: tips counted, their sum, BoM's total and
its quality code, the largest single record and largest hour, the verdict, and for
implausible days which cap fired. This is the evidence behind every hour written or not.

## The rules, in the order they apply

Everything below is set in `config/canonical/standard.yaml`; nothing is hard-coded.

### 1. Which records count

BoM attaches a quality code to every record (table from the KiWIS `getQualityCodes`
request, 2026-09-05):

| Code | BoM label | Tips | Daily totals |
|---|---|---|---|
| 10 | quality-A, best available | counted | trusted |
| 110 | quality-C, estimate | counted | trusted |
| 140 | quality-E, unknown | counted | trusted |
| 130 | no rainfall | | verified total of 0.0 |
| 210 | quality-F, missing or incorrect | ignored | unverifiable |
| -1 | undocumented, always a null value | gap marker, never a value | unverifiable |
| 254 | undocumented | | unverifiable |

Codes are BoM's opinion of a record; the daily check below is the arbiter for a day. A
code-140 tip survives only if the day it sits in verifies. The tip series also contains
marker records with value `0.0` at fixed times each day; they contribute nothing to sums
and are not evidence the gauge observed that hour.

### 2. Dry versus missing: the daily check

For each station and BoM rain day, 09:00 to 09:00 local standard time (BoM serves a
fixed `+10:00` offset year-round, so this is always 23:00 UTC to 23:00 UTC):

- sum the counted tips in `(start, end]`;
- compare with BoM's daily total for that day;
- if they agree within **one tip increment** (per station) the day is **verified**: all
  24 hours are written, tips summed by UTC hour, `0.0` where none fell;
- otherwise the day is **mismatch** and nothing is written;
- if BoM has no usable total (codes -1, 210, 254, or no record) the day is
  **unverifiable** and nothing is written.

Only rain days fully inside the period are judged, so the period's first 9 and last 15
local hours are never written.

### 3. Plausibility caps

BoM's daily total is computed from the same tips, so a stuck gauge passes the daily check
(Shanes Park, 29 July 2021: 1,750 tips, 875 mm, BoM total 875 mm). Three caps reject the
whole day, each catching a different failure:

| Cap | Default | Failure it catches |
|---|---|---|
| `max_record_mm` | 2.0 | a single record that is not a tip: the series stores totals, not tips, that day |
| `max_hourly_mm` | 150 | a burst of tips too fast to be rain |
| `max_daily_mm` | 400 | a total no Sydney day reaches (record about 330 mm) |

Rejecting the day, not the hour, is deliberate: a gauge that produced one impossible hour
was malfunctioning and the rest of its day is not trustworthy.

### 4. Station rules

- **no records in period**: no verified day at all (32 stations in the Sydney box).
- **coverage**: `hours_observed / judgeable hours` below `min_coverage` (0.9).
- **dead gauge**: `dead_gauge_days` (90) consecutive verified-dry days during which at
  least `dead_gauge_neighbours` (3) stations within `dead_gauge_radius_km` (50) recorded
  rain. A blocked gauge reports zero tips and BoM records a zero total, so it passes the
  daily check while being useless. Rogan Hill reservoir was dry for 265 days in 2021.

Rejected stations keep their row with `use_in_study = false` and the reason.

## Decisions made here, for review

- **Tip increment is inferred, not read.** The requirements doc says to read millimetres
  per tip from series metadata. That field does not exist in the API; the increment is
  the smallest positive counted record at each station and is recorded in `stations.csv`.
- **Tolerance is one increment.** Tight by design: 282 days fail on a single tip's
  difference. Loosening it is one line in `verify.py`.
- **Co-located stations are kept, not merged.** Fourteen public numbers in the box map
  to two internal ids. Twelve pairs have one silent member; two pairs (Linden, Prospect
  Reservoir) are both active, 20 m apart, and disagree on 28-43% of days. They are
  flagged in `colocated_with` for Component 03 to keep in one fold. The inherited audit's
  "13 duplicates with identical data" does not hold for the API series.
- **`gauge_hourly.csv` includes non-study stations.** Filter on `stations.csv`.

## What the Sydney run produced (2026-09-08)

| | |
|---|---|
| stations in box | 184 |
| use_in_study | 127 |
| excluded | 32 no records, 24 coverage, 1 dead gauge |
| rain-days judged | 108,001 |
| verified | 98,859 (91.5% of days with a usable BoM total) |
| unverifiable | 8,439, of which 8,244 are BoM's explicit gaps |
| implausible | 421 (420 per-record, 1 per-hour) |
| mismatch | 282 |
| hourly rows | 2,372,616 (2,162,736 at study stations) |
| max hourly value | 62.0 mm, March 2022 floods |

Notable in the data: a network-wide gap in BoM's daily totals for about three weeks in
April-May 2021 across roughly 30 stations; three series named `STP-*` / `WTP-*` that are
daily loggers, not tipping buckets; Gladesville at 89.85% coverage, just under the line.

## Open questions

- **Short blockages produce confident false zeros.** A blocked gauge reports no tips
  and BoM's daily total, computed from the same tips, reports 0.0, so the day verifies
  and 24 hours of `0.0` are written as observed-dry. The dead-gauge rule only catches
  runs of 90+ days. A day-level check (verified-dry while most near neighbours recorded
  rain) would close this; the threshold is a scientific choice for the client. This is
  the top risk in the current tables.
- **Timestamp fidelity within a day** is assumed. If any logger batch-reports, the day
  total is still right but the hourly distribution is wrong; only BoM can confirm.
- Whether the client wants the plausibility caps, tolerance, or coverage threshold set
  differently. All are config values.
- The cause of the April-May 2021 gap (per-owner outage?).
- Radar and satellite tables: blocked on Component 01's provenance answers.
