# D13 — Source grids from NCI: radar and satellite

Template: component standard (the eleven sections in
[AUTHORING_STANDARD.md](AUTHORING_STANDARD.md)) — this deliverable is code with a
contract. Owner: TO CONFIRM. What counts as work, and the rules, are in [README.md](README.md).

## 1. Summary

Read the radar and satellite products in place on NCI's filesystem, declare what they
are from the files themselves, and write the grid tables Component 04 already knows how
to read. This is the "read the source, not the hand export" deliverable: it replaces
the two team-produced pickle files Sprint 1 planned around — files with no metadata and,
for satellite, no coordinates — with the archive products they were cut from. It
resolves ISS-02 (provenance), ISS-03 (satellite coordinates), ISS-04 (coverage) and
ISS-30 (the reader gap) in one go, because all four exist for the same reason.

The data is not downloaded. On Gadi the archive is a mounted directory; the program is
pointed at it. The radar fetcher (`src/ingestion/radar/`) keeps two uses: it produced the
committed real sample that lets tests run without Gadi, and its reader and coordinate
code is reused here. Its download path is not the production path.

## 2. System position

```text
Gadi filesystem (confirmed via the public THREDDS catalog; not yet cross-checked
against a Gadi shell listing)
  /g/data/rq0/rainfields3/<site>/<year>/prcp-m15/<site>_<date>.prcp-m15.zip        radar
  /g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph/v2.1/
    <year>/<month>/<day>/S_NWC_CRRPh_HIMA08_HIMA-N-NR_<date>T<time>Z.nc            satellite
      |
      v
  D13 readers  --declare product, units, time step, correction from the files' own attributes
               --derive node coordinates from the files' own grid definitions
               --align to hours, subsample to a recorded rule
      |
      v
  <run>/canonical/radar_hourly.csv, radar_nodes.csv, satellite_hourly.csv, satellite_nodes.csv,
  grid_report.md                                  (the shapes Component 04 reads today)
      |
      v
  04 graphs (D09.2 edges) --> 06 model (D09.4, D09.5)
```

**Worked example.** Sprint 1's satellite input was an array of 726 grid points per hour
with no file saying where those points are (ISS-03). Nobody can make that file: the 726
were whoever's subsample of a 2 km grid, and the choice was not written down. Reading
the Himawari product itself, each file carries its own latitude and longitude for every
cell, so the question "where are the nodes" is answered by the file, and the subsample
becomes a rule we write down and can change.

## 3. Inputs

| Input | Where | State today |
|---|---|---|
| Rainfields3 `prcp-m15` — merged gauge/radar accumulation, 15-minute, 512 × 512 cells of 0.5 km, one zip per day of 96 NetCDF files, site 71 (Terrey Hills), 2021–2022 | NCI project `rq0`; on Gadi under `/g/data/rq0/rainfields3/…` (TO CONFIRM — the THREDDS catalog shows the layout, the mount path is the standard NCI one) | product, units, time step and `contributing_gauges` confirmed from real files by the fetcher work; nothing reads them downstream |
| Himawari CRRPH — `crrph_intensity`, convective rain rate, physically based, full-disk 5500 × 5500 cells of 2 km, one file per 10 minutes, HIMA08 (Himawari-8/9), 2015–2026 (2021–2022 present) | NCI project `rv74`; on Gadi under `/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph/v2.1/…` (confirmed via the public THREDDS catalog and OPeNDAP metadata, access confirmed 2026-09-26 — not yet cross-checked against a Gadi shell) | product, grid, time step and the `intensity`-vs-`accum` choice confirmed from real files and data; see [grids/SATELLITE_SOURCE.md](../grids/SATELLITE_SOURCE.md); nothing reads them downstream |
| `tests/data/ingestion/radar/71_20220103.prcp-m15.zip` | committed; two real 15-minute files | exists |
| `config/canonical/grids.yaml` | Component 02 | today points at pickle files and declares `product`, `units`, `gauge_correction` as `UNKNOWN` |

## 4. Functionality requirements

### What is being built

Two readers and a report, behind the grid adapter that already exists
(`src/canonical/grids.py`). The adapter today opens one pickle per source; after D13 it
opens the archive's own files through a reader per product, and everything from the
adapter onward (validation, table shapes, the graph builder) is unchanged. Config gains
one root path per source. The subtask id in brackets is the unit of work.

### FR01 [D13.1] — The program knows where the data is, and says so.

**Must do:** `config/canonical/grids.yaml` takes `radar.root` and `satellite.root` —
absolute paths on Gadi, or a local sample directory. `--check-inputs` confirms each root
exists, lists the days found for the configured period, and refuses to continue if a root
is missing, saying which. `docs/grids/README.md` gains a "Where the data is" section:
project codes, mount paths, layout, years present, and how to get access.
**Why:** today nothing in the repository knows the archive exists; the pickle paths are
the only ones written down.
**Done when:** TEST01.

### FR02 [D13.2] — Radar: from daily zips to hourly tables.

**Must do:**
- Open each daily zip, read every NetCDF member's `precipitation` and time, using
  `xarray`, `netCDF4` and `pyproj` (already dependencies); reuse `src/ingestion/radar/`'s
  reader and its coordinate derivation from the file's CF grid mapping.
- Sum four consecutive 15-minute files into one hourly value with an hour-beginning
  UTC timestamp. An hour with fewer than four files is missing, not partial.
- Subsample the 512 × 512 grid to nodes by a rule written in the config and explained
  in one paragraph (for example the cell nearest each point of a 5 km grid), giving on
  the order of a thousand nodes; record the rule and the node count in the grid report.
- Write `radar_hourly.csv` and `radar_nodes.csv` in the shapes the adapter and Component
  04 already accept; declare in `grids.yaml` and the report: product `prcp-m15`, units
  "mm accumulated per hour", time convention "hour beginning, UTC", gauge correction
  "yes — see contributing gauges".
- `radar_hourly.csv` carries a `valid` column, the same shape satellite already has: the
  model cannot accept a null, so a cell missing in one of its four contributing files is
  `rain_rate` 0 with `valid` false, not NaN; an hour where all four files agree on zero
  stays `rain_rate` 0 with `valid` true. The missing/dry distinction lives in `valid`,
  not in the value.
**Why:** this is ISS-30 — the four-part gap between what the archive holds and what the
adapter reads (format, time step, node count, declarations).
**Done when:** TEST02, TEST03, TEST04.

### FR03 [D13.3] — Satellite: from archive files to hourly tables.

**Must do:** the same shape as FR02 for the Himawari product, confirmed against the
real archive on 2026-09-26 (see [grids/SATELLITE_SOURCE.md](../grids/SATELLITE_SOURCE.md)):

- Read `crrph` (not `crr`, which is a 12-class categorical index, not a continuous
  value) from `rv74`, product `crrph_intensity` specifically — the instantaneous
  convective rain rate — not `crrph_accum`, which is the algorithm's own undocumented
  smoothed/rolling estimate and cannot presently be declared from the file the way
  D13's other provenance fields are. Derive coordinates for every cell from the file's
  own geostationary CF grid mapping (`pyproj`, the same method
  `src/ingestion/radar/coords.py` uses for radar's Albers grid, a different
  projection).
- The product's time step is 10 minutes, confirmed from real files (six files per
  hour, not radar's four): align to hours by a rule this project writes down — a
  completeness threshold (how many of the six 10-minute files must be present) and an
  integration rule (e.g. time-weighted average) turning six instantaneous samples into
  one hourly value — the satellite equivalent of FR02's subsample-stride rule, and a
  genuine choice, not a default.
- The native grid is the full 5500x5500 Himawari disk (2 km cells, confirmed via the
  file's own metadata), not pre-cropped to any region the way radar's 512x512 already
  is. A crop-to-study-area step is needed before subsampling by a recorded rule; the
  paper used the native 2 km spacing, so this may be crop-only with no further
  subsampling, TO CONFIRM once implemented.
- Write `satellite_hourly.csv` with its `valid` flag and `satellite_nodes.csv`,
  declare product, units, time convention and correction.

**Why:** ISS-03 disappears the moment coordinates come from the file rather than from a
lost export.
**Done when:** TEST05.

### FR04 [D13.4] — Provenance and the contributing-gauges count in the report.

**Must do:** `grid_report.md` states, per source and taken from the files' attributes,
not typed in: product, units, time step, coordinate system, period covered, days and
hours present and missing, and correction status. For radar it also reports how many
study stations appear in `contributing_gauges` and in how many hours — the size of the
caveat D09 attaches to every radar result. It reports whether the 120 missing radar
hours of ISS-04 exist in the archive.
**Why:** ISS-02 closes on written provenance; D09's caveat needs the count; ISS-04 needs the
archive checked rather than the export.
**Done when:** TEST06.

Landed 2026-09-26: coordinate system and native time step (computed from the files, not
assumed) for both sources; days/hours present and missing against the *configured*
period, not just the span found; and, for radar, the contributing-gauges cross-reference
against Component 02's `stations.csv` (`use_in_study` rows, matched on `station_no`,
graceful when `stations.csv` doesn't exist yet). Closes ISS-02 fully. ISS-04's 120-hour
question is now answerable by the report; nobody has run it against the real archive yet,
so it stays open pending that. See [grids/README.md](../grids/README.md), "Reading
grid_report.md".

### FR05 [D13.5] — Offline samples so the tests run without Gadi.

**Must do:** a committed real satellite sample of a few files (as the radar sample is),
small enough for git, and a `--sample` path for both readers that runs on the committed
samples. The NCI guard tests (`scripts/nci/`) cover the Gadi-only parts.
**Why:** rule 8 — each component commits a small real sample so a fresh clone tests
offline.
**Done when:** the suite passes on a laptop with no archive mounted.

Landed 2026-09-27: `tests/data/ingestion/satellite/` holds six real, consecutive
10-minute Himawari files (one complete hour), each cropped from a real downloaded file
to the Sydney window and reduced to the three variables the reader uses — real,
unmodified values in that window, ~25 KB/file, since a full file is the whole 5500x5500
disk across six variables (~18.6 MB) and can't be committed as-is. The radar sample
(`tests/data/ingestion/radar/71_20220103.prcp-m15.zip`) gained three more real,
byte-identical files from the same archive day, completing one real hour there too — it
previously supported only per-file checks (TEST02), not a full build. `src.canonical
.grids`'s CLI gained `--sample`, pointing every archive-mode layer at its committed
sample and writing to `local/grid-sample/`; running it against the real committed
`grids.yaml` now succeeds end-to-end for both sources with no Gadi access. See
`tests/data/ingestion/README.md` and [grids/README.md](../grids/README.md).

## 5. Non-negotiable rules

- Source-faithful: no unit change, no smoothing, no gap-filling; a missing hour stays
  absent. A missing *cell* within a written hour is the one deliberate exception the
  `valid` column exists for: `rain_rate` 0 with `valid` false is a flagged placeholder
  for the model boundary, not an estimate of what the rain was — the distinction stays
  recorded, never silently lost the way an unflagged 0 or a dropped row would lose it.
- Every declaration in the report comes from a file attribute or a computed count, never
  from an assumption typed into config. `UNKNOWN` is allowed only where the file truly
  says nothing, and then it is named as such.
- Coordinates come from the files. No coordinate is ever reconstructed from a node count
  or an assumed grid.
- The archive is read in place. Downloading it is not the production path.
- The column name `rain_rate` is the adapter's existing contract name, not a claim about
  units; the report's units line is the truth.

## 6. Outputs

The existing grid-table shapes, written under the run folder when run through D07 or a
chosen directory otherwise:

| File | Columns |
|---|---|
| `radar_hourly.csv` | `node_id`, `timestamp`, `rain_rate`, `valid` |
| `radar_nodes.csv` | `node_id`, `latitude`, `longitude` |
| `satellite_hourly.csv` | `node_id`, `timestamp`, `rain_rate`, `valid` |
| `satellite_nodes.csv` | `node_id`, `latitude`, `longitude` |
| `grid_report.md` | per source: declarations from the files, node count and subsample rule, days and hours present and missing, contributing-gauges count |

Read by Component 04 (D09.2 edges, then the graph builder) and, through it, Component 06.

## 7. Testing

| Test | Checks | Subtask |
|---|---|---|
| TEST01 | `--check-inputs` with a missing root refuses and names it; with the sample roots it lists the days found. | D13.1 |
| TEST02 | The radar reader on the committed sample gives the expected values and the expected coordinates for known cells. | D13.2 |
| TEST03 | Four synthetic 15-minute files sum to the right hourly value; three give a missing hour. | D13.2 |
| TEST04 | The subsample rule is deterministic — same input, same node list — and the node count in the report matches the table. | D13.2 |
| TEST05 | The satellite reader gives expected values, coordinates and `valid` flags. Against synthetic fixtures until D13.5 commits a real sample (the same gap radar had between D13.2 and D13.5, closed there by the already-committed `71_20220103.prcp-m15.zip`). | D13.3 |
| TEST06 | `grid_report.md` carries no `UNKNOWN` for radar, and the contributing-gauges count equals a hand count on the sample. | D13.4 |

Coverage: both readers and `src/canonical/grids.py` sit in the 90 % class (rule 10) —
they write the standard tables.

## 8. Documentation requirements

- `docs/grids/README.md`: "Where the data is", how to run on Gadi and on the samples, what
  the report says and how to read the contributing-gauges line.
- `docs/ingestion/README.md`: one paragraph making clear the radar fetcher's download is
  for samples and local tries, and production reads the archive in place.

## 8a. Branching for this deliverable

D13 departs from the general working agreement's one-PR-per-subtask branch (see
[README.md](README.md)) by deliberate choice, recorded here rather than left to
rediscover: each subtask (`feature/D13.1-...`, `feature/D13.2-...`, ...) still gets its
own branch, but they chain onto one another (`D13.2` branches from `D13.1`, `D13.3` from
`D13.2`, and so on) rather than each branching independently from
`integration/sydney-gauge-baseline`. Only one pull request is opened for the whole
deliverable, from the final subtask's branch (renamed `feature/D13`) into
`integration/sydney-gauge-baseline`, once all five subtasks land. This keeps every
subtask's commits individually labelled and reviewable inside that one PR without
needing five separate merges into the integration branch first.

## 9. Dependencies and blockers

Gadi access for whoever runs it ([../nci/README.md](../nci/README.md)) — `rv74`
product and access confirmed 2026-09-26, not yet cross-checked against an actual Gadi
shell listing. D09 owns the gauge-correction caveat (build on the gauge-corrected
product; report the caveat). Issues: ISS-02, ISS-03, ISS-04, ISS-30. D07 is not required
— the readers can be run with a chosen output directory — but runs should go through it
once it lands so the archive files read are fingerprinted in the provenance record.

## 10. Current technical findings

- The fetcher work confirmed from real files: `prcp-m15` is millimetres accumulated per
  15-minute window on an Albers grid of 0.5 km cells, 512 × 512 per site, gauge-corrected,
  with the correcting gauges listed per file. Both 2021 and 2022 are in the archive for
  site 71, so ISS-04's "2022 only" came from the export, not the source.
- The adapter refuses anything not on an hourly boundary and any coordinate file whose
  row count differs from the data; both rules stay and are why the readers must align
  and subsample before handing over.
- Reading the whole 0.5 km grid into the adapter's dense hours × nodes array is not an
  option: Sydney alone would be about 36 TB. The subsample rule is mandatory, and it is
  a scientific choice — the paper used roughly 5 km spacing for radar and the native 2 km
  for satellite.

## 11. This week and backlog

**This week:** D13.1, D13.2.
**Backlog (week 7+):** D13.3, D13.4, D13.5.

All five subtasks landed 2026-09-27, chained on the branches described in section 8a;
one pull request for the whole deliverable is next. Real-archive verification (an actual
run on Gadi) is not part of any subtask here and remains open.
