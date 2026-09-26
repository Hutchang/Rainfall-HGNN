# Satellite source: what's on `rv74`, and why D13.3 reads `crrph_intensity`

Investigation note for D13.3, recorded 2026-09-26. Read this alongside
[ISS-03](../ISSUES.md#iss-03), which it provides evidence for. Confirmed by browsing
NCI's public THREDDS catalog and OPeNDAP metadata/data endpoints directly — no Gadi
shell access was used or needed; the catalog and OPeNDAP service mirror the mounted
archive.

## Where the data is

```text
/g/data/rv74/satellite-products/arc/der/himawari-ahi/
  cloud/    cma (cloud mask), cmic (cloud microphysics), ct (cloud type),
            ctth (cloud-top temperature/height) -- not rainfall, not read by D13
  precip/   crr/v2.0 (categorical rain-rate class, 12 bins) and
            crrph/v2.1 (continuous rain rate, mm/h) -- crrph is the one the
            paper cites [6] and D13.3 reads
  solar/    p1s/p1h/p1d (10-min/hourly/daily solar irradiance) -- unrelated
```

`crr`'s main variable is a 12-class categorical index (bins like "2-3 mm/h", "50+
mm/h"), not a continuous value -- confirmed from its own attributes (`flag_values`,
`flag_meanings`). It cannot support a regression target. `crrph` gives two continuous
`mm/h` variables instead, confirmed present for 2021-2022 at `crrph/v2.1/<year>/<month
>/<day>/S_NWC_CRRPh_HIMA08_HIMA-N-NR_<YYYYMMDD>T<HHMMSS>Z.nc`, 144 files/day (10-minute
cadence, not radar's 15-minute).

## The grid

5500x5500 cells at 2 km, the **full Himawari disk** (81°S-81°N), confirmed from the
file's own `.dds`/`.das` (OPeNDAP metadata, fetched without downloading the file).
Unlike radar's 512x512, this is not pre-cropped to any region — D13.3 needs its own
crop-to-study-area step, which radar didn't. There is no stored `lat`/`lon` array
either (same situation as radar): coordinates come from the file's own CF grid-mapping
attributes (`+proj=geos +lon_0=140.7 +h=35785863 ... +sweep=y`), decoded with `pyproj`
the same way `src/ingestion/radar/coords.py` already does for radar's Albers
projection -- a different projection, the identical method.

## `crrph_intensity` vs `crrph_accum`

The file carries two rain-rate variables and it is not obvious from their attributes
alone which one an hourly table should use, so this was checked against real data: six
consecutive 10-minute files across one hour (2022-01-03, 02:30-03:50 UTC) at a strong
convective cell near Darwin, NT (132.2°E, -14.9°S; January is the wet season there,
which is why this cell was picked over anything in Sydney -- Sydney had nothing this
strong to check the mechanism against on the sampled day):

| Time (UTC) | `crrph_intensity` (mm/h) | `crrph_accum` (mm/h) |
|---|---|---|
| 02:30 | 49.8 | 29.3 |
| 02:40 | *(file absent from the archive -- confirmed 404, not a query error)* | |
| 02:50 | 49.8 | 45.1 |
| 03:00 | 49.8 | 49.3 |
| 03:10 | 49.8 (product ceiling) | 49.6 |
| 03:20 | 40.1 | 49.6 |
| 03:30 | 27.4 | 47.9 |
| 03:40 | 16.4 | 44.3 |
| 03:50 | 26.4 | 39.1 |

`intensity` is a true instantaneous snapshot: it saturates at the product's ceiling
while the cell is strongest, then tracks its real, noisy decay (27.4, 16.4, back up to
26.4). `accum` does not move with it: it keeps climbing after `intensity` has already
peaked, and keeps falling for two frames after `intensity` has already dropped and
partly recovered. That lag is the signature of an internal temporal-smoothing or
short-term-persistence step inside the NWC-SAF algorithm, not a running sum of the
`intensity` samples -- nothing in the file's own attributes documents what window it
smooths over, so it cannot presently be declared from the file the way D13's other
provenance fields are. The 02:40 gap also confirms the same class of missing-file
problem radar has, at the 10-minute level.

## The two paths, and the decision

**Path A -- read `crrph_intensity`, integrate six 10-minute samples into an hourly
value ourselves**, the same shape of work as radar's `hourly_grids()`: decode each
file, define and record a completeness rule (an "N of 6" threshold, radar's "4 of 4"
counterpart), and a stated integration rule. More implementation work, but every
number in the output table traces to a rule this project wrote down and can defend,
matching the non-negotiable rule that a declaration comes from the file or a computed
count, never an unexamined third-party smoothing window.

**Path B -- read `crrph_accum` once per hour** (the file nominally timestamped
`HH:00:00`), no combining logic at all. Far less work, but it inherits an
undocumented smoothing window, and it is a single point of failure per hour instead of
six independent looks.

**Decision: Path A.** `crrph_intensity`, integrated hourly by a rule this project
writes down and tests, the same standard FR02 already held radar to. This also keeps
faith with what D13.3 is for: the source paper's Sydney testbed runs at hourly
resolution using "Himawari-8/9 CRRPH" [6] without specifying which of its two
variables or how it reached an hourly value, so there is nothing in the paper itself
to reproduce numerically here -- the choice is ours, made in the direction that keeps
the baseline's derivation transparent and reproducible rather than borrowing an
opaque smoothing window, so that a from-source Sydney baseline can actually be built
and defended. Not yet decided: the exact completeness threshold (how many of the six
files must be present for an hour to count) and the integration rule (e.g.
time-weighted average vs. trapezoidal) -- these are D13.3's version of D13.2's
subsample-stride choice and belong in that implementation, recorded the same way.
