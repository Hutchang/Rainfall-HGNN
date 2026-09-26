# Radar source: rainfields3 vs. level_2, and why we stayed with rainfields3

Investigation note for D13, recorded 2026-09-26. Read this alongside
[ISS-02](../ISSUES.md#iss-02), which it provides evidence for. It does not change
[D13_SOURCE_GRIDS_REQUIREMENTS.md](../requirements/D13_SOURCE_GRIDS_REQUIREMENTS.md) — that
document already scopes D13.2 around `rainfields3`'s `prcp-m15` product and already
records the gauge-correction caveat as accepted, not open. This note exists because the
question "should we use the uncorrected archive instead" came up while confirming Gadi
paths, and the answer — no ready rainfall product exists there — is worth keeping on
record rather than re-discovering later.

## The question

NCI project `rq0` (confirmed mounted at `/g/data/rq0` on Gadi) holds two separate radar
trees for the same physical sites, including site 71 (T_Hills, Sydney):

- `rainfields3/` — the merged gauge/radar product the fetcher (`src/ingestion/radar/`)
  already downloads samples from. It is **gauge-corrected**: each file's
  `contributing_gauges` attribute lists the BoM stations blended into it.
- `level_2/` — per-site, per-product radar output, **not** gauge-corrected.

Gauge correction is the substance of ISS-02: if a held-out test station's gauge readings
were used to correct the radar field, holding that gauge out of the graph does not remove
its influence from the radar input, and any claim that the radar result is independent of
the gauge network is false. An uncorrected source would remove that leak outright. So
before committing D13.2 to `rainfields3`, it is worth asking whether `level_2` is a live
alternative.

## What `level_2` actually contains

Confirmed by browsing the public THREDDS catalog (no Gadi login needed — the catalog
mirrors the mounted filesystem layout 1:1 for this archive):

```text
https://thredds.nci.org.au/thredds/catalog/rq0/level_2/71/catalog.html
  AZSHEAR/               azimuthal shear
  COLUMNMAXREFLECTIVITY/ column-maximum reflectivity
  ECHOTOPHEIGHT/         echo top height
  REFLECTIVITY/          raw reflectivity (dBZ), one file per day
  SHI/                   severe hail index
  STEINER/               convective/stratiform classification
```

Filenames follow `<site>_<YYYYMMDD>_<product_lowercase>.nc` (e.g.
`71_20090516_reflectivity.nc`), one file per day, flat under `<site>/<PRODUCT>/` — no
year subfolder, unlike `rainfields3`.

**None of these six products is a rainfall accumulation.** The closest candidate,
`REFLECTIVITY`, is dBZ — a measure of returned radar power, not rainfall in millimetres.
Turning it into rainfall requires a Z-R relationship (a reflectivity-to-rain-rate
conversion, typically calibrated per radar/climate and itself a source of error) applied
to every cell of every scan. `STEINER` classifies convective vs. stratiform rain, which
a Z-R conversion would need to pick the right coefficients, but is not a rain value
itself. Nothing under `level_2` is ready to write into `radar_hourly.csv`'s `rain_rate`
column without that conversion step being designed, implemented and validated first —
which is a scientific undertaking in its own right, not a parsing task, and is not
currently scoped anywhere in Sprint 2.

## What `rainfields3` contains

Confirmed the same way:

```text
https://thredds.nci.org.au/thredds/catalog/rq0/rainfields3/71/catalog.html
  2019/ 2020/ 2021/ 2022/ 2023/ 2024/ 2025/ 2026/
    gndrefl/       raw reflectivity volumes (~100-200 MB/day)
    prcp-crate/    instantaneous precipitation rate
    prcp-m15/      Merged Gauge/Radar Accumulation, 15 min  <- what D13.2 targets
```

`prcp-m15` is already a rainfall accumulation in millimetres, already confirmed present
for both 2021 and 2022 at site 71 (this checkout's 2026-09-26 catalog browse; the
fetcher's earlier work found the same from downloaded files). It is the only product in
either tree that is immediately usable as `rain_rate` without new science.

## The actual tradeoff

> The only radar rainfall product this project has archive access to is already
> gauge-corrected. An uncorrected radar rainfall product does not exist in `rq0`; getting
> one means deriving it ourselves from `level_2` reflectivity via a Z-R relationship.

That is a real choice between two costs, not a bug to fix:

| | `rainfields3` `prcp-m15` | `level_2` `REFLECTIVITY` + Z-R |
|---|---|---|
| Ready to use | Yes | No — conversion must be designed and implemented |
| Gauge-independent | No — carries ISS-02's leak | Yes |
| Scientific risk | Known and stated (D09's caveat) | New: Z-R coefficient choice, validation, and its own error, unscoped |
| Matches the source paper | Yes — the paper also used a gauge-corrected product (§4.1) | No — a paper deviation, and a bigger one than most Sprint 2 gaps |

## Decision

D13.2 stays targeting `rainfields3`'s `prcp-m15`, as `D13_SOURCE_GRIDS_REQUIREMENTS.md`
already scopes it, carrying the gauge-correction caveat that D09 already states on every
radar result (ISS-02). This is not a new decision — it is confirming the existing one
after checking whether `level_2` offered a way out, and finding that it doesn't without
taking on Z-R conversion as new, unscoped work.

`level_2` is not being discarded as an option: if the team later decides the Z-R
conversion work is worth doing (a real gauge-independent radar comparison would be a
meaningfully stronger result), the archive path and product list above are the starting
point, and it would be a new deliverable rather than a change to D13.2.

## Gadi paths confirmed for D13.1

Both found by browsing the public THREDDS catalog, which mirrors the mounted filesystem
path (`/g/data/<project>/...`) for this archive:

- `rainfields3`: `/g/data/rq0/rainfields3/<site_id>/<year>/<product>/<site_id>_<YYYYMMDD>.<product>.zip`
- `level_2`: `/g/data/rq0/level_2/<site_id>/<PRODUCT>/<site_id>_<YYYYMMDD>_<product_lowercase>.nc`

Neither has been verified against an actual Gadi shell listing yet (this session has no
Gadi access) — only against the public catalog, which normally tracks the real mount
1:1 but has not been cross-checked with an `ls` on Gadi itself. Whoever runs D13.2 on
Gadi should confirm the exact path resolves before relying on it.
