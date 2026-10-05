# Satellite and radar adapter

This implements the grid-input portion of the satellite/radar work. The adapter covers
the grid reshaping portion of
[02](../archive/sprint1/02_CANONICAL_DATA_REQUIREMENTS.md) and supplies the node-layer
interface described in [04](../archive/sprint1/04_GRAPH_SAMPLE_REQUIREMENTS.md) (both
are the archived Sprint 1 contracts).

It reads the trusted, team-produced Australian pickle exports, validates their
coordinates and times, and writes the existing standard grid tables. It does not fetch
source data, infer satellite coordinates, change rainfall units, build graph edges,
select evaluation splits or train the model.

## Where the data is (D13.1)

The archive is not downloaded for production use; on Gadi it is a mounted directory
and D13's readers point at it in place. This is separate from `src/ingestion/radar/`,
whose download path exists only to produce the committed offline sample and to let
someone try the fetcher without Gadi access.

| Source | NCI project | Layout | Years present | Access |
|---|---|---|---|---|
| Radar (Rainfields3 `prcp-m15`) | `rq0` | `/g/data/rq0/rainfields3/<site_id>/<year>/<product>/<site_id>_<YYYYMMDD>.<product>.zip` — path confirmed via the public THREDDS catalog, not yet cross-checked against a Gadi shell listing | 2021 and 2022 confirmed present for site 71 (T_Hills) — see `src/ingestion/radar/fetch.py` and CURRENT_STATE.md | [Gadi access](../nci/README.md), project `rq0` membership |
| Satellite (Himawari CRRPH `crrph_intensity`, the paper's source) | `rv74` (access confirmed 2026-09-26) | `/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph/v2.1/<year>/<month>/<day>/S_NWC_CRRPh_HIMA08_HIMA-N-NR_<date>T<time>Z.nc` — confirmed via the public THREDDS catalog and OPeNDAP metadata, not yet cross-checked against a Gadi shell listing | 2021 and 2022 confirmed present (site data spans 2015–2026) | project `rv74` membership confirmed |

`rq0` also holds a second, uncorrected radar tree (`level_2/`) considered and set aside
for D13.2 — see [RADAR_SOURCE.md](RADAR_SOURCE.md) for why. `rv74` also holds `crr`
(a categorical rain-class product, not used) and unrelated `cloud`/`solar` product
families under the same `himawari-ahi` tree — see
[SATELLITE_SOURCE.md](SATELLITE_SOURCE.md) for the full investigation and the
`intensity`-vs-`accum` decision.

Point `layers.<name>.root` in `config/canonical/grids.yaml` at the mount path (or a
local sample directory holding files named the same way). For radar that is one site's
folder, `/g/data/rq0/rainfields3/71` for Sydney: `rainfields3/` itself holds every
radar in Australia. Then run:

```console
uv run --locked python -m src.canonical.grid_inputs config/canonical/grids.yaml --check-inputs
```

This confirms each configured root exists and reports how many archive days, within
the config's `start`/`end` period, have a file whose name embeds that date — nothing
is opened or read yet. A `root` of `null` (both sources ship this way on `main`,
since neither location is confirmed) is not checked. A missing root refuses to
continue and names it; this is a separate, earlier check from the pickle-input
`--check-inputs` below, which validates the current hand-exported files.

## Radar: from daily zips to hourly tables (D13.2)

When `layers.radar.root` is set, `build()` (below) reads the archive in place instead
of the pickle: it opens every `*.prcp-m15.zip` daily zip under the root that falls in
the config's `start`/`end` period (the site's other products, `gndrefl` and
`prcp-crate`, are never read), refuses any zip whose grid differs from the first —
another radar site's — and decodes each 15-minute NetCDF member's `precipitation` (CF
scale/offset applied, `_FillValue` masked to NaN) and `start_time`, and sums each
complete run of four consecutive quarter-hours into one hour-beginning UTC value.
Filenames embed each file's *end* time, not its start, so hours are grouped by each
file's own `start_time` floored to the hour — not by the filename — to avoid
misassigning every file to the wrong hour. An hour with fewer than four files is
missing, not written; a cell that is NaN in even one of an hour's four files makes
that cell's hourly value NaN too, the same source-faithful rule applied one level
down. Radar keeps one value per cell-hour, as the paper's gridded nodes do; adding a
validity flag is [ISS-36](../ISSUES.md#iss-36). See `src/canonical/radar_archive.py`.

The native 512x512, 0.5 km grid is far too large to carry densely (~36 TB for
Sydney), so it is subsampled to the cell nearest each point of a `subsample_stride`
grid — stride 10 gives ~5 km spacing, matching the paper's radar spacing, and about
2,700 nodes for one site. This is a recorded scientific choice
(`layers.radar.subsample_stride` in `config/canonical/grids.yaml`), not an
incidental default; see [RADAR_SOURCE.md](RADAR_SOURCE.md) for the fuller picture of
why `rainfields3` is the source at all. The report states product, units, site and
gauge correction from every file's own attributes, and the build stops if two files
disagree. The same fields in `grids.yaml` are notes for readers; the report uses them
only for pickle inputs.

Coordinates and the report's per-source declarations are otherwise unchanged: the
archive reader produces the same `GridData` the pickle path does, so everything from
validation onward — CSV shapes, `to_heterodata`, the report — is the adapter's
existing, unchanged machinery.

## Reading grid_report.md (D13.4)

Each source's section states, taken from the files themselves rather than typed into
config:

- **Product, units, site and gauge correction** — "(from the files)": radar's `title`,
  `precipitation` units (`kg m-2` per 15-minute file; 1 kg m-2 of water is 1 mm) and
  `station_id`/`station_name`; satellite's `product_name`, `title`, `source` and
  `crrph_intensity` units; and, for both, how many files read list
  `contributing_gauges`. Every file must agree, or the build stops. The hourly rule
  (sum of four, mean of six), the satellite crop and the subsample rule follow.
- **Coordinate system** — the file's own `grid_mapping_name` (`albers_conical_equal_area`
  for radar, `geostationary` for satellite).
- **Native time step** — computed, not assumed: radar from `valid_time - start_time`
  within one file (15 minutes); satellite from the smallest gap between consecutive
  files' `nominal_product_time` (10 minutes).
- **Configured period coverage** — hours possible/present/missing against the config's
  own `start`/`end`, not just the span of hours the run happened to find. This is
  deliberately a single count, not an itemised list, so a real multi-year run doesn't
  flood the report; the existing "Absent hours within that range" line above it stays
  scoped to what was actually read and keeps its itemised list. This is also how
  ISS-04's "120 missing hours" question gets answered: compare this count, from a run
  against the real archive, against that hand-export number.
- **Contributing-gauges cross-reference** (radar only) — how many *study* stations (from
  Component 02's `stations.csv`, `use_in_study` rows only, since an excluded station can
  never be a held-out test station) appear in at least one written hour's
  `contributing_gauges`, and in how many of the written hours. This is the actual size
  of D09's leakage caveat, not just "yes, it's gauge-corrected" — see
  [RADAR_SOURCE.md](RADAR_SOURCE.md) and ISS-02. The match is on `station_no`, the only
  key the archive gives us; `station_no` is not a unique key project-wide
  (docs/ingestion/README.md), so the report names this explicitly rather than implying
  more precision than the key supports. If `stations.csv` doesn't exist yet under
  `output_dir` (Component 02 hasn't run), this line says so instead of failing the
  build — a soft cross-component dependency, not a hard one.

## Relationship to Component 06

The grid reshaping here feeds the radar/satellite model support in Component 06. See the
[model runner](../models/README.md) and the [graph integration](../graphs/README.md).
Full real-data acceptance remains separate from successful synthetic tests: source
product, units and gauge-correction provenance are unresolved and tracked in
[ISSUES.md](../ISSUES.md).

## Run

From the repository root:

```console
uv sync --locked
uv run --locked python -m src.canonical.grids config/canonical/grids.yaml --check-inputs
uv run --locked python -m src.canonical.grids config/canonical/grids.yaml
```

Input paths resolve beneath `raw_dir`. Each configured input must have exactly one
`status=ok` row in `raw_dir/index.csv`, using a relative `path`. Pickle can execute
code: use only the team's trusted files, never an arbitrary downloaded pickle. A
layer with `root` set (radar, since D13.2) skips this entirely and reads the archive
in place instead — see "Radar: from daily zips to hourly tables" above. A config where
every layer is archive-mode has no pickle input at all, so `raw_dir/index.csv` isn't
required either.

### Try it without Gadi (D13.5)

```console
uv run --locked python -m src.canonical.grids config/canonical/grids.yaml --sample
```

Points every archive-mode layer (`root` already set) at its committed offline sample
under `tests/data/ingestion/` instead of the configured Gadi root, and writes to
`local/grid-sample/` instead of the real `output_dir`, so it never touches
`database/standard/` or needs a mounted archive. Against the committed `grids.yaml`
this runs both readers successfully end to end: radar's sample is five real files (one
complete hour plus a fragment of the previous one); satellite's is six real files (one
complete hour, cropped to the Sydney window — see
`tests/data/ingestion/README.md` for why it's cropped and radar's isn't). Delete
`local/grid-sample/` before running it again; the existing-output refusal above applies
here too.

The default config requires both modalities. A missing satellite coordinate file
blocks it explicitly. A radar-only config contains only `layers.radar`; it produces
no placeholder satellite files. All enabled inputs are validated before outputs are
written. Existing grid outputs are refused; choose a fresh output directory for a
rerun. Gauge tables and the gauge `report.md` are untouched.

Outputs:

| File | Columns |
|---|---|
| `radar_hourly.csv` | `node_id`, `timestamp`, `rain_rate` |
| `radar_nodes.csv` | `node_id`, `latitude`, `longitude` |
| `satellite_hourly.csv` | `node_id`, `timestamp`, `rain_rate`, `valid` |
| `satellite_nodes.csv` | `node_id`, `latitude`, `longitude` |
| `grid_report.md` | Counts, missing hours/payloads, input paths/hashes and declared source provenance |

Hourly files are ordered by UTC time, then node ID. IDs are the zero-based positions
in the source arrays. Coordinate rows must already have that same order. If a
coordinate file includes `node_id`, the adapter verifies that it is `0..N-1` in order.
Without IDs, positional ordering is an upstream assertion; matching counts alone
cannot prove that the coordinates correspond to the values.

The input radar shape is `[nodes]` per hour; satellite is `[nodes,2]`, with source
availability in column 1. Node counts are validated against coordinates rather than
hard-coded, allowing the documented 952/726-node inputs and small offline fixtures.
Times must be timezone-aware, sorted, unique and exactly on hourly boundaries. UTC
conversion preserves instants; it does not guess whether an hour label marks its
start or end. Missing hours remain absent. Radar NaNs remain missing values in CSV;
satellite invalid payloads and flags are preserved exactly. Observed dry zeros remain
valid zeros. No resampling, smoothing, gap filling, regridding or unit scaling occurs.

`rain_rate` is the existing contract's column name, **not proof of source units**.
For pickle inputs, record product, units, timestamp convention, gauge-correction status
and supporting evidence in the config. These are declarations for review, not automatic
certification. Archive inputs take them from the files (see "Reading grid_report.md").

## Feed Component 04

```python
from src.canonical.grids import load_grid

radar = load_grid(radar_pickle, radar_coordinates, "radar")
layer, coordinates = radar.to_heterodata(approved_graph_timestamps)
```

The result has `layer['radar'].x` shaped `[N,T,1]`, or satellite `[N,T,2]`, plus
`observed` and node IDs. The returned coordinates have `latitude, longitude` in tensor
node order. `layer.timestamps` records the exact requested UTC sequence. The graph
builder owns edges, distances, neighbour counts and attachment to each fold/role.

The caller must provide the approved timeline. The adapter never intersects or drops
target hours automatically. If a requested source hour is missing, it fails with the
first missing timestamp. Radar's one-feature contract cannot represent missing pixels
as distinct from dry zeros, so those also fail at the graph boundary. Agree a common
timeline or a validity-aware modality interface with the graph owner first. For
satellite, invalid rain payloads are zeroed only in the model tensor while the source
0/1 flag is retained; the source arrays and CSV remain unchanged. Tensor conversion is
float32 and checked for overflow.

This creates node features only; it does not choose the disputed satellite topology.
The integrated graph builder validates attachment and fixes the inherited coordinate
ordering. Its numerical choices and actual source alignment still need 04 review.

## Reproducible example

```console
uv run --locked python tests/data/grids/create_sample.py experiments/grid-adapter-demo
uv run --locked python -m src.canonical.grids experiments/grid-adapter-demo/grids.yaml --check-inputs
uv run --locked python -m src.canonical.grids experiments/grid-adapter-demo/grids.yaml
uv run --locked python -m pytest tests/test_grid_adapters.py -q
```

The committed JSON contains two invented nodes and two hours, separated by one missing
hour. It includes dry zeros and an unavailable satellite payload of 999. The CSV keeps
999 with `valid=0`; the graph adapter turns that feature into `[0,0]`. This demonstrates
missingness handling and is not a rainfall research result. The sample creator refuses
an existing destination.

## Remaining production handoff

- **Amos/source owner:** accessible raw radar/satellite files, matching coordinates and
  `ok` index rows, plus product, unit and timestamp provenance (ISS-02, ISS-03, ISS-04).
  Do not manufacture the satellite coordinate file from node count or an assumed grid.
- **Angus/graph owner:** approved fusion timeline, missing-modality treatment, topology
  and successful attachment to the actual fold graphs (ISS-09).
- **Rick/integration:** integrate the adapter and model runner into the Sprint 2
  experiment runner. A full training run is separate acceptance evidence.
- **Scientific review:** confirm the supplied file's provenance and contributing gauges
  before claiming an independent comparison. The inherited Rainfields `prcp-m15` stream
  merges gauges, and holding a gauge out of the graph does not remove information already
  blended into the input — recorded under ISS-02.

The production inputs were not available in this checkout, and the NCI SSH attempt
could not verify the host key. The adapter is testable offline; real-data validation
and downstream graph acceptance must be recorded separately when those inputs arrive.

## Verification — 18 September 2026

This component's acceptance run, kept as dated evidence. The repository's current figures
are in [CURRENT_STATE.md](../CURRENT_STATE.md); the skip count in particular depends on
whether the local checkout has the legacy gauge data.

- Combined test suite: 133 passed, 29 legacy-data/live-API skips. Dependency deprecation
  warnings remain. The tests include a forward/backward pass through the inherited
  HGNN using both modality feature widths and explicitly synthetic edges.
- The 8 NCI guards pass on Linux/macOS and fail on a Windows checkout for a line-ending
  reason unrelated to this component; see [ISSUES.md](../ISSUES.md).
- New Python files pass Ruff; compilation and whitespace checks pass.
- Repository-wide Ruff reports 692 findings in inherited, transitional and other
  pre-existing files.
- The example creator, input check and CSV build run successfully. The production
  input check fails explicitly because `database/raw/index.csv` is absent locally.
