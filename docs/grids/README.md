# Satellite and radar adapter

This implements the grid-input portion of the satellite/radar work Oliver accepted in
the 10 September Teams discussion. This adapter covers the grid reshaping portion of
[02](../requirements/02_CANONICAL_DATA_REQUIREMENTS.md) and supplies the node-layer
interface described in [04](../requirements/04_GRAPH_SAMPLE_REQUIREMENTS.md).

It reads the trusted, team-produced Australian pickle exports, validates their
coordinates and times, and writes the existing standard grid tables. It does not fetch
source data, infer satellite coordinates, change rainfall units, build graph edges,
select evaluation splits or train the model.

## Scope checked in Teams and Notion — 10 September 2026

The native Notion app in **Kiran McLernon's Space / Rainflow** was checked directly.
[Sprint One](https://www.notion.so/3cede437d58a8039a588db42a9838aa9) assigns **Oliver**
to **06 — Model Training, Prediction and Baselines**. Its checklist includes adapting
the model for radar and satellite, the prediction schema and checker, training, IDW,
network mean, and comparison. Its direct dependencies are 03 (Kiran) and 04 (Angus).

The [D06 card](https://www.notion.so/3cede437d58a80d3a1b5e6574b31fdf2) is titled
“Deliverable D06 - Open”, remains To do, has Rick Honda in its Person property, and
links to the repository's training/prediction requirements. That Person property
conflicts with the explicit Oliver ownership in Sprint One; it is not evidence that
Rick has taken over implementation.

In Teams on 10 September, Rick's 20:06 list called 06 “Satellite & Radar adapter
(backlog)”; Oliver accepted that work at 20:48. Treat this as the immediate requested
slice, while keeping the broader written 06 deliverable visible. This grid adapter
alone does **not** complete the Notion checklist. The local `feature/06-model-training`
implementation handles training/predictions/baselines and the corresponding model
input adapter; the two branches are separate local handoffs pending integration.

## Run

From the repository root:

```console
uv sync --locked
uv run --locked python -m src.canonical.grids config/canonical/grids.yaml --check-inputs
uv run --locked python -m src.canonical.grids config/canonical/grids.yaml
```

Input paths resolve beneath `raw_dir`. Each configured input must have exactly one
`status=ok` row in `raw_dir/index.csv`, using a relative `path`. Pickle can execute
code: use only the team's trusted files, never an arbitrary downloaded pickle.

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
Record product, units, timestamp convention, gauge-correction status and supporting
evidence in the config. These are declarations for review, not automatic certification.

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
The inherited graph attachment code must still be tested by 04 with the approved
coordinates, timestamps and its corrected distance convention.

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
  `ok` index rows, plus product, unit and timestamp provenance. The satellite coordinate
  file has not been verified. Do not manufacture it from node count or an assumed grid.
- **Angus/graph owner:** approved fusion timeline, missing-modality treatment, topology
  and successful attachment to the actual fold graphs.
- **Rick/integration:** integrate the adapter and model runner into the experiment
  workflow, and reconcile the D06 Person field and Teams shorthand with Sprint One.
  A full training run is separate acceptance evidence.
- **Scientific review:** the earlier source investigation identified gauge merging in
  the inherited Rainfields `prcp-m15` stream. Holding a gauge out of the graph does not
  remove information already blended into that input. Confirm the supplied file's
  provenance and contributing gauges before claiming an independent comparison.

The production inputs were not available in this checkout, and the NCI SSH attempt
could not verify the host key. The adapter is testable offline; real-data validation
and downstream graph acceptance must be recorded separately when those inputs arrive.

## Verification — 10 September 2026

- Full test suite: 30 passed, 29 existing legacy-data skips. Dependency deprecation
  warnings remain. The new tests include a forward/backward pass through the inherited
  HGNN using both modality feature widths and explicitly synthetic edges.
- New Python files pass Ruff; compilation and whitespace checks pass.
- Repository-wide Ruff still reports 120 findings in unchanged inherited files.
- The example creator, input check and CSV build run successfully. The production
  input check fails explicitly because `database/raw/index.csv` is absent locally.
