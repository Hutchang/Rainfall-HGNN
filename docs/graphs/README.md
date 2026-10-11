# Integrated tables, splits, graphs and training

The repository-wide pipeline guide: how Components 02, 03, 04 and 06 connect through
their published files. Run from the repository root with `uv sync --locked`. The
[current state](../CURRENT_STATE.md) records integration state and repository
boundaries; [ISSUES.md](../ISSUES.md) records open defects and scientific review
points.

## From published source inputs

```console
uv run --locked python -m src.canonical.gauge.build config/canonical/standard.yaml
uv run --locked python -m src.study.split config/splits/sydney_5fold.yaml
uv run --locked python -m src.graph.builder_new config/graphs/gauge_baseline.yaml
uv run --locked python -m src.models.run config/models/hgnn.yaml --check-inputs
uv run --locked python -m src.models.run config/models/hgnn.yaml
```

Check the canonical report before releasing a split. The study CLI preserves an existing
byte-identical output and refuses to replace it with different content: a released
split must never be overwritten. Skip the first two commands when
using existing, reviewed canonical tables and a released split. The graph builder also
requires a fresh output directory and records a failure report after a partial build.

`config/graphs/` is the repository boundary owned by Component 04. Its corrected builder writes
`database/graphs/gauge_baseline/fold-<i>/{train,validation,test}.pt`, matching the model
config. Graph config can specify `data_dir`, `split_file` and `output_dir`; CLI flags
`--data-dir`, `--split-file`, `--output-dir` and `--seeds` provide explicit overrides.
All paths resolve from the repository root.

## Add radar or satellite

First run the [grid adapter](../grids/README.md) against its indexed source files. It
writes standard grid tables beside the gauge tables without changing them. In a copy
of the graph config set `input_layers` to `[raingauge, radar]`,
`[raingauge, satellite]` or `[raingauge, radar, satellite]`; specify
`layer_k_neighbours` explicitly and use a fresh `output_dir`. Set the model config's
`input_layers` and `graphs_dir` to match. Gauge-only is the default.

For every sensor, 04 reads `<layer>_hourly.csv` and `<layer>_nodes.csv`. It requires the
exact gauge timeline, complete coordinate-node rows for each present hour, valid
coordinate order and unchanged source availability.

A gauge hour that a timeline layer has no value for is dropped from every graph the
build writes ([DEC-01](../DECISIONS.md#dec-01)): for radar, an hour missing from
`radar_hourly.csv` or with any cell that has no value; for satellite, an hour missing
from `satellite_hourly.csv` (an hour with any image is always written, DEC-02). Grid
layers in `input_layers` are always timeline layers. `timeline_layers: [radar,
satellite]` applies the same cut to a build that does not read those layers, so a
gauge-only run can be scored on exactly the hours of the G+R and G+R+S runs beside it.
`dropped_hours.csv` in the output folder lists every dropped hour with its layer and
cause (just a header when nothing is dropped), the report gives the counts, and
Component 06 removes those hours from every method's targets. Missing values are never
filled with dry zeros. Source product, units and
held-out-gauge independence must be established before interpreting fusion scores.

The graph contains gauge `[N,T,6]`, radar `[R,T,1]` and satellite `[S,T,2]` features.
The first gauge channel remains physical rainfall, the second is binary validity, and
four positional features are constant over time. Sensor edges connect each gauge to
its configured nearest sensor nodes, directed **from the sensor into the gauge**.
`layer_k_neighbours: 9` is the production default. `reverse_cross_support: false`
is the default even when the key is absent; set it to `true` to explicitly include
gauge-to-sensor edges for a comparison with the earlier bidirectional topology.
The setting must be a YAML boolean. Each stored relation keeps its own receiving-node
normalization when `edge_normalisation: per_node` is selected. The graph report records
the effective reverse-edge setting and the counts and weight ranges of every relation.

Each radar or satellite cell also links to its immediate neighbours on its own grid
(paper §3.2): up to eight, four along the grid and four diagonal, so a cell on the edge
of the grid gets fewer rather than links to cells two steps away. A neighbour is any of
a cell's eight nearest cells within 1.9 times its closest spacing; on the D13 sample
grids that picks exactly the surrounding cells (radar's diagonals sit at 1.42, the
satellite grid's at up to 1.76, and the next ring out at 2). The links run both ways
and are weighted by inverse distance, normalized at the receiving cell like every other
relation. `grid_edges: true` is the default even when the key is absent; `false` leaves
sensor cells unlinked, as before D09.2. The report records the setting and lists the
`(<layer>, 'connects', <layer>)` relations.

The direct `GaugeGraphNew.add_heterodata` API also defaults to nine neighbours and grid
links, and accepts the keywords `reverse_cross_support=True` and `grid_edges=False`. Existing callers with smaller fixtures must
pass their intended `knn`; an oversized neighbour request fails instead of silently
clamping it. To reproduce an earlier fusion graph, explicitly set both the previous
neighbour count, `reverse_cross_support: true` and `grid_edges: false`.

## Fully offline integration example

This creates invented raw-format observations for 12 gauges and twelve nodes per sensor,
exercising the production nine-neighbour setting.
It then runs the actual implementations of 02, 03, 04 and 06, not a mock graph builder.
Choose a new fixture directory if this example already exists.

```console
uv run --locked python tests/data/pipeline/make_fixture.py experiments/pipeline-demo
uv run --locked python -m src.canonical.gauge.build experiments/pipeline-demo/canonical.yaml
uv run --locked python -m src.canonical.grids experiments/pipeline-demo/grids.yaml
uv run --locked python -m src.study.split experiments/pipeline-demo/split.yaml
uv run --locked python -m src.graph.builder_new experiments/pipeline-demo/graph.yaml
uv run --locked python -m src.models.run experiments/pipeline-demo/model.yaml --check-inputs
uv run --locked python -m src.models.run experiments/pipeline-demo/model.yaml --output-dir experiments/pipeline-demo/run
uv run --locked python -m src.models.run experiments/pipeline-demo/model.yaml --predict-from experiments/pipeline-demo/run/best.pt --output-dir experiments/pipeline-demo/replay
```

Expected: five folds, 288 observed target rows per method, 864 total prediction rows.
The replay is exactly equal on the tested CPU environment. These numbers establish
software integration only; all fixture observations and coordinates are invented.

Run `uv run --locked python -m pytest tests/test_pipeline_integration.py -q` for the
same chain plus graph determinism, held-out input isolation, coordinate ordering,
coincident-node weights, gauge-only selection and missing-radar failure checks.
