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
coordinate order and unchanged source availability. Missing radar values/hours fail;
there is no automatic time intersection or dry-zero fill. Source product, units and
held-out-gauge independence must be established before interpreting fusion scores.

The graph contains gauge `[N,T,6]`, radar `[R,T,1]` and satellite `[S,T,2]` features.
The first gauge channel remains physical rainfall, the second is binary validity, and
four positional features are constant over time. Sensor edges connect each gauge to
its configured nearest sensor nodes in both directions. The graph report names the
coordinate and edge-weight corrections; Angus should review them before acceptance.

## Fully offline integration example

This creates invented raw-format observations for 12 gauges and three nodes per sensor.
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
