# Model training and prediction

Component 06 consumes the standard tables from Component 02, the released split and
seed file from Component 03, and the graph samples from Component 04. It produces a
single prediction table for the HGNN, IDW and network-mean methods. It does not fetch
raw data, produce canonical data, choose station splits or rebuild graphs.

The default path is the Sydney gauge-only baseline. The adapter also consumes the
documented radar and satellite layers when the supplied Component 04 graphs contain
them. No result from the
legacy hand export is used. The small synthetic test fixture checks software behaviour;
its metrics are not rainfall research results.

## Component 06 scope and integration

Component 06 covers training, predictions, baselines and radar/satellite model support.
It is integrated with 02, 03 and 04; their published outputs are connected through the
[pipeline guide](../graphs/README.md).

Open decisions and dependencies affecting this component are tracked in
[ISSUES.md](../ISSUES.md), not here: **ISS-07** (04 and 06 let a method read different
gauges when predicting),
**ISS-08** (IDW exponent unprescribed), **ISS-09** (no normalization metadata in the graph
contract), and **ISS-02**, **ISS-03**, **ISS-04** (radar and satellite provenance,
coordinates and coverage).
Do not claim accepted fusion results while those are open.

## Install and run

Run commands from the repository root. Use the locked environment, which currently
requires Python 3.14 or later, following the repository's dependency instructions in
[AGENTS.md](../../AGENTS.md).

```console
uv sync --locked
uv run --locked python -m src.models.run config/models/hgnn.yaml --check-inputs
uv run --locked python -m src.models.run config/models/hgnn.yaml
```

The component boundary assigns implementation to `src/models/`, and the repository
does not install `src` as an import root. The executable command is
`python -m src.models.run`; the requirements entrypoint was corrected in this change.
`src/models/models.py` holds the inherited architectures, moved under this component's
boundary from the former top-level `models/gnn.py`. The runner uses `GNNInductiveHetero`;
the other six classes are retained unused for possible later work.

The YAML supplies paths to the published inputs:

| Key | Input |
|---|---|
| `gauge_file` | `database/standard/gauge_hourly.csv` |
| `stations_file` | `database/standard/stations.csv` |
| `split_file` | `config/splits/sydney_5fold.json` |
| `seeds_file` | `config/seeds.yaml`, containing `model_seed` |
| `graphs_dir` | `database/graphs/gauge_baseline/fold-<i>/{train,validation,test}.pt` |

Paths resolve from the working directory, normally the repository root. Graphs are
trusted team-produced PyG `HeteroData` objects saved by Component 04; use only the
documented types and do not load arbitrary downloaded pickle files. Each graph must
retain `station_ids`, UTC `timestamps`, node features/labels, masks and weighted edges.
Its observed rainfall and validity must agree with the canonical table. Missing
observations cannot be silently represented as observed zeros.

`--check-inputs` reads and validates all required inputs without training or writing a
run. A missing upstream file produces a concrete `BLOCKED` error. It does not create
placeholder production files. Configuration errors and CUDA availability are also
checked before training.

For the first run on real upstream outputs, use a bounded smoke check:

```console
uv run --locked python -m src.models.run config/models/hgnn.yaml --epochs 2 --device cpu
```

This still uses all five supplied folds and all supplied hours. Two epochs over a full
production dataset are not guaranteed to finish quickly. Use the committed synthetic
fixture test for the sub-minute CPU acceptance check; do not edit a released split to
make a private scientific subset. The [Gadi guide](../nci/README.md) provides a prepared
environment and a bounded A100 PBS job.

## What a run writes

Every run gets its own directory under `experiments/`, or an explicit `--output-dir`.
Existing directories are never overwritten.

- `config.yaml`: effective settings, seed, package/CUDA versions, code revision,
  source hashes and input hashes. Uncommitted local code is identified explicitly.
- `history.csv`: fold, epoch, training loss, validation loss, best validation loss,
  accumulated deteriorations, remaining patience and early-stopping status. The same
  progress is printed once per epoch for live runners.
- `best.pt`: each fold's lowest-validation-loss weights and model construction
  settings. Intermediate checkpoints preserve the completed folds.
- `predictions.csv`: all requested methods, on exactly the same observed test rows.
- `report.md`: overall/per-fold RMSE, MAE and Pearson correlation; explicit assumptions
  and missing-observation counts. Undefined correlations are reported as undefined.

A caught failure leaves the run directory and error report in place. A scheduler kill
may interrupt before the failure report can be written; check PBS exit status and logs
as well as files. A partial checkpoint is not a completed five-fold run.

To reproduce predictions from completed best weights without further training:

```console
uv run --locked python -m src.models.run config/models/hgnn.yaml \
  --predict-from experiments/<completed-run>/best.pt
```

This is checkpoint-based inference, not resumption of the optimizer mid-epoch. Input
content and model settings must match the saved run. GPU-to-CPU or different hardware
can introduce numerical differences; the exact repeatability check is on CPU.

## Prediction contract

| Column | Meaning |
|---|---|
| `method` | `hgnn`, `idw`, or `network_mean` |
| `fold` | integer 0–4 |
| `station_id` | BoM internal station ID, preserved as a string |
| `timestamp` | observed hour start, timezone-aware UTC |
| `predicted_mm` | finite nonnegative rainfall prediction |
| `actual_mm` | unchanged rainfall from Component 02 |

Rows are sorted by method, fold, station ID and timestamp. For each method, the target
set is every `gauge_hourly.csv` row belonging to that fold's `ml.test` stations. The
checker rejects missing/extra rows, duplicate identities, wrong canonical labels,
naive timestamps, non-finite values and negative predictions. Dry zeros remain valid
targets; absent observations do not become targets.

Scores can be recomputed without importing the training model:

```console
uv run --locked python -m src.models.evaluate experiments/<run>/predictions.csv
```

The scoring function checks that the included methods share identical targets and
truth values. Full validation against the released target set happens in the runner;
the scorer alone cannot reconstruct a missing split file.

## Radar and satellite model inputs

`input_layers` selects the required graph schema. It defaults to `[raingauge]` for
existing gauge-only configs. Explicit options are `[raingauge, radar]`,
`[raingauge, satellite]`, or `[raingauge, radar, satellite]`. Set `graphs_dir` to the
corresponding graphs supplied by 04. Every fold and role must have exactly those layers
and the same relation types/feature widths. This runner does not remove nodes, generate
missing layers, rebuild edges or change the released target rows for an ablation.

| Node type | Features for each hour |
|---|---|
| raingauge | rainfall in mm, observed 0/1, four fixed positional features |
| radar | one finite nonnegative rainfall value, `[N,T,1]` |
| satellite | rainfall and original availability 0/1, `[N,T,2]` |

The adapter passes all selected layers through the inherited HGNN, including batching
and checkpoint restoration. Sensor tensors must share the graph's exact time axis and
the gauge tensor's floating dtype. Source hours and coordinate order must already have
been aligned by 04; equal dimensions cannot prove correct physical alignment.

Unavailable satellite payloads are zeroed only on a cloned computational tensor, with
availability kept at 0. Source graphs and labels remain unchanged. Radar has no validity
feature in the current contract, so missing radar values fail explicitly. No automatic
intersection with the radar's shorter 2022 coverage or replacement of missing values
with observed dry zeros occurs. Each sensor type must have a nonempty relation directed
to gauges. Incoming sensor relations are optional: the stacked HGNN retains unchanged
features for source-only sensor types across layers, while types with incoming relations
(including supplied within-grid edges) keep their normal convolution updates. The model
never invents a reverse edge. PyG warns that source-only types are not updated; retaining
their features is intentional and covered by multi-layer gradient and replay tests.

The graph builder's `reverse_cross_support` setting and neighbour count are documented
in the [pipeline guide](../graphs/README.md). Existing bidirectional graphs remain valid;
one-way graphs require a new run and checkpoint because their relation schema differs.
The remaining D09.2 grid-topology work is [ISS-38](../ISSUES.md#iss-38).

Rescaling is one setting, `input_rainfall_scale`. `physical_mm` feeds the model raw
millimetres. `zscore_train` (D08.2; paper §4.3) subtracts the mean and divides by the
spread of the observed rainfall in each fold's **training** graph, per node type, and
applies those two numbers to the validation and test graphs as well. Only the rainfall
input channel is scaled: validity, positional features, targets and predictions are not,
so scores stay in millimetres. The numbers used are written to the run's `config.yaml`
under `input_scaling`, and a checkpoint cannot be replayed under the other setting. Graph
files stay in physical millimetres either way; do not change labels to fit a model, and
agree any transform of the files themselves with Component 04 first. Component 01/02 must establish
product, units, timestamp meaning and gauge correction before physical interpretation. All
fusion run reports flag that tensor compatibility alone does not establish independent
held-out gauge evaluation. The supervisor's broader national-run objective does not make
the first Sydney fixture an Australia-wide result.

To exercise the model adapter without production data:

```console
uv run --locked python tests/data/models/make_fixture.py experiments/models-fusion-fixture --modalities radar satellite
uv run --locked python -m src.models.run experiments/models-fusion-fixture/config.yaml --check-inputs
uv run --locked python -m src.models.run experiments/models-fusion-fixture/config.yaml
uv run --locked python -m pytest tests/test_model_modalities.py -q
```

These sensor values and edges are explicitly invented software fixtures. The [grid adapter](../grids/README.md) supplies trusted raw grid conversion; the
[integrated 04 builder](../graphs/README.md) consumes its tables and writes the graphs
used here. Their end-to-end path is tested with synthetic inputs.

## Baselines and model behaviour

`network_mean` averages the observed **visible** gauges at the target hour: the fold's
training and validation gauges, the same ones the HGNN's test graph holds (D08.2). `idw`
reads those same gauges, WGS84 haversine distances, and the explicitly configured
`idw_power` (the example is 2.0). A gauge at the same location as the target, if it is
observed, takes precedence over the rest; several such gauges are averaged. No hour is
gap-filled. If no visible gauge is observed at that hour, the run fails with the
fold/hour instead of dropping targets or inventing rainfall.

The model adapter keeps the inherited `GNNInductiveHetero` and leave-one-gauge-out
training objective. It masks rainfall and validity while preserving positional
features. Missing labels are excluded from loss and prediction, and labels are kept
separate from masked features. The selected checkpoint is evaluated after training.

To add a method, return the exact prediction columns and invoke
`validate_predictions(table, gauge, split, method=...)` before writing output. Use the
shared split and seed; do not create private target lists. The standalone scorer can
evaluate any method name that meets the same contract.

## Verification

The integration branch passed **133 tests with 29 skips** on 18 September 2026. The
**8 NCI guards** pass on Linux/macOS and fail on a Windows checkout for a line-ending
reason unrelated to this component; see [ISSUES.md](../ISSUES.md). The reconciled
integration modules pass focused Ruff and compilation checks; repository-wide Ruff
reports 692 findings in inherited, transitional and other pre-existing files. Dependency deprecation warnings remain.
Each sensor combination passed deterministic CPU prediction and checkpoint replay
tests. The [pipeline guide](../graphs/README.md) also exercises the real 02/03/04
implementations with invented raw inputs. None of these tests establishes a completed
real-data or GPU run.

```console
uv run --locked python -m pytest tests -q -rx
uv run --locked ruff check src/models tests/test_model_contracts.py tests/test_model_training.py tests/test_model_modalities.py tests/data/models
uv run --locked python -m pytest scripts/nci/test_guards.py -q
git diff --check
```

The tests cover exact target rows and labels, deliberate target-data deletion, a target
hour with no gauge observed, collocated IDW, metric recomputation, CPU determinism,
checkpoint inference,
and a two-epoch synthetic run. Multimodal tests cover each sensor combination through
five folds, deterministic predictions and checkpoint replay; they also prove that
sensor values affect the output and receive gradients while hidden gauge readings do
not affect it. Invalid widths, availability, missing values/hours and disconnected
sensor layers fail before training. The historical regression tests may skip without the
untracked legacy dataset. Existing lint failures in inherited reference files are
reported separately from the new component's checks.
