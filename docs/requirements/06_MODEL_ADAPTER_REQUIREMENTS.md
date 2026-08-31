# 06 — Train models and write comparable predictions

**Owner:** _unassigned_ · **Reviewer:** future evaluation owner, with component 03 owner
**Status:** interface and run-directory work can start against documented test samples;
full acceptance waits for Component 04 samples

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive
**First method:** inherited gauge-only `GNNInductiveHetero` (HGNN)

## 1. Summary

This component gives every learned model the same way to train and write predictions.
An **adapter** is a thin wrapper around existing model code: it translates the project's
standard graph inputs into that model's functions and translates its outputs into the
project's standard prediction table. The first adapter wraps the inherited heterogeneous
graph neural network (HGNN) without rewriting its working mathematics.

It turns graph samples into understandable run directories, restart files called
checkpoints, and one prediction table containing exactly the station-hours defined by
Component 03.

The prediction table is the important cross-team boundary: future HGNN variants,
temporal models, inverse-distance weighting (IDW), Kriging, and other methods must write
the same rows and required columns before a common evaluator compares them.

Responsibility begins with accepted Component 04 graph samples and ends with checked
predictions and complete records of how the run was performed. This component does not
invent splits, rebuild graphs, collect data, or calculate official experiment metrics.
Per-epoch loss is training progress information; final metrics belong to the future
evaluation component.

## 2. System Position

```text
04 graph samples + mappings       03 exact target list
               │                         │
               └────────────┬────────────┘
                            ▼
                    MODEL ADAPTER
                            │
       restart files + training history + predictions.parquet
                            ▼
            future common evaluation / comparison
```

Parquet is the typed table format explained in Component 02. It preserves required
column types such as UTC timestamps and numbers, unlike a plain CSV file.

### Worked example — one HGNN run

1. Load the model settings and accepted `sample_version`.
2. Seed Python, NumPy, PyTorch, and data loaders before model creation.
3. Train fold 0 with the inherited loss/optimiser settings, writing restartable
   state after every epoch and the best validation checkpoint.
4. Load the best checkpoint and predict fold 0's exact target rows.
5. Join predictions through the graph's internal station-id and UTC timestamp mapping.
6. Repeat for all folds, then write one sorted `predictions.parquet` table.
7. Validate its required columns, values, input dataset/split/sample ids, and exact
   equality with Component 03's target list.
8. Mark the run complete only after all files and their fingerprints (hashes) are present.

## 3. Inputs

### 3.1 Settings owned by Component 06

#### IN01 — Model and training settings

The Component 06 owner creates and maintains
`config/models/hgnn_inherited_v1.yaml`. It supplies method id, architecture, optimiser,
loss, batch size, epochs, early stopping, checkpoint interval, device policy, and
repeatability policy. Every field affecting numerical results is copied into the run
directory.

The initial inherited HGNN settings are:

```yaml
method_id: hgnn_inherited_v1
hidden_channels: 32
num_layers: 5
dropout: 0.15
loss: weighted_mse
weighted_loss_alpha: 0.18
optimizer: adam
learning_rate: 0.0003
weight_decay: 0.0001
gradient_clip_norm: 1.0
early_stop_patience: 10
eval_clamp_min: 0.0
```

Epoch count and batch size are explicit run fields, not hidden defaults.

#### IN02 — Run request

A run request supplies `sample_version`, fold set (normally all), output root, requested
device (`cpu` or `cuda`), CPU-fallback policy, and optional resume run id.

### 3.2 Files from Components 02–04

#### IN10 — Accepted graph samples

Component 04 OUT01–OUT04, including `manifest.json` (the machine-readable sample-set
record), graph settings/modes, fitted
statistics, station/time mappings, and target-list hash.

#### IN11 — Accepted split and prediction target list

Component 03 OUT01/OUT02/OUT04. Predictions must match these rows exactly.

#### IN12 — Standard rainfall values

The adapter obtains dataset version and true rainfall through sample mappings and the
Component 02 Python loader. It does not parse raw source data or legacy CSV files.

#### IN13 — Project seeds

Component 03 OUT05, loaded and applied before any model, sampler, or data loader exists.

## 4. Functionality Requirements

### FR01 — Provide one common Python interface

"Typed" below means the functions and returned data classes use Python type hints so
callers and development tools can detect the wrong input kind early.

Provide under `src/models/adapters/`:

```python
train_fold(samples, model_config, run_context) -> CheckpointRef
predict_fold(checkpoint, samples, run_context) -> PredictionTable
run_experiment(run_request) -> RunResult
```

- `train_fold` accepts only train/validation samples from one accepted release.
- `predict_fold` runs standalone from a checkpoint without retraining.
- `run_experiment` checks the input dataset/split/sample records, runs the requested
  folds, combines predictions, checks the final table, and writes run metadata.
- Public data classes and functions have type hints and plain-language docstrings.

### FR02 — Wrap the inherited HGNN before changing it

**Required**

- Use `models/gnn.py:GNNInductiveHetero` and tested logic from
  `training/logic_hetero.py` as the first implementation.
- Preserve the IN01 inherited settings: leave-one-target masking supplied by Component 04,
  weighted mean squared error `1 + 0.18*log1p(target)`, Adam, gradient clipping,
  validation early stopping, best-checkpoint reload, and evaluation-only non-negative
  clamp.
- Remove plotting and hidden file writes from core train/predict functions; the run
  layer writes the output files.
- Any numerical change becomes a new `method_id` or explicit settings change, never an unlabelled
  cleanup.

### FR03 — Seed before construction and record determinism

Where this section records a hash, it means a SHA-256 content fingerprint. It lets a
resumed run detect that a settings or seed file has changed since training began.

**Required**

- Call component 03 `seed_everything` before model construction and data-loader setup.
- Use `dataloader_seed` in any loader generator/worker initialisation.
- Record seed-file hash, resolved seed values, deterministic-algorithm settings, and
  backend/library versions.
- If a requested backend cannot satisfy deterministic mode, fail by default. The settings
  may explicitly allow measured nondeterminism and must label the run.

### FR04 — Save enough state to restart training

A **checkpoint** is a `.pt` file containing the model and optimiser state needed to
continue training or reproduce predictions without starting again.

Write per fold:

```text
checkpoints/fold-<n>/last.pt
checkpoints/fold-<n>/best.pt
```

`last.pt` contains model state, optimiser state, epoch, best validation loss, patience
counter, Python/NumPy/PyTorch random states, settings/input hashes, and library
versions. Write through a temporary file and atomic rename after every epoch.

`best.pt` is the best validation state and is the only checkpoint used for final
predictions unless a run request explicitly names another. Resume validates every
identity/hash before continuing; mismatch fails rather than partially loading.

### FR05 — Save understandable training history

Write `history.parquet` with run id, fold, epoch, train loss, validation loss, learning
rate, elapsed seconds, best-so-far flag, and stop reason. Stop and record fold/epoch/
batch when a loss or gradient is infinite or not a number.

### FR06 — Define the prediction table and required columns

Write one `predictions.parquet`, sorted by `(fold, station_id, timestamp)`, with exactly:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | unique run attempt identifier |
| `experiment_id` | string | fingerprint of the method, settings, and input versions |
| `method_id` | string | named method/settings family |
| `dataset_version` | string | component 02 version |
| `split_id` | string | component 03 version |
| `sample_version` | string | component 04 version |
| `fold` | int16 | fold index |
| `station_id` | string | BoM internal station id |
| `timestamp` | timestamp[UTC] | hour-beginning target time |
| `y_true_mm` | float64 | accepted Component 02 rainfall in millimetres |
| `y_pred_mm` | float64 | method prediction in millimetres |

- One row represents one exact held-out target.
- Aggregate-only output is prohibited.
- Predictions are finite and non-negative after the declared evaluation clamp.
- `y_true_mm` must join exactly from the named Component 02 dataset, not a model tensor that
  could have been transformed.

### FR07 — Separate experiment identity from run attempts

- `experiment_id` is calculated from the method settings, dataset/split/sample ids,
  seed-file hash, and code revision.
- `run_id` is `<experiment_id>-<UTC-start>-<short-random-suffix>` and identifies one
  attempt/resume chain.
- Repeated attempts with identical inputs have the same experiment id and different run
  ids. Their prediction rows must match after excluding `run_id` and runtime metadata in
  deterministic mode.

The experiment id answers "are these scientifically the same settings and inputs?"
The run id answers "which concrete attempt produced these files?" Keeping both prevents
a retry from being mistaken for a different experiment.

### FR08 — Check predictions against the exact target list

Provide shared:

```python
validate_prediction_file(predictions_path, split_path, targets_path, dataset_loader)
```

It checks required columns/types, constant input version ids, unique row identifiers,
exact equality with Component 03's target rows, true-rainfall equality, finite and
non-negative predictions, complete folds, sorted order, and the recorded file hash. The
adapter marks a run complete only when all checks pass. Future comparison methods reuse
this function unchanged.

### FR09 — Use a self-describing run directory

Every attempt writes:

```text
experiments/<run_id>/
  run.json
  RUN.md
  model-settings-used.yaml
  inputs/
    split.json
    sample-record.json
    dataset-record.json
    seeds.yaml
  checkpoints/fold-*/
  history.parquet
  predictions.parquet
  validation.json
  failure.json                 # only when failed
```

`run.json` is the machine-readable record for one attempt. It stores status (`running`,
`failed`, `complete`), ids/hashes, code revision,
command, host/device, libraries, start/end UTC, runtime, seed/determinism settings,
checkpoint/prediction hashes, and failure reference.

### FR10 — Handle device policy explicitly

- `cuda` requested and unavailable fails before training by default.
- `allow_cpu_fallback: true` permits CPU fallback and records requested/actual devices.
- `cpu` never initialises CUDA.
- Moving checkpoints across devices uses an explicit map location and validates model
  compatibility.
- Device choice cannot change sample construction or target rows.

### FR11 — Resume interrupted runs safely

- `--resume <run_id>` continues incomplete folds from `last.pt` and skips validated
  complete folds.
- It never overwrites a complete run or restarts silently after an incompatible/missing
  checkpoint.
- A deliberate `--restart-failed-fold <n>` creates a linked new attempt or records the
  destructive replacement policy explicitly; default is non-destructive.
- Resumed and uninterrupted deterministic runs produce equivalent predictions.

### FR12 — Leave failed runs inspectable

On invalid input files, non-finite training, checkpoint error, device failure,
or prediction-validation failure:

- set run status to `failed`;
- write `failure.json` with stage, fold/epoch/batch where available, exception type,
  message, and last valid output file;
- retain prior valid checkpoints/history;
- return non-zero.

### FR13 — Implement `network_mean_v1` through the same output contract

Provide a non-neural reference method that, for each target timestamp, predicts the mean
of available non-test gauge observations visible in that fold's test graph. It must not
use any test-station value. If no predictor gauge is available, fail that target rather
than silently write a global constant.

It writes the same run layout and prediction-table columns (checkpoints are marked not
applicable). Purpose: prove the interface works for a second method before the formal
baseline component exists, not establish the final statistical baseline.

### FR14 — Keep metric calculation outside the adapter

The adapter may report training/validation loss and validator counts. It does not write
official RMSE, Pearson correlation, IDW deltas, or comparison claims. The future
evaluation component calculates all final metrics from checked row-level prediction
files.

## 5. Non-negotiable Rules

1. The accepted sample set and Component 03 target list define all rows; the adapter does
   not invent or filter targets.
2. Every prediction uses accepted Component 02 millimetre truth and UTC timestamps.
3. Station identity is the BoM internal id, never a positional index.
4. A run is complete only after exact-target prediction validation.
5. Core train/predict functions do not hide plotting or untracked filesystem side effects.
6. Numerical behaviour changes receive explicit method/settings identities.
7. All random state comes from `config/seeds.yaml` and restart checkpoints preserve it.
8. Device fallback, nondeterminism, missing targets, and failures are never silent.
9. The future evaluation component computes official metrics from prediction tables.

## 6. Outputs

### OUT01 — HGNN adapter with Python type hints

`src/models/adapters/hgnn.py` implementing FR01–05 around the inherited HGNN.

### OUT02 — Prediction-table checker

A model-independent shared module implementing FR08, with a command-line entry point
for future model and comparison-method owners.

### OUT03 — Self-describing run directories

`experiments/<run_id>/` using FR09. Do not edit it after marking the run complete.

### OUT04 — Row-level predictions

Checked `predictions.parquet` using FR06, read by the future common evaluator and
component 05 acceptance tests.

### OUT05 — Restartable checkpoints and history

Per-fold best/last checkpoints and `history.parquet` using FR04/05.

### OUT06 — `network_mean_v1` simple comparison method

A non-neural adapter with the same run layout and prediction columns, used to check the
common interface before formal statistical methods are implemented.

### OUT07 — Run/validation commands

Documented commands to start, resume, predict-only, validate, and inspect a run. Each
returns a meaningful non-zero status on failure.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Exact-target end-to-end run

On the golden fixture, train/predict all requested folds. Prediction keys equal
component 03 targets exactly, truth joins exactly, and validation passes.

#### TEST02 — Run directory explains itself

From one completed run directory alone, a reviewer can reconstruct all input versions,
settings, seeds, code/library versions, device, checkpoints, prediction hash, and command.

#### TEST03 — Deterministic repeat

Two attempts with identical experiment identity produce equivalent checkpoints and
prediction content after excluding attempt-specific `run_id`/runtime metadata. Changing
`model_init_seed` changes model state while preserving valid target rows.

#### TEST04 — Failure visibility

Plant an invalid input hash, unavailable requested CUDA, NaN loss, missing target mapping,
and bad prediction. Each leaves a correctly staged failed run and non-zero exit.

### 7.2 Training/checkpoint tests

#### TEST10 — Inherited regression anchor

The inherited method/settings combination on Component 05's compatibility samples falls within frozen
execution bands and matches expected training/checkpoint semantics.

#### TEST11 — Resume equivalence

Interrupt mid-epoch boundary, resume from `last.pt`, and compare with uninterrupted
deterministic predictions. Epoch/history records have no duplicate or missing epoch.

#### TEST12 — Best checkpoint use

Construct a case where the final epoch is worse than best validation. Predictions load
`best.pt`, and run metadata records its epoch/hash.

#### TEST13 — Device handling

CPU, unavailable-CUDA failure, explicit CPU fallback, and cross-device checkpoint load
all behave and record metadata as FR10 requires.

### 7.3 Prediction/second-method tests

#### TEST20 — Validator mutation suite

Reject an extra row, missing row, duplicate key, wrong fold/station/time, transformed
truth, NaN/negative prediction, wrong input version id, unsorted output, and bad hash with a
specific diagnostic.

#### TEST21 — Predict-only execution

Run `predict_fold` from accepted best checkpoints in a fresh process without training;
combined predictions validate and match the original deterministic attempt.

#### TEST22 — Network-mean leakage guard

Changing a test-station target does not change `network_mean_v1`; changing an available
non-test gauge at the same timestamp changes the expected prediction.

#### TEST23 — No metric ownership

Adapter outputs contain training-progress information but no official comparison metrics.

## 8. Documentation Requirements

All documents are Markdown under `docs/models/`.

### DOCS01 — Adapter usage guide

Copy-pasteable examples for start, resume, predict-only, validate, CPU/CUDA policy, and
expected files/status transitions.

### DOCS02 — Prediction-table guide

Explain every FR06 column, sorting/uniqueness, exact-target rule, true-value lookup, hashes, and
provide a five-row filled example for future baseline/evaluation owners.

### DOCS03 — Checkpoint and determinism statement

Explain checkpoint contents, random state, supported deterministic environments,
known backend nondeterminism, and measured tolerance where allowed.

### DOCS04 — Model configuration guide

Explain inherited HGNN settings, which choices belong to component 04 versus 06, and
how a numerical change receives a new method/settings identity.

### DOCS05 — Run directory and failure runbook

Explain every file/status, how to diagnose/retry failure, and how to hand a validated
prediction table to evaluation.

### DOCS06 — Method changelog

One entry per method/settings identity with reason, numerical changes, input sample
modes, and previous version.

## 9. Dependencies / Blockers

- Requires accepted component 03/04 interfaces and target mappings.
- Scientific runs remain blocked on regenerated standard BoM data; legacy distorted
  data is regression-only.
- Formal IDW/Kriging/evaluation requirements are not yet specified. `network_mean_v1`
  proves the interface but does not replace them.
- Production Gadi/PBS orchestration is a later work package; restartable runs and
  explicit device handling are required now so it can wrap this adapter later.

## 10. Current Technical Findings

- `train_fused.py` now has a main guard and can run gauge-only, but it remains a
  monolithic transitional driver with timestamp-named output and incomplete metadata.
- `training/logic_hetero.py` provides working training/evaluation internals; its current
  prediction station ids are positional and must be replaced at the adapter boundary.
- Existing model initialisation/dropout is unseeded.
- Current scripts discard or scatter some return values, compute metrics inside test
  logic, and do not save the required exact row-level prediction table.
- The validated 2026-08-31 transitional runs prove end-to-end runnability only; their distorted
  source values and reduced training length make their RMSE unsuitable as science.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines HGNN, checkpoint, prediction table, held-out,
  RMSE, Pearson correlation, IDW, and deterministic seeds.
- Required earlier component documents:
  [03_STUDY_SPLIT_REQUIREMENTS.md](03_STUDY_SPLIT_REQUIREMENTS.md) and
  [04_GRAPH_SAMPLE_REQUIREMENTS.md](04_GRAPH_SAMPLE_REQUIREMENTS.md).
- Working inherited core: `models/gnn.py`, `training/logic_hetero.py`, and transitional
  `train_fused.py`.
- Evidence: `tests/test_split_integrity.py` and the retained inherited core above.
