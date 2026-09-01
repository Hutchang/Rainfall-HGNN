# 06 — Train the model and write predictions

**Owner:** _unassigned_ · **Reviewer:** Component 03 owner

## What this component is for

Train the graph neural network on Component 04's samples and write one prediction table
that any method can be scored from.

The prediction table is the point of this component. The model is one method among
several — inverse-distance weighting, kriging, a future temporal model — and they can
only be compared if they all write the same rows in the same columns. Everything else
here (run directories, checkpoints) exists so a result can be re-created next month.

The training code already works: `models/gnn.py` and `training/logic_hetero.py` run.
Your job is to wrap them behind one interface, feed them Component 04 samples, and make
the outputs traceable.

## What you receive

| Input | From | Why you need it |
|---|---|---|
| `HeteroData` per fold and role | 04 | the model's input: features, edges, `mask`, `station_ids`, `timestamps` |
| `split.json` | 03 | defines which rows you must predict |
| `gauge_hourly.csv` | 02 | the true rainfall you write beside each prediction |
| `config/seeds.yaml` | 03 | seed Python, NumPy and PyTorch before the model is constructed, or runs are not repeatable |

### Your own config

`config/models/hgnn.yaml` — the inherited settings, which are known to run:

```yaml
method: hgnn
hidden_channels: 32
num_layers: 5
dropout: 0.15
learning_rate: 0.0003
weight_decay: 0.0001
loss: weighted_mse
weighted_loss_alpha: 0.18
batch_size: 128
epochs: 100
early_stop_patience: 10
device: cpu          # or cuda
```

## What you produce

```text
experiments/<method>_<YYYYMMDD-HHMM>/
  config.yaml           copy of the settings actually used
  history.csv           epoch, train_loss, val_loss
  best.pt              checkpoint with the lowest validation loss
  predictions.csv
  report.md
```

**`predictions.csv`** — the cross-team contract:

| Column | Type | Notes |
|---|---|---|
| `method` | string | `hgnn`, `idw`, `network_mean` — whatever produced the row |
| `fold` | int | 0–4 |
| `station_id` | string | must be a test station of that fold |
| `timestamp` | timestamp, UTC | |
| `predicted_mm` | float | clamped at 0; negative rainfall is not a thing |
| `actual_mm` | float | from `gauge_hourly`, carried so scoring needs no join |

Sorted by `method, fold, station_id, timestamp`.

The required rows are exactly:

```python
targets = gauge_hourly[gauge_hourly.station_id.isin(split["folds"][i]["ml"]["test"])]
```

for every fold. Not a row more, not a row fewer. Write a checker that asserts this and
run it before a run is called finished — this is the single check that keeps methods
comparable, and it takes ten lines.

**`report.md`** — RMSE, MAE and correlation overall and per fold, plus the run settings.

Because there is no separate evaluation component, the scoring function lives here, in
`src/models/evaluate.py`, reading `predictions.csv` and nothing else. Any method that
writes a valid table can be scored by it without touching model code.

## How your output is used

`predictions.csv` is the project's result. It is what the report and any comparison
table are built from. Keeping `actual_mm` in the file means a reviewer can recompute
every published number from that one file.

## Also build the two cheap baselines

Both are short, and without them nobody can tell whether the network is doing anything:

- **`network_mean`** — predict the mean rainfall of the training gauges at that hour.
  Must never read the target station's own value.
- **`idw`** — inverse-distance weighting from the training gauges. This is the standard
  method for spatial rainfall interpolation and the number the graph network has to
  beat to be interesting.

Both write the same `predictions.csv` schema through the same interface.

## Rules

1. Never predict a row that is not in the target set, and never skip one.
2. Never read a test station's rainfall as model input. `y` is for the loss only.
3. Seed before constructing the model, and record the seed in `config.yaml`.
4. Do not rebuild graphs or re-derive splits here.
5. A crashed run leaves its directory in place with the error in `report.md`.
6. `experiments/` is gitignored. Do not commit checkpoints.

## Done when

- [ ] `python -m models.run config/models/hgnn.yaml` trains all five folds and writes the
      run directory.
- [ ] The target checker passes: predicted rows equal the split's target rows exactly.
- [ ] Test: two runs with the same seed on CPU give identical predictions.
- [ ] Test: `network_mean` still produces a valid table when the target station's data is
      deleted — proving it never read it.
- [ ] Test: resuming from `best.pt` and predicting gives the same numbers as predicting
      at the end of training.
- [ ] `report.md` shows hgnn, idw and network_mean scored on identical rows.
- [ ] A two-epoch CPU run over the committed test sample finishes in under a minute in CI.
- [ ] `docs/models/README.md` explains the prediction schema and how to add a method.

## Open question

- **The inherited RMSE numbers are in unknown units.** The legacy hand export was
  distorted, so any millimetre claim from a run on that data is not real. The first run
  on Component 02 data is the first result worth quoting. Do not carry old numbers
  forward into the report as a baseline.
