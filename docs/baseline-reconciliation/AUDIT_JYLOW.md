# AUDIT — Jylow implementation (Singapore, 0D + 1D + 2D fusion)

**Audited tree:** `integration/sydney-gauge-baseline` @ `1c2489a`, branched from `jylow-clone`.
Upstream: `github.com/jylow/rainfall_data_fusion` (remote `temp`), imported as a single
squashed merge commit. `README.md` is a one-line stub.
**Audit date:** 2026-08-29 · **Method:** read-only static analysis + one executed
gauge-only probe on synthetic data.

> **Provenance note.** The session's `gitStatus` snapshot was stale (it showed
> `niramay-clone` @ `8368cbf`), and two audit passes inherited that label. Provenance was
> re-verified directly: every file cited below exists in this tree and none exist on the
> Niramay worktree. The **content** is Jylow; only those headers were wrong.

---

## 1. Executive summary

Jylow is the paper's full framework: three support-typed node layers (`raingauge` 0D,
`cml` 1D, `radar` 2D) fused by cross-support message passing into a 0D prediction layer.
It is substantially larger and more capable than Niramay's tree — and substantially less
runnable.

**Three facts dominate everything downstream:**

1. **Genuine 0D-only execution is achievable** (FR-201 answer: **(c), shallow**). The
   library layer is already support-agnostic; only the top-level drivers block it. This
   was verified by execution, not inference — see §2.
2. **The repo cannot run at all in its current state.** No data, an uninstallable
   `requirements.txt`, a `.venv` missing four required packages, and one mandatory
   pipeline step that exists only inside a notebook. See §6.
3. **Feeding Sydney gauge data in will fail on contact** — a hardcoded `+08:00` timestamp
   format and a `×12` unit conversion that assumes 5-minute accumulations. See §7.

---

## 2. FR-201 — Can Jylow run genuine 0D gauge-only? **YES, with driver-level changes**

**Answer: (c), but a shallow (c) — very close to (b).** Option (d) is definitively false:
no model mathematics breaks.

### Evidence by execution
A synthetic gauge-only dataset was pushed through the real
`GaugeGraphNew` → `HeterogeneousWeatherGraphDatasetInductive` → `GNNInductiveHetero` →
`train_epoch`/`validate`/test path, **never calling `add_heterodata()`**, with no CML and
no radar present:

```
train node_types=['raingauge']  edge_types=[('raingauge','connects','raingauge')]
      x=(16,40,6) y=(16,40,1) E=(2,108) attr=(108,)
train_loss=5.2167  val_loss=4.3652   out=(192,1) masked_nodes=32 rmse=2.2154
```

This satisfies the substance of **FR-204** (build a 0D graph, forward pass, finite loss,
backprop, finite predictions) — on synthetic data. It must be repeated on real Sydney
gauge data before the milestone can be signed off.

### How support selection works
Driven by `config.yaml:62` `datasources: ['raingauge', 'cml']` — a plain list-membership
test, **not** file presence. Guards at `train_fused.py:33, 91, 95, 103, 112` (and the
equivalents in `train_homo.py`, `train_st.py`). Everything below the drivers is generic:
node/edge types are built by f-string from a `layer_name`
(`src/graph/gaugegraphnew.py:196`), and the dataset, model and training loops all iterate
`.node_types` / `.edge_types` (`gaugegraphnew.py:205,210,435-443`; `models/gnn.py:306-309`;
`training/logic_hetero.py:56-60,131-135,208-217`).

The only hardcoded literals are the 0D ones — node type `'raingauge'` and edge tuple
`('raingauge','connects','raingauge')` — which a gauge-only run has by construction.
**`'cml'`/`'radar'` literals appear only inside `datasources`-guarded blocks.** There is
no `'satellite'` node type anywhere; 2D is radar-only, so the paper's `G+R+S` Sydney row
has no implementation here.

### What actually blocks it
`train_fused.py:55-73` loads radar and CML **unconditionally, before** the guards, and
uses them to define the timestamp axis:

```python
55:  radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")  # FileNotFoundError
62:  merged_df = radar_df.merge(raingauge_df, on=['timestamp'], how='inner')    # radar defines timestamps
64:  cml_df, cml_coordinates_df = load_cml_dataset(...)                          # xarray
67:  merged_df = merged_df.merge(cml_df, on=['timestamp'], how='inner')          # CML narrows further
```

`train_st.py:182-185` and `train_homo.py` repeat the pattern.

### Minimal change set (FR-203-compliant — 0D/1D/2D path stays intact)
1. `config.yaml:62` → `datasources: ['raingauge']`
2. `train_fused.py:55` → guard radar load with `if 'radar' in datasources`
3. `train_fused.py:60-73` → gauge-only branch deriving `raingauge_df` (with its
   `timestamp` column) directly from the resampled gauge frame. **The only non-trivial
   edit** — the timestamp axis is currently *defined* by the radar∩CML inner join.
4. `train_fused.py:64-68` → guard CML load/merge
5. *(recommended)* replace the hand-written `in_channels_dict` (which hardcodes
   `"radar": 1` in both branches) with `train_st.py:426-431`'s pattern. Currently
   harmless — see §4 — but a latent trap.

Nothing in `src/graph/`, `src/sampling/`, `models/` or `training/` needs to change.

### ⚠️ `paramtuning_gauge.py` is misnamed — it is **not** gauge-only
It never touches `config["datasources"]`, so it inherits `['raingauge','cml']` from the
config file and runs **gauge + CML**. `run_search.sh:18` submits exactly this script, so
any prior Optuna results labelled "gauge" are gauge+CML. This materially affects how any
inherited tuning results should be read.

---

## 3. Entry points (FR-001)

No entry point accepts a config path, seed, experiment name, or data root. Only the two
CNN benchmarks have a CLI. **Zero environment variables are read anywhere.**

**Five top-level scripts execute their full workload at import** (no `__main__` guard):
`train_fused.py:292-293`, `train_homo.py:344-345`, `train_st.py:521-522`, `run_idw.py:64`,
and all three `paramtuning*.py`. Consequence: `from train_fused import train_fused`
**fires a complete 5-fold training run before Optuna trial 0.** Confirmed defect.

| Script | Role | Status |
|---|---|---|
| `train_fused.py` | 5-fold inductive hetero-GNN over gauge + radar + CML | **CANONICAL** |
| `train_homo.py` | Homogeneous-GNN ablation | current (tip commit) |
| `train_st.py` | Spatio-temporal LSTM+GNN variant | current |
| `paramtuning_gauge.py` | Optuna search — **see §2 caveat** | canonical search driver |
| `benchmarks/run_cnn_lstm.py` | CNN / LSTM / CNN-LSTM baselines (`--model`, `--debug`) | current; subsumes `run_cnn.py` |
| `run_idw.py` | IDW baseline | **will crash** — `idw.py:163` saves into `idw_results/` with no `makedirs`, and `regression_plot=True` is hardcoded at `:60` |
| `train.py` | gauge-only precursor | **BROKEN** — calls `train_epoch(..., verbose=, random_noise_masking=)`; current signature has neither → `TypeError` |
| `paramtuning.py` | Optuna, 3 objectives | **BROKEN** — returns a 3-tuple against `directions=["minimize"]` |
| `analyze.py` | Optuna dumper | **unrunnable** — absolute path to another user's home |
| `visualize_error.py` | error plots | **ORPHANED** — reads `predictions/station_errors_fold{i}.csv`, which nothing writes |

**Two silent Optuna no-ops:** `paramtuning_gauge.py:24` writes `weighted_loss_alpha` to
`dataset_parameters` but `train_fused.py:220` reads it from `training_params`; and
`include_lpe` (`:27`) is gated at `gaugegraphnew.py:179`, which reads a **module-level
config from disk at import** (`:17`) and cannot be overridden programmatically. Any
recorded search covers **5 of its 7 nominal dimensions**.

---

## 4. Model contract (FR-007)

**Canonical model:** `GNNInductiveHetero` (`models/gnn.py:275`), constructed at
`train_fused.py:114` (CML branch, active under the shipped config).

```
h = x_dict
for conv in convs:                      # num_layers=5 × HeteroConv(aggr='mean')
    h = conv(h, edge_index_dict, edge_weight_dict=edge_attr_dict)
    h = {k: dropout(relu(v)) for k, v in h.items()}
out = {'raingauge': Linear(hidden → 1)(h['raingauge'])}
```

- **Input** `x_dict['raingauge']`: `[B·N, 6]` — col 0 rainfall, col 1 validity,
  cols 2-5 Laplacian PE. **Z-score normalised** (`train_fused.py:159`).
- **Output:** `[B·N, 1]`, **raw mm/hr, linear space**. Targets are never normalised
  (`train_fused.py:154` touches `.x` only), so no de-normalisation step exists or is
  needed. No output activation; `clamp(min=0)` applied **only at eval**
  (`logic_hetero.py:147, 221`) — so train and val loss curves are not the same functional.
- **Per-edge-type weights:** yes, one independent `GraphConv` per edge type per layer
  (`models/gnn.py:306-309`). Cross-type aggregation is **unweighted `mean`**, so modality
  contribution is fixed 50/50 and adding radar mechanically dilutes each to 1/3.
- **Edge weights are genuinely used** as scalar message scalings, via PyG's
  `GraphConv.message`: `edge_weight.view(-1,1) * x_j`.

> **`in_channels_dict` is decorative.** `models/gnn.py:308` uses lazy
> `GraphConv((-1,-1), hidden)`; the dict is only stored in `self.config`. So
> `raingauge_features = 6 if include_lpe else 2` has **no effect on the model** and a
> wrong value goes undetected. It *is* load-bearing in `gnn_homo.py` and `gnn_st.py`.

**Training:** weighted MSE `w = 1 + 0.18·log1p(target)` (`logic_hetero.py:453-454`);
Adam lr 3e-4, wd 1e-4; 100 epochs; patience 10; batch 128 timesteps; `shuffle=False`;
`clip_grad_norm_(1.0)`; no scheduler. Leave-one-out masking of cols `[:2]` one node
position at a time, gradients accumulated at `1/N` scale, one `optimizer.step()` per batch.

**Dead/broken code inside the model layer:** `models/hgnn.py` is a **`SyntaxError` stub**
despite carrying the project's headline name; `models/models.py` (`nn.module` lowercase)
and `models/base_model.py` fail to import; `training/loss.py`, `training/training.py`,
`training/logic.py`, `training/metrics.py`, `src/validation_stats.py`, `src/dataset.py`
are dead, four of them failing to import. `models/gnn_radar.py` — the **only** class
modelling radar as a first-class 2D layer with attention — is unimported legacy.

---

## 5. Graph, splits and leakage (FR-005, FR-006)

**Edge tuples:** `('raingauge','connects','raingauge')`, `('cml','connect','cml')`,
`('radar','connect','radar')`, plus `('cml'|'radar','connects','raingauge')`.
Note the inverted naming: `'connects'` = support→gauge. With `is_directed: True`
(`config.yaml:68`) cross-support edges are **strictly one-way into the gauges**.
`ToUndirected()` runs at `gaugegraphnew.py:50-52`, i.e. *before* `add_heterodata()`.

> `src/graph/diagnostics.py:68-79` prints a ⚠️ warning that one-directional edges "will
> NOT contribute to raingauge predictions" and advises `is_directed: False`.
> **That warning is inverted** — `cml→raingauge` is precisely the feeding direction.
> Do not act on it.

**Three different distance metrics in one pipeline:**

| Pair | Metric | k |
|---|---|---|
| gauge–gauge | **Euclidean on raw (lon,lat) degrees** (`gaugegraphnew.py:105`) | 5 |
| radar→gauge | **haversine** on radians (`:320-328`) | 9 |
| CML→gauge | **projected metres**, EPSG:3857 (`:294-295`) | 5 (×2 endpoints) |

**Edge-weight normalisation is GLOBAL per split, not per-node** (`:172-173`,
`max_attr = max(edge_attr)`). Because train/val/test graphs contain different node sets,
**each split gets a different divisor** — a real train/test scale shift. This also
diverges from the paper, which specifies per-node normalisation
`w_ij = (1/d_ij)/max_k(1/d_kj)`.

**Splits are purely spatial** (`src/sampling/main.py:232-442`): KMeans(k=8, hardcoded) on
(lon,lat), one test station per cluster per fold, 20% per-cluster validation carve-out.
Per fold ≈ 72% train / 18% val / 10% test. **Every timestamp appears in all three splits**
— inductive over stations, fully transductive over time. `seed=123` is hardcoded at
`train_fused.py:83`; `config.training_params.seed` is read only by the CNN benchmarks.
A station in a cluster's `train_pool` is in train for all 5 folds and **never evaluated**.

**Normalisation is leak-free across splits** (train-only stats, `.x` only) — FR-304 satisfied.

### Masking-semantics defects
1. **Zeroing happens in normalised space**, so a "masked" node decodes to the
   **training-mean rainfall**, not to "no observation" (`logic_hetero.py:75/138/206` runs
   after `train_fused.py:174-186`).
2. **The validity flag is destroyed by normalisation.** With `uptime_threshold: 0.9` the
   flag is near-constant; `std(...).clamp(min=1e-8)` (`train_fused.py:149`) drives
   normalised validity to ≈0 for *all* nodes. In the probe, masked and unmasked nodes
   became **bit-identical on that channel** — the model has no reliable "unobserved"
   sentinel. This also makes `train_st.py` raise "No valid target timesteps found" on a
   fully-clean dataset (`st_dataset.py:60-62`).
3. **LPE is computed independently per split** (`gaugegraphnew.py:179-191`) on three
   different graphs. Eigenvector signs are arbitrary and node sets differ, so LPE features
   are **not comparable between the train graph and the test graph**, yet train-derived
   normalisation stats are applied to them. *Niramay's tree explicitly fixed this
   (commit `cc23a25`); Jylow has not.*
4. **Density mismatch.** Train graph ≈72% of gauges, test graph 100%. At test time all
   train *and* validation gauges keep their true rainfall as inputs. The model is
   evaluated against a **denser** network than it ever trained on.
5. **The target IS input feature 0** (`gaugegraphnew.py:140-144` — same tensor). Only the
   runtime zeroing prevents total leakage. All forward paths were checked and all mask
   correctly, but there is **no assertion guarding it** — the design is one missing line
   from a 100% leak.

**No RNG seeding of model init, dropout, or LPE sign.** `grep torch.manual_seed` finds one
hit, in a dead module. Runs are **not reproducible** (NFR-001 risk).

### IDW baseline is not comparable to the GNN
- Uses `split_info['ml']['train']` (~72%) while the GNN sees ~90% at test → **IDW is
  handicapped by ~18% of the network.**
- **Different ground truth:** `run_idw.py:29` resamples `.first()`; `train_fused.py:52-53`
  resamples `.mean()`. The two pipelines score against *different target values*.
- **Different timestamp sets:** IDW merges radar `how='left'`; the GNN inner-joins radar
  **and** CML.
- Kriging has **no runnable path** — `benchmarks_analysis.ipynb` imports
  `run_kriging_benchmark`, which exists nowhere in the repo.

---

## 6. What blocks a fresh run (FR-003)

**Hard blockers:**

1. **No data.** `database/` is gitignored and absent. Required, all hardcoded relative:
   `database/raingauge_nea_data/{2024,2025}/weather_station_data_{year}.csv`,
   `.../weather_stations_{year}.csv`,
   `database/combined_microwave_link_data_2024_2025.nc`,
   `database/processed_radar_dataset.pkl`, and `database/area_23/**/*.tif` to build it.
   No fixture, sample, reference output or checkpoint is committed.
2. **`requirements.txt` is not installable** — a `pip freeze` from a **conda** env;
   ~40 of 139 lines are local build artefacts
   (`affine @ file:///Users/ktietz/demo/mc3/conda-bld/...`). No lockfile, no
   `pyproject.toml`, no pinned torch/PyG index.
3. **The checked-in `.venv` cannot import any entry point.** *Verified directly:*
   `geopandas`, `shapely`, `rasterio`, `xarray`, `netCDF4` are all **missing**, and
   `src/utils.py:7-8` imports xarray+rasterio at module scope. Every trainer imports
   `src.utils.read_config`, so the failure precedes line 1 of any run:
   ```
   src/graph/gaugegraphnew.py:7  ModuleNotFoundError: No module named 'geopandas'
   ```
4. **🛑 `eda.ipynb` is a MANDATORY pipeline step.**
   `database/processed_radar_dataset.pkl` is read by **five** `.py` entry points but its
   only writer is `src/radar/utils.py:74`, whose sole caller is `eda.ipynb` cell 14. The
   `.py` alternative `src/radar/preprocessor.py` **cannot be imported** (references a
   nonexistent `load_weather_station_dataset`), and `src/graph/gaugegrid.py` /
   `gauge_radar_graph.py` are **0 bytes**. **The pipeline is not reproducible from CLI
   alone.** The notebook's crop bounds (`bottom: 1.19`) also disagree with
   `config.yaml`'s `geography_bounds.bottom: 1.188`.
   *Mitigation for our purposes: a 0D-only run does not need radar, so this blocker is
   bypassed by the §2 change set.*
5. **Everything must run with CWD = repo root** — three modules read bare `"config.yaml"`
   at import time (`training/logic_hetero.py:20`, `src/graph/gaugegraphnew.py:17`).
6. Directories never created: `idw_results/`, `results/`, `logs/`.
7. `run_train.sh` / `run_search.sh` are **SLURM** scripts for the NUS cluster
   (`--gpus=h100-47`), and activate `venv/` while the repo has `.venv/`.

**Dead config keys:** `database_folder`, `cml_file`, `test_size`, `country_info.*`,
`model.cml_mode`, `dataset_parameters.seed` (fully dead); `train_size`,
`analysis.selected_station` (dead in practice). `model.weight_decay: 1e-4` **parses as a
string** (PyYAML, no decimal point) — every live consumer wraps it in `float()`, so it
works, but is one un-cast consumer from a `TypeError`.

---

## 7. 🛑 Sydney-specific blockers for the port (feeds FR-301)

These will break on contact with Niramay's data and are the core of the adapter work:

| # | Location | Assumption | Niramay's data | Consequence |
|---|---|---|---|---|
| 1 | `src/raingauge/utils.py:30-32` | timestamp format `"%Y-%m-%dT%H:%M:00+08:00"` — **SGT offset baked into the format string** | `2021-01-01 00:00:00`, tz-naive | **`ValueError` on parse** |
| 2 | `src/raingauge/utils.py:33` | `value * 12` — 5-minute accumulation → mm/hr | **hourly** accumulation, and *units are suspect* (see `AUDIT_NIRAMAY.md` §8.2) | **~12× scale error**, compounding an already-unresolved ~10× unit question |
| 3 | `train_fused.py:52-53` | resample to 15-min mean | hourly native | upsampling to 15-min produces mostly-NaN or forward-filled values |
| 4 | `src/sampling/main.py:292` | `n_clusters=8` hardcoded, tuned for 51 SG stations | 173 (188 rows) Sydney stations | splits not tuned; not configurable |
| 5 | `src/graph/radargraph.py:24-25` | 0.01° grid, `+0.005` half-cell offsets | n/a for 0D | irrelevant in gauge-only mode |
| 6 | `src/graph/cmlgraph.py:21` | `coordinates_df.iloc[::2]` assumes exactly 2 rows per link | n/a for 0D | irrelevant in gauge-only mode |

Items 1–3 are unavoidable for a Sydney gauge-only run and must be handled by the adapter
(FR-301), **not** by editing the Singapore loader in place (NFR-004, FR-203).

---

## 8. Useful for the golden fixture (FR-401/402)

**Niramay's Sydney gauge data is already reachable from this repo's history** — 175 files
under `database/australia/` at commit `cc23a25`, verified:

```bash
git ls-tree -r --name-only cc23a25 -- database/australia   # 175 files
```

So a deterministic fixture can be cut without re-fetching anything or crossing branches.

---

## 9. Open questions

1. **Where is the Singapore data?** Without it, Level A validation for Jylow (reproducing
   its own paper results) is **impossible**. We can never sanity-check the framework
   against a known-good number before porting Sydney data in — a standing risk to any
   conclusion drawn from the port.
2. **Which Optuna study, if any, produced the paper's hyperparameters?** Three DBs are
   referenced, none committed, and the canonical search has two inert dimensions (§3).
3. **Was `is_directed: True` intended?** `diagnostics.py`'s advice contradicts it.
4. **Do the paper's Sydney numbers come from this tree?** If so, §5's LPE inconsistency,
   global edge-weight normalisation, and IDW non-comparability all apply to them.
