# AUDIT — Niramay implementation (Australian / Sydney, 0D gauge-only)

> **Audit snapshot:** Later dated notes inside this document supersede its opening
> “not attempted” status. Native reproduction, the opt-in edge fix, and BoM source
> investigation recorded here is complete. The hand-exported
> data is regression evidence only, not an accepted scientific dataset.

**Audited tree:** worktree `../COMP3888-niramay`, branch `niramay-clone` @ `4fcb313`
(`4fcb313` = `8368cbf` + a docs-only README commit; no code differences).
**Audit date:** 2026-08-29 · **Method:** read-only inspection + re-execution of the
repo's own loader/sampler code paths against the bundled data.
**Current status:** Phase 0, native reproduction, the opt-in edge-normalisation
correction, and BoM investigation are complete. The original pre-run status is retained
in the dated narrative below; §8 contains the executed evidence.

---

## 1. Provenance and relationship to the source paper

This code corresponds to *"Spatial Support Matters: Geometry-Aware Graph Fusion for
Rainfall Field Reconstruction"* (arXiv:2607.01621), whose author list includes
Niramay Kachhadiya. **It implements only the paper's gauge-only ablation**
(`HGNN (G)`, Table 1), not the multi-support fusion that is the paper's contribution.
There are no non-gauge node types anywhere in this tree.

Deltas from the paper's stated method are recorded in §7.4. They matter because a
result from this code cannot be compared to the paper's headline `HGNN (G+C+R)` number.

---

## 2. Entry points (FR-001)

Execution is **CLI-driven**, with a `config.yaml` supplying data paths and two defaults.
There are no notebooks and no mandatory preprocessing step in-repo.

| Entry point | Purpose | Invocation |
|---|---|---|
| `train.py` | Train the GNN across spatial CV folds | `python3 train.py [--hidden --layers --knn --lr --epochs --patience --folds --fraction --include_lpe --name --batch_size]` |
| `benchmark_idw.py` | IDW statistical baseline | `python3 benchmark_idw.py [--folds --n_nearest --power --name]` |
| `paramtuning.py` | Optuna sweep, resumable via SQLite | `python3 paramtuning.py [--n_trials N]` |
| `run_smoketest.pbs` | NCI Gadi job — 5% data, 1 fold, 20 epochs | `qsub` (paths hardcoded to `/scratch/no50/nk4326/…`) |
| `run_paramtuning.pbs` | NCI Gadi job — Optuna, 48h walltime | `qsub` (same hardcoded paths) |

`train.py` and `paramtuning.py` both guard execution behind `if __name__ == "__main__"`.
No module runs work at import time.

**Blocking issue for a fresh run:** the PBS scripts hardcode `/scratch/no50/nk4326/refactor`
and a venv at `/scratch/no50/nk4326/venvs/rainfall` (NFR-002 violation). They are reference
only; the CLI commands inside them are portable.

---

## 3. End-to-end call chain (FR-002)

```
train.py:main()
  └─ config.yaml                          (train.py:78-79)
  └─ data/loader.py:load_raingauge_dataset()      (train.py:113)
       ├─ pd.read_csv(cfg.data.gauge_csv)         (loader.py:29)   ← MISSING FILE, see §4
       ├─ raw.pivot(timestamp × station_id)       (loader.py:32)
       ├─ uptime filter ≥ 0.9                     (loader.py:36-39)
       └─ read station_metadata.csv, .isin() join (loader.py:41-56) ← DEFECT, see §8.1
  └─ optional --fraction head-slice of timesteps  (train.py:119-122)
  └─ data/sampling.py:stratified_spatial_kfold()  (train.py:125)
       └─ KMeans(n_clusters=8) on (lon,lat), per-cluster test rotation  (sampling.py:49-64)
  └─ per fold:
       ├─ graph/gauge_graph.py:GaugeGraph(...)    (train.py:138)
       │    ├─ _make_masks()                      (gauge_graph.py:93-103)
       │    ├─ _compute_full_lpe()  (if --include_lpe)  (gauge_graph.py:137-149)
       │    ├─ _build_knn_edges() ×3 splits       (gauge_graph.py:121-135)
       │    └─ _build_heterodata() ×3, ToUndirected()   (gauge_graph.py:71-73)
       ├─ GaugeGraphDataset → PyG DataLoader      (train.py:145-156)
       ├─ models/gnn.py:GNNInductiveHetero(...)   (train.py:158)
       ├─ training/trainer.py:train_epoch()       (train.py:177)
       ├─ training/trainer.py:validate()          (train.py:178)
       └─ training/trainer.py:test_model()        (train.py:224)
  └─ mean RMSE across folds printed               (train.py:232-236)
```

---

## 4. Data dependencies (FR-003)

| Asset | Path in code | Classification | Notes |
|---|---|---|---|
| Combined gauge CSV | `config.yaml:4` → `all_stations_rainfall_hourly_combined.csv` (repo root) | **MISSING / must be generated** | Removed from git in `8e90ef0` "Remove large CSV from tracking"; `.gitignore` excludes it by name. **Every entry point fails without it.** |
| Per-station raw CSVs | `database/australia/raw/*.csv` | **bundled in repo** (173 files, 174 tracked CSVs incl. metadata) | Same schema as the combined file; trivially concatenable — see §9. |
| Station metadata | `config.yaml:5` → `database/australia/station_metadata.csv` | **bundled in repo** | 425 rows, headerless. **Contains duplicates — see §8.1.** |
| Optuna study DB | `sqlite:///optuna_australia.db` (`paramtuning.py:178`) | generated on demand | `.gitignore`d. |
| Experiment outputs | `experiments/<name>/` | generated | `.gitignore`d. A prior `smoketest` run is present in the working tree but untracked. |

No NetCDF/xarray/Zarr readers, no `np.load`, no environment variables, no network fetches.
The only `torch.load` is the checkpoint reload at `train.py:221`.

**The combined CSV is reconstructible** from the bundled per-station files — verified,
§9 — so this repo is self-contained for the gauge-only baseline. No NCI dependency.

---

## 5. Data dictionary (FR-004)

### 5.1 Raw gauge observations — `database/australia/raw/<station_id>_rainfall_hourly.csv`

| Column | Dtype | Meaning | Notes |
|---|---|---|---|
| `timestamp` | str → `datetime64[ns]` | observation time | **timezone-naive**; no tz recorded anywhere. Assumed local (AEST/AEDT?) — **unverified**. |
| `station_id` | int64 | gauge identifier | numeric only; matches the filename in all 173 files (0 mismatches). |
| `rainfall_mm` | float64 | hourly accumulation | **Name is misleading — values are not millimetres. See §8.2.** |

### 5.2 Station metadata — `station_metadata.csv` (headerless)

Columns assigned positionally at `loader.py:41-45`: `name, id, network, latitude, longitude`.
**Coordinate order in the file is (latitude, longitude)**, but every downstream consumer
reads `[["longitude","latitude"]]` (`sampling.py:44`, `gauge_graph.py:124`) — the swap is
handled correctly and consistently.

`id` is mixed-format: 354/425 purely numeric, 71 alphanumeric (`GW040994`, …).
Only numeric IDs ever match gauge data.

### 5.3 Model tensors

| Tensor | Shape | Content | Where |
|---|---|---|---|
| `x` | `[N, T, 2]` or `[N, T, 6]` | col 0 `log1p(rainfall)`, col 1 validity flag, cols 2-5 Laplacian PE (if `--include_lpe`) | `gauge_graph.py:156-160`, `:186-189` |
| `y` | `[N, T, 1]` | **raw** rainfall, untransformed | `gauge_graph.py:161` |
| `edge_attr` | `[E]` | inverse distance `1/d`, globally normalised | `gauge_graph.py:133`, `:172` |
| `mask` | `[N]` bool | prediction targets for the split | `gauge_graph.py:177` |

**Transform asymmetry:** inputs are `log1p`-compressed, targets are raw. The model
therefore maps log-space inputs → linear-space mm. Metrics are computed in raw units
with no de-normalisation step required. Note this differs from the paper, which
z-scores per node type using training-split statistics (§7.4).

---

## 6. Splits and leakage (FR-005)

**Splitting is purely spatial.** `stratified_spatial_kfold` (`sampling.py`) partitions
*stations*; the time axis is untouched. All three splits carry the full 43,824-timestep
record, so **every timestamp appears in train, validation and test**. The design is
inductive over stations, fully transductive over time. This is intentional (it mirrors
the paper's protocol) but must be stated in any result.

Mechanism: KMeans(k=8) on (lon, lat) → per cluster, `idx[:n_splits]` becomes the test
rotation and `idx[n_splits:]` becomes a permanent training pool (`sampling.py:56-64`).
Validation is drawn per-cluster from the training pool at 20% (`sampling.py:88-98`).

Normalisation statistics: **none are computed**. `log1p` is a fixed transform with no
fitted parameters, so FR-304 (train-only statistics) is trivially satisfied — but also
means the pipeline diverges from the paper's z-scoring.

Target masking: features `[:2]` (rainfall + validity) are zeroed; LPE columns are
preserved. Training masks **one node at a time** (`trainer.py:60-76`); validation and
test mask **all held-out nodes at once** (`trainer.py:112-113`, `:204-205`). This
train/eval asymmetry makes evaluation systematically harder than training.

**Two serious defects found — see §8.1 and §8.3.**

IDW baseline (`benchmark_idw.py:51-52`) predicts held-out stations using
`train + validation` stations only — the held-out value is never available to it.
That part is correct.

---

## 7. Graph and model contract (FR-006, FR-007)

### 7.1 Graph
- **Node types:** one — `raingauge`. No 1D or 2D supports exist.
- **Edge types:** one — `('raingauge','connects','raingauge')`, made undirected by
  `ToUndirected()` (`gauge_graph.py:71-73`).
- **Neighbours:** `NearestNeighbors(n_neighbors=knn+1, algorithm="ball_tree")` on raw
  (lon, lat) degrees — **Euclidean on unprojected degrees, not haversine**
  (`gauge_graph.py:126`). At Sydney's latitude a degree of longitude is ~0.83× a degree
  of latitude, so the metric is anisotropic and kNN is mildly biased N–S.
- **Edge weight:** `1/d`, or `1e6` when `d == 0` (`gauge_graph.py:133`), then divided by
  the **training graph's global maximum** (`gauge_graph.py:171-172`).
- **Topology varies by split:** train sees train nodes; val sees train+val; test sees all
  (`gauge_graph.py:105-111`). Edge sets and LPE subsets therefore differ per split.

### 7.2 Model — `GNNInductiveHetero` (`models/gnn.py:233`)
`num_layers` × `HeteroConv({et: GraphConv((-1,-1), hidden)}, aggr='sum')`, ReLU between
layers, then `Linear(hidden → 1)`. No dropout, no normalisation layers, no output
activation. With a single edge type the heterogeneous machinery is vestigial.
Edge weights are genuinely consumed, passed as `edge_weight_dict` (`models/gnn.py:281`).

Six further classes in `models/gnn.py` (`HeteroGNN`, `HeteroGNN2`, `HeteroGCNGNN`,
`HeteroSAGEGNN`, `GNN`, `GNNInductive`) are **legacy and unreferenced** by the live path;
the file's own docstring says so. They use an older `general_station`/`rainfall_station`
convention and none model line or grid geometry.

### 7.3 Training
Loss **unweighted MSE** in raw mm (`trainer.py:72`); Adam; `lr` default 1e-3;
**no weight decay in `train.py`** (only swept in `paramtuning.py:107`); epochs 100,
patience 20; batch 1024 timesteps; `shuffle=False`. One `optimizer.step()` per batch after
looping all node positions (`trainer.py:74-76`). Best-val checkpoint saved and reloaded
before test (`train.py:187-190`, `:220-223`). Metrics: global RMSE + Pearson r, plus
**stratified RMSE/bias** over dry / light / moderate / heavy bands (`trainer.py:128-160`).

**Seeding:** only the split is seeded (`config.yaml:7` `split_seed: 123`). `grep` finds
no `torch.manual_seed` in the tree. Model init and the `AddLaplacianEigenvectorPE`
random sign flip are unseeded, so **runs are not bitwise reproducible** (NFR-001 risk).

### 7.4 Deltas from the paper (record before any comparison)
| Paper | This code |
|---|---|
| per-node edge-weight normalisation `w_ij = (1/d_ij)/max_k(1/d_kj)` | **global** max over the whole training graph (`gauge_graph.py:172`) |
| z-score per node type, training-split stats | `log1p` on inputs, raw targets |
| dropout p=0.15 | none |
| 5 layers, lr 3e-4, weight decay 1e-4, patience 10 | 4 layers, lr 1e-3, no wd, patience 20 |
| LPE always on | opt-in via `--include_lpe`, **off by default** |
| inference at arbitrary target grids | not implemented; predicts only at fixed held-out stations |

---

## 8. Findings

### 8.0 🛑🛑 CRITICAL — the edge-weight normalisation reduces the GNN to a per-node MLP

**This is the root cause of the baseline not learning, and it is independent of §8.1.**

Two *distinct* station IDs sit 0.1 metres apart — the same physical gauge registered
under two networks with **different** IDs, so de-duplication by ID cannot remove them:

```
212285  GREAVES CREEK DAM   (-33.654723, 150.308889)
563086  Greaves Creek       (-33.654722, 150.308889)
        separation = 4.506e-07 deg  (~0.1 m)  ->  1/d = 2,219,128.6
```

The next-closest pair is 571 m apart (`1/d = 194.3`), so this single pair is
**~11,400× stronger than anything else in the graph**. `gauge_graph.py:172` then divides
*every* edge weight by the **global** maximum:

| | with duplicates (188 nodes) | de-duplicated (173 nodes) |
|---|---|---|
| median raw weight | 13.746 | 13.202 |
| max raw weight | 2,219,128.6 | **2,219,128.6** (unchanged) |
| median **normalised** weight | 6.19e-06 | 5.95e-06 |
| edges < 1e-3 after normalisation | 928 / 940 (98.7%) | 863 / 865 (**99.8%**) |

`GraphConv` computes `x'_i = W₁·x_i + W₂·Σ_j(e_ji · x_j)`. With `e_ji ≈ 6e-6`, the
neighbour-aggregation term is **five orders of magnitude below the self term**. The
network degenerates into a per-node MLP with effectively no message passing — which for a
spatial-interpolation GNN is fatal. During training the target node's own features are
zeroed, so that MLP can only emit a constant.

**Confirmed empirically.** For the fold-0 test set (9 nodes × 2,191 timesteps = 19,719
records, target mean 1.9642):

| Predictor | RMSE |
|---|---|
| predict 0 everywhere | 9.0171 |
| predict the mean | 8.8005 |
| **bundled `experiments/smoketest/`** | **8.9024** |
| **this reproduction** | **8.9645** |

Both runs land **inside the constant-predictor band [8.801, 9.017]**, with
Pearson r = −0.0131 (bundled: `NaN`). The model has learned nothing beyond a constant.
This also explains why the two runs agree to 0.7% despite different code, batch size and
pandas version — **the number is a property of the data, not the model.**

**The paper is immune to this.** arXiv:2607.01621 §3.2 Eq. (1) specifies **per-node**
normalisation, `w_ij = (1/d_ij) / max_{k∈N(j)} (1/d_kj)`, where each node normalises by
its *own* neighbourhood maximum. One pathological pair cannot then crush the global graph.
Both implementations deviate to a global max — a delta already noted in §7.4, here shown
to be catastrophic on Sydney data.

The retained `src/graph/gaugegraphnew.py` path inherited the same graph-wide
normalisation defect, so it is covered by the same regression test and corrected mode.

**Minimal correction:** implement the paper's per-node normalisation. Note that the `1e6`
guard at `gauge_graph.py:133` is a second-order concern — the real problem is that *any*
very small distance dominates a global normaliser.

#### 8.0.2 ✅ IMPLEMENTED (2026-08-30) — per-node normalisation, authorised by operator

Operator authorisation received (per the §1 override this document's companion
`HANDOFF.md` requires) to implement arXiv:2607.01621 §3.2 Eq. 1 —
Low, Kachhadiya, Herath, Rasnayaka & Marshall, *"Spatial Support Matters: Geometry-Aware
Graph Fusion for Rainfall Field Reconstruction"* — as a separate, opt-in experiment
alongside the unmodified baseline (FR-420 / BR-004).

**What changed, both trees:** a new `edge_normalisation` parameter, `"global"` (default)
or `"per_node"`. `"global"` is a no-op change — the exact original computation path,
byte-for-byte, runs when the parameter is omitted or left at its default. `"per_node"`
divides each node's own row of inverse-distance weights by that row's own maximum,
instead of one scalar shared across the whole graph.

- `../COMP3888-niramay/graph/gauge_graph.py` — `_build_knn_edges` and `_build_heterodata`;
  `train.py` gained a matching `--edge_normalisation {global,per_node}` CLI flag (also
  folds into the auto-generated experiment name and the performance-log config, so a
  per-node run is never silently conflated with the baseline).
- `src/graph/gaugegraphnew.py` (this repo) — the gauge-to-gauge edge calculation. The
  correction is wired through the retained gauge-only driver and protected by tests.

**Validation status — ✅ DONE (2026-08-30), full end-to-end.**

A fresh venv was built in this session (`.venv`, Python 3.13, CPU-only torch —
`pip install --index-url https://download.pytorch.org/whl/cpu torch`, then
`torch_geometric numpy pandas scikit-learn scipy matplotlib optuna pyyaml tqdm pytest`),
landing on the exact versions `HANDOFF.md` §3 documents (torch 2.13.0, PyG 2.8.0.post1,
pandas 3.0.5, numpy 2.5.2, sklearn 1.9.0, pytest 9.1.1) — that committed venv was not
present in this session's filesystem, so this one was built from `requirements.txt`
instead of reused.

- **Math confirmed standalone** (plain numpy against real `station_metadata.csv`, before
  the venv existed):

  | | median normalised weight | edges below 1e-3 |
  |---|---|---|
  | global (baseline) | 9.6e-06 | 99.1% |
  | per_node (fix) | 0.593 | 0.6% |

- **Both new tests pass** against the real `GaugeGraph` class:
  `.venv/Scripts/python.exe -m pytest tests/ -q -rx` → **20 passed, 7 xfailed** (up from
  18/7 — `test_global_normalisation_default_matches_baseline_defect` and
  `test_per_node_normalisation_fixes_the_real_graph` both pass). Fixing this also
  surfaced and fixed a real, pre-existing gap in `tests/conftest.py`'s `gauge_wide`
  fixture: it pivoted `station_id` without casting to `str` the way the real
  `data/loader.py:48` does, so it could never have been fed into `GaugeGraph` before —
  no earlier test exercised that path.
- **Full end-to-end re-run — HANDOFF.md §7's "done when" criterion, met:**
  ```
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe train.py --hidden 32 --layers 4 --knn 5 \
      --lr 0.001 --epochs 20 --patience 10 --folds 1 --fraction 0.05 --batch_size 128 \
      --include_lpe --edge_normalisation per_node --name smoketest_per_node_norm
  ```
  (same params as the reference native run, `run_niramay_smoketest_native.log`, plus the
  new flag; `PYTHONIOENCODING=utf-8` works around an unrelated Windows console-encoding
  crash in `data/loader.py`'s `≥` character — a platform issue, not a code defect, not
  fixed in source). Full log: `smoketest_per_node_norm.log`.

  | | RMSE | Pearson r | vs constant-predictor band `[8.801, 9.017]` |
  |---|---|---|---|
  | baseline (native reproduction) | 8.9645 | −0.0131 | inside — learned nothing |
  | **per_node (this fix)** | **5.3024** | **0.7985** | **well outside — genuine learning** |

  RMSE fell 41%; Pearson r went from ~zero to strong positive correlation. Stratified
  breakdown (fold 0 test set): dry 2.05/+0.48, light 1.22/+0.10, moderate 5.78/+1.21,
  heavy 20.88/**−7.87** (1,004 samples) — heavy-rain under-prediction bias is large and
  worth a follow-up, but is a separate, pre-existing MSE-regression tendency, not a defect
  introduced by this change.

**Comparison to the paper's own Sydney numbers — NOT a validated reproduction, read
carefully.** The paper reports (graph-only ablation, `HGNN(G)`, Table 1): IDW 0.667,
HGNN(G) 0.686 mm/h. Our 5.3024 is in `rainfall_mm`'s raw, unconfirmed unit — if the §8.2
÷10 hypothesis is right, that's ≈0.53mm, numerically *better* than both paper figures.
**Do not read that as a successful reproduction.** Five independent reasons it isn't one:

1. The ÷10 units hypothesis is unconfirmed (§8.2) — if wrong, the honest number is 5.3024,
   far worse than the paper, not better.
2. **Different station population AND different time period.** §11 item 3: the paper's
   Sydney testbed is **2022 only, 75 gauges**; this dataset is **2021–2025, 173 stations**
   (188 with duplicates) — confirmed by reading the paper itself, not inferred. Selection
   rule undocumented for either. Not the same evaluation set, in two independent ways.
3. Smoketest scale only — 5% of timesteps, 1 fold, not the paper's full data/full spatial
   CV.
4. Known architecture/hyperparameter deltas from §7.4 (dropout, weight decay, feature
   normalisation, LPE defaults) — this run doesn't match the paper's setup.
5. **§8.1's duplicate-metadata leakage is still present and unfixed in this run.** That
   matters more here than it might elsewhere: some of the same near-duplicate stations
   behind §8.1's train/test leakage now carry real, non-negligible edge weight (because of
   *this* fix), so the GNN can exploit that leakage through the graph more than it could
   under the old, collapsed weights. Part of the RMSE improvement could be leakage, not
   genuine generalisation — not separable with the current run.

Net: the fix demonstrably repairs the mechanism the paper's method depends on (solid,
directly measured). The fact the resulting number lands in a plausible range under an
unconfirmed unit correction is a mildly interesting coincidence worth mentioning to
whoever resolves the units question — not evidence this is a working reproduction.

#### 8.0.1 The Greaves Creek pair are NOT the same gauge

> **✅ RESOLVED 2026-08-31 (BOM KiWIS API, back online).** `getStationList` returns
> `563086` as "Greaves Ck TBRG" (internal `station_id` 49565602) with **true coordinate
> (−33.654722, 150.308889)** — Blue Mountains, not the copy-pasted `212285` coordinate.
> The exclusion can be lifted by substituting this coordinate; the analysis below stands
> as the historical record of the metadata defect.

Tested directly by comparing the two series over all 43,824 timestamps:

| | 212285 | 563086 |
|---|---|---|
| bit-identical? | **No** | |
| Pearson r | **0.9247** | |
| rows differing | 3,211 (7.33%) | |
| max abs difference | 144.0 | |
| annual total | 7,497 units/yr | **10,895 units/yr (+45%)** |

Contrast the 15 same-ID pairs of §8.1, which *are* bit-identical (`.equals()` True,
r = 1.000000). Two truly co-located gauges would correlate above ~0.99, not 0.92, and
would not differ by 45% in annual total.

**So this is a coordinate-quality problem, not a duplication problem.** These are two
genuinely distinct instruments, and one of them almost certainly carries the other's
coordinates. It **cannot** be resolved by dropping a row — that would discard real
observations. Resolution needs an external authority (BOM WaterData, the same source the
paper cites as its reference [4]).

**Partial resolution (2026-08-30), via BOM's published rainfall station catalogue
(`rainfall.json`, see §8.1.2 for source and outage caveat):**

- `212285` — BOM lists `GREAVES CREEK AT GREAVES CREEK DAM`, Water NSW, at
  `(−33.65472266, 150.30888921)` — an **exact match** to `station_metadata.csv`. **This
  id's coordinates are confirmed correct.**
- `563086` — **absent from BOM's catalogue entirely.** Checked directly (no `GREAVES` hit
  under any other id; the neighbouring range `563085`–`563089` has no entries at all,
  though `563084` and `563090` do). BOM cannot currently confirm or supply a coordinate
  for this id.
- However, `station_metadata.csv`'s existing coordinate for `563086`
  (`−33.654722351, 150.308888882`) is now known to be **wrong regardless**: it differs
  from `212285`'s coordinate only in the 8th decimal place — i.e. it is a copy of
  `212285`'s location, not a real independent survey point. Combined with §8.0.1's proof
  that the two series are genuinely different instruments (r=0.9247, +45% annual total),
  the current coordinate for `563086` cannot be correct and should not be used as-is.

**Still open:** `563086`'s true coordinates. BOM's public catalogue doesn't carry this id
under the "Rainfall" parameter type — next step is direct contact with the data owner
(`network` = Water NSW per the existing, now-suspect metadata row) rather than further
scraping. **Do not substitute a guessed coordinate** — per FR-102/§17, this stays open for
a human decision, though it's now narrowed to "what is 563086's real location" rather than
"is this even a real distinct site."

#### 8.0.2 Closest distinct-ID station pairs

| rank | separation | 1/d | stations |
|---|---|---|---|
| 1 | **~0.05 m** | **2,219,128.6** | 212285 GREAVES CREEK DAM ∥ 563086 Greaves Creek |
| 2 | 571 m | 194.3 | 568045 W'GAMBA MET STATION ∥ 568044 WARRAGAMBA WFP |
| 3 | 1,252 m | 88.6 | 568350 PICTON BOWLING CLUB ∥ 212053 STONEQUARRY @ PICTON |
| 4 | 1,469 m | 75.6 | 212063 NERRIGORAN@THIRLMERE ∥ 568232 THIRLMERE RAINFALL 1 |
| 5 | 1,497 m | 74.2 | 563086 Greaves Creek ∥ 563071 MEDLOW BATH |
| 6 | 1,497 m | 74.2 | 212285 GREAVES CREEK DAM ∥ 563071 MEDLOW BATH |

Rank 1 is **11,400× stronger than rank 2**, and ranks 2-6 are unremarkable for a real
gauge network. The pathology is confined to a single pair — which is precisely why a
*global* normaliser is so fragile here and a *per-node* one is not.

### 8.1 🛑 BLOCKER — duplicate metadata rows cause node inflation and train/test leakage

`station_metadata.csv` has **425 rows but 409 unique IDs**. Fifteen duplicated IDs belong
to stations that have data — the same physical gauge registered under two networks:

```
567102  DURAL (WPS14)  NSW - Water NSW   -33.697500 151.029500
567102  DURAL (WPS14)  Sydney Water      -33.696900 151.027700
```

`loader.py:53` filters with `.isin()` and never de-duplicates, then assigns
`order = range(len(meta))`. Verified consequences (by re-running the repo's own code):

1. **188 graph nodes for 173 real stations.** `GaugeGraph.gauge_df` is `(43824, 188)`;
   all 15 duplicate column pairs are **bit-identical** (`.equals()` True, r = 1.000000).
2. **`_make_masks` (`gauge_graph.py:100-102`) flags a duplicated station in both roles.**
   `id_to_order[...isin(ids)]` returns *both* orders whenever the ID appears in either
   list. Fold 0: station `567102` at nodes [48, 49] has `train_mask=[True,True]` **and**
   `test_mask=[True,True]`. `train_mask & test_mask` is non-empty in 4 of 5 folds;
   `train_mask & val_mask` is non-empty at 4 nodes.
3. **The repo's own sampler prints the symptom:**
   `WARNING: fold 2 test set overlaps previous folds: {'567102'}` — the disjoint-test-set
   guarantee in `stratified_spatial_kfold`'s docstring is already violated at runtime.
4. **The twin is the #1 nearest neighbour for 12 of 15 duplicates**, at distances down to
   0.0018°, giving raw inverse-distance weights ~530 — among the strongest edges present.
   During `train_epoch`'s leave-one-out masking, zeroing node 48 leaves its identical twin
   at node 49 unmasked and maximally weighted: exactly the trivial identity shortcut the
   masking scheme exists to prevent (paper §3.4).

**Impact:** a station used for training is also scored as held-out in 4 of 40
test-station-fold instances (~10%). Any RMSE from this pipeline is optimistically biased,
and any copied split would propagate the contamination into later components.

**Train/validation leakage is more widespread than train/test.** Because `_make_masks`
maps a duplicated id to *both* node orders, stations hold conflicting roles in **all five
folds** — up to 8 stations in fold 4:

| fold | in train ∧ test | in train ∧ validation |
|---|---|---|
| 0 | 567102 | 567083, 567084 |
| 1 | 567087 | 567100, 567102, 567163 |
| 2 | 567102 | 563083, 567083, 567084, 567163, 568156 |
| 3 | 567148 | 563061, 567076, 567083, 567084, 567100 |
| 4 | — | 563061, 563083, 567083, 567084, 567163, 568149, 568156, 568168 |

Validation leakage corrupts **early stopping and model selection**, not merely the
reported test metric.

#### 8.1.1 The 15 duplicated ids are not all the same kind of problem

| id | name (row A) | name (row B) | separation | class |
|---|---|---|---|---|
| **563083** | BLACKHEATH G.C. | **NORTH RICHMOND WFP** | **39.436 km** | 🛑 **ID COLLISION** |
| **567163** | REGENTVILLE | REGENTVILLE RFS | **1.239 km** | 🛑 **ID COLLISION** |
| 568156 | CAMDEN GOLF COURSE | NARELLAN (LODGES RD) | 541 m | ⚠️ ambiguous |
| 567148 | KINGS LANGLEY | KINGS LANGLEY | 363 m | dual registration |
| 568168 | MENANGLE MET STATION | MENANGLE MET STATION | 272 m | dual registration |
| 568149 | CAMDEN BROWNLOW HILL | CAMDEN-BROWNLOW HILL | 262 m | dual registration |
| 568147 | CAMDEN-CAMDEN PK RES | CAMDEN-CAMDEN PK RES | 202 m | dual registration |
| 567102 | DURAL (WPS14) | DURAL (WPS14) | 179 m | dual registration |
| 568130 | WEST CAMDEN STP | WEST CAMDEN STP | 179 m | dual registration |
| 567087 | ST MARYS STP | ST MARYS STP | 162 m | dual registration |
| 567083 | PROSPECT RES | PROSPECT RESERVOIR | 17 m | dual registration |
| 563061 | WENTWORTH FALLS BOD | WENTWORTH FALLS BOD | 12 m | dual registration |
| 567076 | CASTLE HILL STP | CASTLE HILL STP | 0 m | dual registration |
| 567084 | QUAKERS HILL STP | QUAKERS HILL STP | 0 m | dual registration |
| 567100 | RIVERSTONE STP | RIVERSTONE STP | 0 m | dual registration |

**`563083` is the serious one.** Two rows 39.4 km apart with entirely different names,
sharing one id. Since `GaugeGraph` selects the series *by id*
(`gauge_graph.py:54`), **the same rainfall record is placed at both Blackheath and North
Richmond** — injecting a phantom gauge carrying another site's data 39 km away.

**Consequence for the de-duplication decision:** a plain
`drop_duplicates(subset=["id"])` keeps whichever row comes first, so for `563083` and
`567163` it picks a location **arbitrarily**. Those two ids need external resolution
(BOM WaterData) before de-duplication is safe. The remaining 13 are genuine dual-network
registrations within 541 m and can be de-duplicated mechanically.

#### 8.1.2 ⚠️ CORRECTED (2026-08-30) — `563083` is NOT resolved; the static snapshot was incomplete

> **✅ RESOLVABLE 2026-08-31 (BOM KiWIS API, back online).** Both collisions confirmed
> and disambiguated: BOM assigns each site a distinct **internal** `station_id` behind
> the shared public number. `563083` → BLACKHEATH G.C. (`7730155`, −33.646, 150.2901)
> and NORTH RICHMOND WFP (`49645132`, −33.5865, 150.7107). `567163` → REGENTVILLE
> (`49662962`, −33.77028, 150.65917) and REGENTVILLE RFS (`7730470`, −33.7745,
> 150.6716). The API serves each internal station's series separately, so our single
> series per public number can be attributed by direct value comparison — no longer
> blocked on the data owner. The durable fix is to key stations by **internal id** in
> any regenerated dataset (`docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md`).

**Original claim (superseded, kept for the record):** a static export, `rainfall.json`
(NSW open-data mirror of `water-regulations-data-bom-rainfall`,
`datasets.seed.nsw.gov.au`, dataset id `47bb96b8-c7cb-4604-a594-c2dde2bbd44a`), showed
exactly one record for `station_no` `563083` — BLACKHEATH GOLF COURSE — and one for
`567163` — REGENTVILLE RURAL FIRE SERVICE — leading to a "settled, de-dup and move on"
conclusion below.

**That conclusion was wrong for `563083`.** Operator confirmation (2026-08-30): the live
`bom.gov.au/waterdata` interactive portal genuinely shows **two distinct stations** under
`563083` — **North Richmond Water Filtration Plant** and **Blackheath Golf Course** —
i.e. exactly the same two names as `station_metadata.csv`'s colliding rows
(`NORTH RICHMOND WFP` / `BLACKHEATH G.C.`). So this is **not** a `station_metadata.csv`
compilation artifact after all — BOM's own live system carries both registrations under
one number.

**Root cause of my error:** the static NSW mirror export is not a complete dump of BOM's
station register — it silently dropped the North Richmond WFP registration for `563083`
(possibly filtered to active/current stations, or scoped to a subset of data owners; not
established). I could not independently re-verify against the live KiWIS API myself
because it returned `DatasourceError: Error connecting to WDP` / timed out on every
request type tried across the session (total outage, not query-specific) — this
correction rests on the operator's direct observation of the live portal, which is more
authoritative than my snapshot.

**Status:** `563083` de-duplication is **still blocked** — needs the live portal's own
per-registration detail (`from`/`to` date ranges, `dataowner` codes, or an explicit
"current" flag) to determine which of the two registrations is live vs superseded, or
whether both are simultaneously active phantom nodes. Retry the live API before deciding,
or read the two records directly off the portal page.

**`567163` is also confirmed NOT resolved.** Operator confirmation (2026-08-30): the live
portal shows two distinct stations under `567163` too — **Regentville Rural Fire Service**
and **Regentville (Factory Road)** (BOM displays it as "Regentville" with "Factory Road" as
a bracketed qualifier, not a separate station name). Naming now maps cleanly to both
existing metadata rows:

| BOM name | metadata row | network |
|---|---|---|
| Regentville Rural Fire Service | `REGENTVILLE RFS` | Sydney Water |
| Regentville (Factory Road) | `REGENTVILLE` | Water NSW |

So both rows are accounted for — this isn't a third unmapped site. It's still a genuine
BOM-side id collision (two real, ~1.24 km-apart sites sharing `567163`), just no longer an
open naming question.

**Retracted:** the claim that "BOM's numbering space has no collisions" and the inference
that the `station_metadata.csv` id collisions are purely a compilation artifact. **Both**
`563083` and `567163` carry two genuinely distinct sites under one BOM station number on
the live register — the collision is real on BOM's side, not manufactured by whoever built
`station_metadata.csv`. This also means the earlier claim that BOM's own numbering is
strictly 1:1 (used to explain away the `station_metadata.csv` duplicates) no longer holds
in general — it was based on an incomplete static snapshot, not the authoritative live
register.

**Consequence for de-duplication:** both ids are back to fully blocked, same as before any
of this session's investigation — a coin-flip `drop_duplicates` is still unsafe for either.
Resolving which registration is the "right" one to keep (if either should be dropped at
all, rather than treated as two real nodes) needs the live portal's per-registration detail
(`from`/`to` date ranges, `dataowner` codes, active/historical status) — not yet obtained.

### 8.2 🛑 BLOCKER — `rainfall_mm` is almost certainly not millimetres

> **⚠️ SUPERSEDED 2026-08-31 — the ÷10 hypothesis is OVERTURNED by direct comparison
> against BOM's tip-level records (KiWIS API, back online).** BOM serves rainfall in
> true mm (0.5 mm tips). For a verified January-2023 storm at `568163`, our hourly
> values are **exactly 6× true mm** — the signature of an export that *summed* six
> 10-minute intensity (mm/h) readings per hour instead of averaging them. But the factor
> is **not constant**: June-2021 at the same station gives ~6.75–9.4× against BOM's
> QC'd daily totals, and every station-year contains 0.5-quantum values impossible under
> a constant ×6. Conclusion: the hand export's values are **internally inconsistent
> (time- and station-varying distortion), not repairable by any global scale factor** —
> worse than a unit mislabel, because the training targets themselves are inconsistently
> scaled. Also found: **five all-zero stations** (`211010`, `212063`, `568057`,
> `568232`, `568295`) — dead series that pass the uptime filter because zeros count as
> present, while BOM shows those gauges live. Timestamps follow the **local Sydney
> clock** (AEDT in summer, AEST in winter — verified +1 h vs BOM in January, aligned in
> June). Remedy: regenerate from the API in true mm — verified feasible, specified in
> `docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md`. The climatology analysis
> below stands as the record of how the problem was first detected.

Median station annual total is **9,321 units/yr**. Sydney climatology is ~1,210 mm/yr;
western-Sydney stations run 700–900 mm/yr. Values are quantised to exactly 0.5 steps,
max 150.0/hr, with ~1% of rainy hours above 99 — impossible as mm/hr.

÷10 gives ~932 mm/yr and a 15 mm/hr max, entirely plausible for this station mix,
pointing to **0.1 mm units**. Not confirmed — provenance is undocumented and FR-004
forbids inferring meaning from the column name. The existing smoketest reported
`RMSE 8.90`; ÷10 → 0.89, near the paper's Sydney figures (IDW 0.667, HGNN(G) 0.686 mm/h),
consistent with but not proof of the hypothesis.

**Impact:** gates any cross-implementation comparison. Treating these values as mm/hr
would silently introduce a large and time-varying scientific error.

**Checked (2026-08-30) and still open:** BOM's public rainfall station catalogue (§8.1.2)
is a metadata-only export (name/id/network/coordinates/date range) and carries no
units-of-measure field; the live KiWIS API that might expose per-station parameter units
was unreachable (`DatasourceError`, total outage as of this check, not station-specific).
No external source settled this. Recommendation stands: ask Niramay or the client
directly for `rainfall_mm`'s provenance.

### 8.3 ⚠️ Only 39 of 188 nodes are ever tested

`sampling.py:56-64` reserves `idx[:n_splits]` per cluster as the test rotation; the
remaining stations enter a permanent training pool. Across all 5 folds the union of test
stations is **39 of 188** — 149 stations never appear in any test set. The "5-fold spatial
CV" evaluates ~21% of the network, so the headline RMSE describes a small fixed subset,
not the region.

### 8.4 ⚠️ Missing input file blocks every entry point
`all_stations_rainfall_hourly_combined.csv` is absent (§4). Reconstructible — §9.

### 8.5 ℹ️ The uptime filter is a no-op
All 173 stations have ≥99.8% uptime (min 0.9980, median 0.9996), so
`uptime_threshold: 0.9` removes nothing. Harmless, but it means the filter has never
been exercised and should not be assumed to work.

---

## 9. Data integrity results (FR-102)

Measured over all 173 bundled station files (7,581,552 rows).

| Check | Result |
|---|---|
| Stations (raw files) | 173, all numeric IDs, 0 filename/in-file mismatches |
| Stations after metadata join | **188 metadata rows for 173 stations** ← defect §8.1 |
| Time range | 2021-01-01 00:00:00 → 2025-12-31 23:00:00 (1825 days) |
| Temporal frequency | exactly hourly; 43,824 unique timestamps; **timezone-naive** |
| Duplicate (timestamp, station_id) | **0** |
| Missing values | 3,204 nulls (0.04%) |
| Negative rainfall | **0** |
| Zeros | 7,013,217 (92.50%) |
| Value range | 0.0 – 150.0, quantised to 0.5 |
| Units | **suspect — see §8.2** |
| Coordinate range (matched) | lat −34.775…−33.245, lon 149.916…151.392 (~136 km E-W × 170 km N-S, Greater Sydney) |
| Stations at identical coordinates | 3 exact-zero-distance pairs (hit the `1/d` guard) |
| Split sizes (5 folds, seed 123) | train 149 / val 31 / test 8 per fold, of 188 nodes |
| Test targets per fold | ~8 stations × 43,824 timesteps ≈ 350k records |

The data is otherwise clean: no duplicates, no negatives, near-complete coverage,
consistent hourly cadence. The problems are in the **metadata join** and the **units**,
not the observations.

Every result above is reproduced by the committed test suite (T-001/T-002):

```bash
.venv/bin/python -m pytest tests/ -q -rx      # 18 passed, 7 xfailed
```

Each xfail encodes one known, unfixed defect with a reason pointing back to this
document. They are non-strict, so a genuine fix makes them start passing — that is the
signal to promote them to ordinary tests. `tests/conftest.py` locates the data via
`$SYDNEY_GAUGE_ROOT`, the `../COMP3888-niramay` worktree, or `./database/australia`, and
skips cleanly if none resolve.

---

## 10. Reconstructing the missing input

Verified to produce a file `loader.py` accepts (43,824 × 173 after pivot):

```bash
python3 - <<'EOF'
import pandas as pd, glob
df = pd.concat((pd.read_csv(f) for f in glob.glob("database/australia/raw/*.csv")),
               ignore_index=True)
df.to_csv("all_stations_rainfall_hourly_combined.csv", index=False)
EOF
```

Deterministic apart from row order, which `pivot` makes irrelevant.

---

## 11. Open questions requiring human input

1. **Units of `rainfall_mm`** — needs provenance from Niramay or the client. Blocks §8.2.
2. **Timezone** of `timestamp` — undocumented; matters for any future radar/satellite alignment.
3. **Why 173 stations over 2021–2025?** ✅ Partially answered 2026-08-31 by reading the
   paper itself (`docs/2607.01621v1.md`): its Sydney testbed is **2022 only, 75 gauge
   stations** (its reference [4] for that data is BOM WaterData, "Retrieved February
   2026" — the same live portal §8.1.2 used). This dataset is **2021–2025 (5 years),
   173/188 stations** — not a subset of the paper's testbed, a materially different
   extraction: more stations, 5× the time span, no documented selection rule for either
   difference. **This means any RMSE comparison to the paper's reported numbers
   (§8.0.2) compares different populations over different periods, not the same
   evaluation** — a stronger caveat than the "different station count" framing this item
   previously had.
4. **Was the duplicate-metadata defect present in the reported results?** If Niramay's
   published Sydney numbers came from this code path, they carry the §8.1 bias.
