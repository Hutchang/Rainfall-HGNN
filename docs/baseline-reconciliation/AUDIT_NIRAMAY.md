# AUDIT — Niramay implementation (Australian / Sydney, 0D gauge-only)

**Audited tree:** worktree `../COMP3888-niramay`, branch `niramay-clone` @ `4fcb313`
(`4fcb313` = `8368cbf` + a docs-only README commit; no code differences).
**Audit date:** 2026-08-29 · **Method:** read-only inspection + re-execution of the
repo's own loader/sampler code paths against the bundled data.
**Status:** Phase 0 (FR-001…FR-007) and Phase 1 data validation (FR-101, FR-102) complete.
FR-103 (native run) **not yet attempted** — gated on the blockers in §8.

---

## 1. Provenance and relationship to the source paper

This code corresponds to *"Spatial Support Matters: Geometry-Aware Graph Fusion for
Rainfall Field Reconstruction"* (arXiv:2607.01621), whose author list includes
Niramay Kachhadiya. **It implements only the paper's gauge-only ablation**
(`HGNN (G)`, Table 1), not the multi-support fusion that is the paper's contribution.
There are no 1D (CML) or 2D (radar/satellite) node types anywhere in this tree.

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

> **⚠️ This applies to Jylow too.** `src/graph/gaugegraphnew.py:172-173` uses the same
> `max(edge_attr)` global normalisation. Porting Sydney gauges into Jylow will hit the
> identical wall unless the formula is corrected first. See `SYSTEM_MAP.md` §6.

**Minimal correction:** implement the paper's per-node normalisation. Note that the `1e6`
guard at `gauge_graph.py:133` is a second-order concern — the real problem is that *any*
very small distance dominates a global normaliser.

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
and FR-303 would propagate the contamination into the Jylow port.

### 8.2 🛑 BLOCKER — `rainfall_mm` is almost certainly not millimetres

Median station annual total is **9,321 units/yr**. Sydney climatology is ~1,210 mm/yr;
western-Sydney stations run 700–900 mm/yr. Values are quantised to exactly 0.5 steps,
max 150.0/hr, with ~1% of rainy hours above 99 — impossible as mm/hr.

÷10 gives ~932 mm/yr and a 15 mm/hr max, entirely plausible for this station mix,
pointing to **0.1 mm units**. Not confirmed — provenance is undocumented and FR-004
forbids inferring meaning from the column name. The existing smoketest reported
`RMSE 8.90`; ÷10 → 0.89, near the paper's Sydney figures (IDW 0.667, HGNN(G) 0.686 mm/h),
consistent with but not proof of the hypothesis.

**Impact:** gates any cross-implementation comparison. Feeding these values to Jylow,
which works in mm/hr, would silently introduce a ~10× scale error.

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

Reproduce with the scripts in the scratchpad (`audit_niramay_data.py`,
`audit_niramay_data2.py`, `verify_leakage.py`) — to be promoted into the repo as tests
per T-001/T-002.

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
3. **Why 173 stations?** The paper's Sydney testbed uses 75 gauges. This is a different,
   larger set; the selection rule is undocumented.
4. **Was the duplicate-metadata defect present in the reported results?** If Niramay's
   published Sydney numbers came from this code path, they carry the §8.1 bias.
