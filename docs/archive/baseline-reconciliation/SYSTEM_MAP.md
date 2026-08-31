# SYSTEM MAP — Niramay ↔ Jylow

Side-by-side comparison of the two inherited implementations, and what the
`Jylow framework + Niramay Sydney gauge data + 0D-only` hypothesis actually requires.

**Companion documents:** `AUDIT_NIRAMAY.md`, `AUDIT_JYLOW.md`, `DATA_INVENTORY.md`.

---

## 1. At a glance

| | **Niramay** (`niramay-clone` @ `4fcb313`) | **Jylow** (`integration/…` @ `1c2489a`) |
|---|---|---|
| Region | Greater Sydney | Singapore |
| Supports | **0D only** | 0D + 1D (CML) + 2D (radar). **No satellite type.** |
| Scale | 8 modules, 1 live model | 78 `.py` + 10 notebooks, 5 model variants, 4 trainers |
| Data bundled | **Yes** — 173 station CSVs + metadata | **None.** `database/` gitignored and absent |
| Runnable today | **Yes**, after regenerating one CSV | **No** — see §5 |
| Native run docs | PBS scripts + README (added by us) | one-line README stub |
| Canonical trainer | `train.py` | `train_fused.py` |
| Canonical model | `GNNInductiveHetero` (`models/gnn.py:233`) | `GNNInductiveHetero` (`models/gnn.py:275`) — **same name, different class** |
| CLI | full argparse | **none** — no script takes any argument |
| `__main__` guard | yes | **no** — five scripts train on import |
| Level A validation | possible | **impossible** — no Singapore data |

> Both trees descend from the same paper (arXiv:2607.01621; Niramay Kachhadiya and
> Low Jun Yu are co-authors). Niramay's tree implements only the paper's gauge-only
> ablation `HGNN (G)`; Jylow's implements the full framework.

---

## 2. Call chain, side by side

```
NIRAMAY                                    JYLOW
train.py:main()                            train_fused.py:293  (runs at IMPORT)
 └ config.yaml                              └ config.yaml  (+2 more disk reads at import)
 └ load_raingauge_dataset()                 └ load_raingauge_dataset()   ← different signature
     loader.py:29  read_csv                     src/raingauge/utils.py:22  per-year CSVs
     loader.py:32  pivot                        :30  strptime "…+08:00"   ← SGT hardcoded
     loader.py:36  uptime filter                :33  value * 12           ← 5-min → mm/hr
     loader.py:53  .isin() join  ← DEFECT       :41  filter_uptime
                                            └ radar .pkl + CML .nc loaded UNCONDITIONALLY
                                                :55, :62, :64, :67  inner joins define timestamps
 └ stratified_spatial_kfold()               └ stratified_spatial_kfold_dual()
     KMeans k=8, seed from config               KMeans k=8, seed=123 HARDCODED
 └ GaugeGraph()                             └ GaugeGraphNew()  (+ add_heterodata per support)
     LPE once on full graph  ← FIXED            LPE per split      ← NOT fixed
     edge w: global max norm                    edge w: global max norm (per split)
 └ GNNInductiveHetero                       └ GNNInductiveHetero
     HeteroConv(aggr='sum'), no dropout         HeteroConv(aggr='mean'), dropout 0.15
 └ trainer.py train/validate/test           └ logic_hetero.py train/validate/test
     unweighted MSE, raw mm                     weighted MSE α=0.18, raw mm/hr
```

**Structurally the two are close.** Both do spatial-only K-fold, both build a kNN gauge
graph with inverse-distance weights, both use leave-one-out masking of features `[:2]`,
both predict raw rainfall and score in raw units. That similarity is what makes the port
plausible. The differences are in preprocessing contracts and a handful of scientific
details (§4).

---

## 3. The data contract gap (feeds FR-301)

This is the core of the adapter. Niramay's data cannot be handed to Jylow's loader as-is.

| Aspect | Niramay produces | Jylow's loader expects | Adapter action |
|---|---|---|---|
| File layout | one combined CSV (`timestamp, station_id, rainfall_mm`), long format | `database/raingauge_nea_data/{year}/weather_station_data_{year}.csv`, per year | reshape + relocate, or bypass the loader |
| Timestamp | `2021-01-01 00:00:00`, **tz-naive** | `strptime("%Y-%m-%dT%H:%M:00+08:00")` — **SGT baked in** (`src/raingauge/utils.py:30-32`) | **must** re-format or patch; raises `ValueError` otherwise |
| Native interval | **hourly** | **5-minute**, converted by `value * 12` (`:33`) | ⚠️ blindly reusing gives a **12× error** |
| Units | `rainfall_mm` — **suspect, likely 0.1 mm** (`AUDIT_NIRAMAY.md` §8.2) | mm/hr after `×12` | **UNRESOLVED — blocks the port** |
| Resampling | none (already hourly) | `.resample('15min').mean()` (`train_fused.py:52-53`) | upsampling hourly → 15-min yields NaN/ffill |
| Station ID | numeric strings, e.g. `211010` | Singapore `S##` codes | rename map; IDs are opaque to the code |
| Metadata | headerless, `(name, id, network, lat, lon)`, **425 rows / 409 unique** | `weather_stations_{year}.csv` | **must de-duplicate** — see §6 |
| Coordinates | file order (lat, lon); code reads (lon, lat) — handled correctly | (lon, lat) | none, but assert it |
| Missing values | NaN, 0.04% | NaN → `fillna(0)` | same convention; document it |
| Station count | 173 real (188 rows) | 51 Singapore stations | `n_clusters=8` is tuned for 51 — revisit |

**Three of these (timestamp format, ×12 conversion, 15-min resampling) are hard failures
on contact.** They must be handled in an adapter, not by editing the Singapore loader
in place (NFR-004, FR-203).

---

## 4. Scientific protocol differences (must be recorded before any comparison)

| Dimension | Niramay | Jylow | Comparable? |
|---|---|---|---|
| Split type | spatial only | spatial only | ✅ same premise |
| Temporal holdout | **none** | **none** | ✅ same (both transductive over time) |
| Fold construction | KMeans k=8, one test station/cluster/fold | identical algorithm | ✅ |
| Split seed | `config.yaml` `split_seed: 123` | **hardcoded** `123` | ✅ same value, different plumbing |
| Test coverage | **39 of 188 nodes ever tested** | same rotation scheme → same limitation | ⚠️ both evaluate a subset |
| Input transform | `log1p` | **z-score**, train-split stats | ❌ **different** |
| Target transform | none (raw) | none (raw) | ✅ |
| Mask semantics | zero on `log1p` → decodes to **0 mm** (correct) | zero on z-score → decodes to **training-mean rainfall** (wrong) | ❌ **Jylow's mask is not "missing"** |
| Validity flag | preserved | **destroyed by normalisation** (std→clamp floor) | ❌ Jylow loses the sentinel |
| LPE | computed **once on full graph**, subset per split | computed **per split** → incomparable train vs test | ❌ **Niramay is correct here** |
| Edge weight norm | global max (train graph) | global max (per split) | ⚠️ both diverge from the paper's per-node formula |
| Distance metric | Euclidean on degrees | Euclidean on degrees (gauge–gauge) | ✅ same (both anisotropic) |
| Conv aggregation | `sum` | `mean` | ❌ different |
| Dropout | none | 0.15 | ❌ different |
| Loss | unweighted MSE | weighted MSE α=0.18 | ❌ different |
| Layers / lr / patience | 4 / 1e-3 / 20 | 5 / 3e-4 / 10 | ❌ different |
| Metrics | RMSE, Pearson r, **stratified RMSE/bias** | RMSE, Pearson r, MAE, P/R/F1@0.5, timestep-RMSE | ⚠️ overlapping, not identical |
| Seeding | split only | split only | ❌ **neither is reproducible** |

**Implication for FR-501/FR-505.** A like-for-like Level B comparison requires either
(a) aligning these knobs explicitly and calling it a *controlled re-evaluation*, or
(b) leaving them native and attributing any delta to a confounded mix. The requirements
document's own FR-303 anticipates this: without a reconstructible identical split, we
must **not** claim reproduction.

---

## 5. Why Jylow cannot run today, and what the 0D path avoids

| Blocker | Bites a 0D-only run? |
|---|---|
| No Singapore data (`database/` absent) | **No** — we supply Sydney gauges |
| `eda.ipynb` is the only writer of `processed_radar_dataset.pkl` | **No** — bypassed by removing the radar load |
| `requirements.txt` uninstallable (conda `pip freeze` with `file://` paths) | **Yes** — must build an env from scratch |
| `.venv` missing `geopandas`, `shapely`, `rasterio`, `xarray`, `netCDF4` | **Yes** — `src/utils.py:7-8` imports xarray+rasterio at module scope, so *every* trainer fails at import. Verified. |
| Radar/CML loaded unconditionally before the `datasources` guards | **Yes** — the §2 change set in `AUDIT_JYLOW.md` |
| SGT timestamp format, `×12` conversion | **Yes** — adapter work |
| Five scripts train on import | **Yes** — hazard when importing anything |

**The good news:** the 0D path sidesteps the two worst structural blockers (missing radar
data and the notebook-only preprocessing step). `geopandas`/`shapely` are needed only by
the CML branch and `xarray`/`rasterio` only by import-time statements in `src/utils.py`, so
a genuine gauge-only environment is small.

---

## 6. Recommended port sequence

Ordered so each step is independently verifiable, per BR-004.

0. **Fix the edge-weight normalisation** (`AUDIT_NIRAMAY.md` §8.0). **Do this first.**
   Both trees divide edge weights by a *global* maximum; a single 0.1 m station pair in
   the Sydney data drives 99.8% of edges to <1e-3, reducing the GNN to a per-node MLP.
   Verified: the native run's RMSE sits inside the constant-predictor band. The paper's
   per-node formula (Eq. 1) is immune. **Until this is fixed, neither implementation can
   produce a meaningful Sydney number, so every later comparison is moot.**
1. **Resolve the units question** (`AUDIT_NIRAMAY.md` §8.2). Everything downstream is
   scale-dependent. Needs human input — see §7.
2. **Decide the de-duplication lane** (`AUDIT_NIRAMAY.md` §8.1). Affects what the
   reference result *means*. Note it does **not** fix step 0 — the offending pair has two
   different station IDs.
3. **Build a working Jylow env** — install the four missing packages, or trim
   `src/utils.py`'s import-time dependencies for a 0D-only variant.
4. **Apply the 5-item `datasources` change set** (`AUDIT_JYLOW.md` §2). Keeps 0D/1D/2D
   intact.
5. **Write the Sydney adapter** — timestamp format, no `×12`, hourly-native resampling,
   metadata de-duplication. New module; do **not** edit the Singapore loader.
6. **Cut the golden fixture** (FR-401/402) from `cc23a25`'s bundled data.
7. **Run Jylow 0D on Sydney**, then the common evaluator (FR-501/503).

---

## 7. Decisions required before step 1

1. **Units of `rainfall_mm`** — provenance needed from Niramay or the client. Blocks any
   cross-implementation number.
2. **De-duplication lane** — reproduce-as-is, reproduce-both, or fix-first.
3. **Which split defines "the same targets"** (FR-303)? Niramay's split is computed over
   **188 inflated nodes**; a de-duplicated split has 173. These are *different splits*, so
   "identical held-out targets" is only achievable after decision 2.
4. **Timezone** of Niramay's timestamps — undocumented; needed for any later 2D alignment.

---

## 8. Standing risk

**Jylow's Singapore data is gone, so the framework can never be validated against a
known-good result before we port data into it.** If a Jylow-on-Sydney run produces an
odd number, we will not be able to separate "the port is wrong" from "the framework was
always like this". This is the single largest threat to the milestone's conclusions and
should be recorded in `BASELINE_RECONCILIATION_REPORT.md` regardless of outcome.

Partial mitigations: the executed 0D probe (`AUDIT_JYLOW.md` §2) shows the machinery is
self-consistent; and the golden fixture (FR-401) gives a deterministic regression target
even without a scientific reference.
