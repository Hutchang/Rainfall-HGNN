# HANDOFF — state of play, environment, and the two open work items

Read this first. It records what a fresh session cannot recover from the code alone:
current branch/worktree state, environment setup, known traps, and the exact framing of
the two authorised work items.

Companion documents: `AUDIT_NIRAMAY.md`, `AUDIT_JYLOW.md`, `DATA_INVENTORY.md`,
`SYSTEM_MAP.md`, `run_niramay_smoketest_native.log`.

---

## 1. ⚠️ Authorisation note — read before starting work item A

`COMP3888_Baseline_Reconciliation_Requirements.md` **gates the work described below**:

- §3 orders `AUDIT -> ... -> VALIDATION REPORT -> ONLY THEN REFACTOR`
- NFR-004 prefers adapters over code changes "until validation is complete"
- §17 lists stop conditions including "reproduction would require changing preprocessing"
  and "Do not choose a scientifically consequential workaround silently"

An agent following that document faithfully **will stop and escalate** rather than change
the normalisation — which is the correct behaviour absent an override. If work item A is
to proceed, the operator must state the authorisation explicitly, e.g.:

> You are authorised to implement the paper's Eq. 1 per-node edge-weight normalisation as
> a separate, clearly-labelled experiment alongside the unmodified baseline. This
> supersedes §17 for this specific change only. Do not alter any other scientific
> behaviour.

Work item B (external verification) is pure investigation and needs no override.

---

## 2. Repository state

| Branch | Purpose | Rule |
|---|---|---|
| `main` | team baseline | — |
| `niramay-clone` | imported researcher branch | **read-only** (BR-001) |
| `jylow-clone` | imported researcher branch | **read-only** (BR-001) |
| `integration/sydney-gauge-baseline` | **the working branch**, off `jylow-clone` | all integration work |
| `repro/niramay-native` | off `niramay-clone`, carries one compat fix | Niramay reproduction only |

**Worktrees** (BR-003 — both trees checked out simultaneously):

```
<repo>/                      integration/sydney-gauge-baseline   (Jylow code)
../COMP3888-niramay/         repro/niramay-native                (Niramay code + data)
```

Recreate if missing:
```bash
git worktree add ../COMP3888-niramay repro/niramay-native
```

**Commits so far on `integration/sydney-gauge-baseline`:**
- `c9843af` Phase 0 audit documentation
- `a708cd1` FR-103/FR-104 reproduction result + root-cause finding
- plus the run log and the `tests/` suite

**On `repro/niramay-native`:**
- `c37c83b` `fix(compat)`: adds the missing `PerformanceLogger.log_stratum_metrics`.
  **Required** — without it `train.py` raises `AttributeError` at `trainer.py:248` right
  after printing test metrics, so no run can complete. Purely additive; no science changed.

---

## 3. Environment

There is **one** usable interpreter — the venv committed under the Jylow tree. It serves
both worktrees.

```bash
PY=<repo>/.venv/bin/python          # torch 2.13.0, PyG 2.8.0.post1,
                                    # pandas 3.0.5, numpy 2.5.2, sklearn 1.9.0, pytest 9.1.1
```

**Caveats:**
- Niramay's `requirements.txt` asks for `pandas>=2.0`; this venv has **3.0.5**, a major
  version ahead. Recorded as a reproduction deviation. Nothing has broken so far.
- The venv is **missing** `geopandas`, `shapely`, `rasterio`, `xarray`, `netCDF4`, so
  **no Jylow entry point can be imported** (`src/utils.py:7-8` imports xarray+rasterio at
  module scope). Install those before any Jylow execution work.
- Jylow's `requirements.txt` is **not installable** — a conda `pip freeze` with ~40
  `file:///` local build artefacts.

**Niramay's input file is not committed** (235 MB, gitignored). Regenerate:

```bash
cd ../COMP3888-niramay
$PY - <<'EOF'
import pandas as pd, glob
df = pd.concat((pd.read_csv(f) for f in glob.glob("database/australia/raw/*.csv")),
               ignore_index=True)
df.to_csv("all_stations_rainfall_hourly_combined.csv", index=False)
EOF
# expect 7,581,552 rows, 235.2 MB
# sha256 93e692ca97c15c98f6c21abeada4f0eb7100d8c54d3d03aba0b3cece9941c097
```

---

## 4. 🪤 Traps that cost time

1. **`batch_size=1024` is silently OOM-killed.** `training/trainer.py:59-76` accumulates
   `batch_loss` across all ~149 node positions and calls `backward()` **once, after the
   loop**, retaining 149 autograd graphs (~11.6 GB). On an 8.6 GB host the process dies
   with **no traceback**, and a naive `$?` capture reports success. **Always pass
   `--batch_size 128`** (or lower) locally. Jylow fixed this — `logic_hetero.py:72-89`
   calls `.backward()` inside the loop.
2. **`train.py` cannot complete without `c37c83b`** (see §2).
3. **The bundled `experiments/smoketest/` artifacts are not reproducible from this
   branch.** Their `training_log.jsonl` has zero `stratum_metrics` records, so they
   predate the feature `cc23a25` introduced. Do not treat `RMSE 8.9024` as an inherited
   reference result.
4. **CWD must be the repo root** for any Jylow entry point — three modules read a bare
   `"config.yaml"` at import time.
5. **Five Jylow scripts train on import** (no `__main__` guard), so
   `import train_fused` starts a full 5-fold run.
6. **`paramtuning_gauge.py` is misnamed** — it never sets `datasources`, so it runs
   gauge+CML, not gauge-only.

---

## 5. Reproducing the native Niramay run

```bash
cd ../COMP3888-niramay
$PY train.py --hidden 32 --layers 4 --knn 5 --lr 0.001 \
    --epochs 20 --patience 10 --folds 1 --fraction 0.05 \
    --batch_size 128 --include_lpe --name smoketest_native
```

Reference result (full log: `run_niramay_smoketest_native.log`), ~1,164 s on CPU:

```
Test RMSE 8.9645   Pearson r -0.0131      early stop at epoch 11
dry 0.2574/+0.2574   light 0.2697/-0.2520
moderate 5.5485/-5.0964   heavy 39.2091/-31.3333
```

**This number is not a usable baseline.** Best val was epoch 1, so the scored checkpoint
had one epoch of training, and the result sits inside the constant-predictor band
`[8.801, 9.017]` with r ≈ 0. See `AUDIT_NIRAMAY.md` §8.0 for why.

## 6. Test suite

```bash
.venv/bin/python -m pytest tests/ -q -rx      # 18 passed, 7 xfailed
```

`tests/conftest.py` finds the data via `$SYDNEY_GAUGE_ROOT`, the `../COMP3888-niramay`
worktree, or `./database/australia`, and skips cleanly otherwise. Each **xfail encodes one
known unfixed defect** with a reason pointing into the audit; they are non-strict, so a
real fix makes them pass. `test_per_node_normalisation_would_fix_it` is a **positive
control that already passes** — it demonstrates the target behaviour for work item A.

---

## 7. Work item A — edge-weight normalisation

**Problem** (`AUDIT_NIRAMAY.md` §8.0): stations `212285` and `563086` sit 0.05 m apart,
giving `1/d = 2,219,128` against a median of 13.7 and a next-closest pair of 194.
`graph/gauge_graph.py:172` divides every edge by the **global** max, so 99.8% of edges
fall below 1e-3 and `GraphConv`'s neighbour term sits ~5 orders of magnitude beneath its
self term. The GNN degenerates into a per-node MLP.

**Fix**: the paper's per-node normalisation (arXiv:2607.01621 §3.2 Eq. 1):

```
w_ij = (1/d_ij) / max_{k in N(j)} (1/d_kj)
```

Each node normalises by its own neighbourhood maximum, so one pathological pair cannot
crush the graph.

**Touch points:**
- `../COMP3888-niramay/graph/gauge_graph.py:121-135` (`_build_knn_edges`) and `:168-173`
  (the `_train_max_edge_weight` division)
- **Jylow shares the same defect** at `src/graph/gaugegraphnew.py:172-173`, and again at
  `:313-314` and `:341-342` for the CML and radar relations

**Constraints:** keep the unmodified baseline runnable and label the corrected run as a
separate experiment (FR-420 — report first, correct separately). Do not change any other
scientific behaviour in the same commit (BR-004).

**Done when:** `test_normalised_edge_weights_permit_message_passing` passes, and a
re-run produces a Pearson r meaningfully above zero and an RMSE outside `[8.801, 9.017]`.

---

## 8. Work item B — resolve the station-identity questions

Three distinct sub-problems. **Only two need external data.**

### B1 — 13 dual-network registrations (no external data needed; already settled)
Thirteen ids appear twice within 541 m with matching or near-matching names, and their
series are **bit-identical** (`.equals()` True, r = 1.000000). Mechanically
de-duplicable. Full table: `AUDIT_NIRAMAY.md` §8.1.1.

### B2 — 2 ID collisions (needs BOM)
| id | row A | row B | separation |
|---|---|---|---|
| `563083` | BLACKHEATH G.C. | NORTH RICHMOND WFP | **39.4 km** |
| `567163` | REGENTVILLE | REGENTVILLE RFS | 1.24 km |

Genuinely different sites sharing one id. Because `GaugeGraph` selects the series by id,
station `563083`'s rainfall is currently placed at **both** Blackheath and North Richmond.
`drop_duplicates()` would pick a location by coin flip. **Determine which site the id
belongs to.**

### B3 — the Greaves Creek coordinate collision (needs BOM)
`212285 GREAVES CREEK DAM` and `563086 Greaves Creek`, nominally 0.05 m apart, are
**not** the same series: r = 0.9247, differing on 7.33% of rows, annual totals 7,497 vs
10,895 (+45%). Two real instruments, at least one carrying wrong coordinates. **Cannot be
de-duplicated** — that would discard real observations. **Determine the true coordinates.**

### Suggested sources
- **BOM WaterData** — `http://www.bom.gov.au/waterdata/`, the authority the paper itself
  cites as its Sydney gauge reference [4]. Note it is a JavaScript portal; a plain fetch
  may not work. It exposes a Kisters KiWIS-style service endpoint — **verify before
  relying on it**, and do not bulk-download (FR-602).
- The `network` column distinguishes **Water NSW** from **Sydney Water Corporation**,
  which is the likely origin of the dual registrations.
- `AUDIT_NIRAMAY.md` §11 also has the open **units** question (§8.2), which BOM metadata
  may incidentally settle — the single highest-value unknown in the project.

**Do not silently pick a coordinate.** Per FR-102 and §17, report findings and let a human
choose if the external evidence is inconclusive.

---

## 9. What is deliberately NOT done

- No adapter (FR-301) — blocked on the units question
- No Jylow 0D run on real Sydney data — FR-204 was proven on synthetic data only
  (`AUDIT_JYLOW.md` §2); the 5-item driver change set is specified but unimplemented
- No golden fixture (FR-401), no common evaluator (FR-501/503), no
  `data_manifest.yaml`, no `BASELINE_RECONCILIATION_REPORT.md`
- **Neither implementation persists row-level predictions**, so FR-503 needs building from
  scratch on both sides. Both also label predictions with a positional index rather than a
  station id (`trainer.py:218`; Jylow's `test_model` accepts a `mapping_df` and never uses
  it), so joining back to metadata means re-deriving node ordering.
