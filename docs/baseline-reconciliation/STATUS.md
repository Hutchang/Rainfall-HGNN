# STATUS — baseline reconciliation

**As at:** 2026-08-30
**Branch:** `integration/sydney-gauge-baseline` @ `d3bfb0d`
**Worktrees:** both clean, nothing uncommitted

Point-in-time status brief. The supporting evidence for every claim below lives in
`AUDIT_NIRAMAY.md`, `AUDIT_JYLOW.md`, `DATA_INVENTORY.md`, `SYSTEM_MAP.md` and
`HANDOFF.md`; this document is the summary view and the single place the **open
decisions** are collected.

---

## Scope you set: Phase 0 + native reproduction — both done

**Phase 0 (audit)** — five documents in `docs/baseline-reconciliation/`, covering FR-001…007 for both trees plus FR-101/102 data validation and the FR-604 2D gap list.

**Phase 1 (native reproduction)** — Niramay runs, result recorded, root-caused.

**FR-201 answered, and it's the good news:** Jylow *can* run genuine 0D gauge-only with no fake sensor data. Verified by executing the real graph→dataset→model→train path with no CML and no radar. The library layer is already support-agnostic; only the top-level drivers load radar/CML unconditionally. Five guarded edits, no model mathematics touched, 0D/1D/2D preserved.

## Three blockers, in priority order

**1. The GNN isn't using its graph.** Stations `212285` / `563086` sit 0.05 m apart → `1/d = 2,219,128` against a median of 13.7. `gauge_graph.py:172` divides by the *global* max, so 99.8% of edges fall below 1e-3 and the neighbour term sits ~5 orders under the self term. It degenerates into a per-node MLP.

Proof: the reproduction scored **RMSE 8.9645, Pearson r −0.0131**, and the constant-predictor band is **[8.801, 9.017]**. The bundled artifacts scored 8.9024 — also inside it. That's why the two agree to 0.7% despite different code, batch size and pandas: *the number is a property of the data, not the model.* The paper's per-node Eq. 1 is immune; **Jylow shares the same defect.**

**2. Units are unresolved.** Median station annual total is 9,321 units/yr against a Sydney climatology of ~1,210. Likely 0.1 mm. Jylow's loader independently applies `×12`, so naive reuse could stack to ~120×.

**3. Station identity** — now three separate problems, only two needing BOM:

| | Needs external data? |
|---|---|
| 13 dual-network registrations (bit-identical, ≤541 m) | No — settled |
| 2 ID collisions (`563083` at 39.4 km, `567163` at 1.24 km) | Yes |
| Greaves Creek (r=0.92, +45% annual totals) | Yes — coordinates, not identity |

## Committed

`integration/sydney-gauge-baseline` — 5 commits: audit docs, reproduction result + root cause, run log, tests + handoff, and your requirements doc.

`niramay-clone` — `c37c83b`, one isolated compat fix (`train.py` couldn't complete a single fold without it). The inherited state is preserved by the tag `niramay-inherited` (`8368cbf`), so `git diff niramay-inherited..niramay-clone` still gives an exact statement of what the team changed.

Tests: `.venv/bin/python -m pytest tests/ -q` → **18 passed, 7 xfailed**, each xfail encoding one known defect. `test_per_node_normalisation_would_fix_it` is a positive control that already passes — a green target for the fix.

## Open — waiting on you

Three decisions, all scientific rather than compatibility, which is why I stopped:

1. **Authorise the normalisation fix?** `HANDOFF.md` §1 flags that your requirements doc gates this via §3/NFR-004/§17, so any agent following it will escalate rather than act. Suggested override wording is in the doc.
2. **De-dup lane** — and note it's now more nuanced: `563083`/`567163` can't be de-duplicated safely without knowing which site owns the ID.
3. **Units provenance** — worth asking Niramay or the client directly.

Nothing downstream (adapter, Jylow 0D run on real data, golden fixture, common evaluator) is started — all of it is blocked behind #1 and #2.

---

## Commit ledger

`integration/sydney-gauge-baseline` (branched from `jylow-clone` @ `1c2489a`):

| Commit | Contents |
|---|---|
| `c9843af` | Phase 0 audit documentation (4 docs) |
| `a708cd1` | FR-103/FR-104 reproduction result + root-cause finding |
| `e85a689` | Native run log (force-added past the inherited `*.log` ignore) |
| `54656b5` | `tests/` suite + `HANDOFF.md`; station-identity findings |
| `d3bfb0d` | Requirements document |

`niramay-clone`:

| Ref | Contents |
|---|---|
| tag `niramay-inherited` → `8368cbf` | the researcher's code **as inherited**, before any team change |
| `4fcb313` | README (team) |
| `c37c83b` | `fix(compat)`: missing `PerformanceLogger.log_stratum_metrics` (team) |

`git diff niramay-inherited..niramay-clone` is the exact statement of what the team
changed — the attribution guarantee BR-001 was protecting, now carried by the tag rather
than by freezing the branch.

`jylow-clone` @ `1c2489a` is unmodified.
