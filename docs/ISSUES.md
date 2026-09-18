# Known issues

Outstanding bugs, gaps and unresolved decisions. An entry here is a **recorded, deliberate
choice**: either it is not worth fixing yet given time and materiality, or it is blocked on
someone else's answer. Anything not written down here is not deferred — it is forgotten.

[IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md) stays clean and describes only the
current state of the sprint. Issues live in this doc.

## Conventions

- **ID** is permanent. Reference it in branch names, commit messages and PR descriptions
  (`fix(ingestion): resolve ISS-11`). Never renumber; close in place.
- **Severity** is about consequence, not effort:
  - `Blocking` — a result computed today would be wrong, unpublishable, or misleading.
  - `Material` — real but survivable; it will bite before the project ends.
  - `Immaterial` — known, understood, knowingly ignored. Revisit only if everything else is done.
- **Owner** is a person or a component. `_unassigned_` means the issue is real and nobody
  has picked it up — it is a queue, not a resolution.
- **Status** is `Open`, `Blocked` (waiting on an external answer), or `Closed` with the
  commit that closed it.
- Adding an issue is not a substitute for a fix when the fix is cheap. Defer on purpose.

## Index

| ID | Title | Severity | Owner | Status |
|---|---|---|---|---|
| ISS-01 | Short gauge blockages produce confident false zeros | Blocking | Component 02 | Open |
| ISS-02 | Radar and satellite provenance unknown | Blocking | Component 01 | Blocked |
| ISS-03 | Satellite node coordinates do not exist | Blocking | Component 01 | Blocked |
| ISS-04 | Radar and satellite cover 2022 only; radar missing 120 hours | Material | Component 01 | Blocked |
| ISS-05 | Within-day timestamp fidelity is assumed, not verified | Material | Component 02 | Blocked |
| ISS-06 | Unexplained April–May 2021 gap in BoM daily totals | Material | Component 02 | Open |
| ISS-07 | Donor policy differs between Components 04 and 06 | Blocking | Kiran, Angus | Open |
| ISS-08 | IDW exponent is unprescribed | Material | _unassigned_ | Open |
| ISS-09 | No reversible normalization metadata in the graph contract | Material | Components 04 + 06 | Open |
| ISS-10 | Spatial-block cross-validation decision unresolved | Material | _unassigned_ | Open |
| ISS-11 | Ingestion config claims Sydney but fetches Australia-wide | Blocking | Component 01 | Open |
| ISS-12 | Plausibility caps, tolerance and coverage threshold need a client decision | Material | _unassigned_ | Blocked |
| ISS-13 | `train_homo.py` and `train_st.py` cannot import | Immaterial | _unassigned_ | Open |
| ISS-14 | Inherited-baseline graph fingerprint test missing | Material | Component 04 | Open |
| ISS-15 | No `.gitattributes`; NCI guard tests fail on Windows checkouts | Material | _unassigned_ | Open |
| ISS-16 | Failed raw downloads are excluded but not reported | Material | Component 02 | Open |
| ISS-17 | `ruff check .` reports 692 findings in inherited files | Immaterial | _unassigned_ | Open |
| ISS-18 | Component 03 split summary and plot are not produced | Material | Component 03 | Open |
| ISS-19 | Component 03 has no component README | Material | Component 03 | Open |
| ISS-20 | Canonical README's Sydney statistics table is stale | Material | Component 02 | Open |

---

## Scientific correctness

### ISS-01 — Short gauge blockages produce confident false zeros
**Severity:** Blocking · **Owner:** Component 02 · **Status:** Open

A blocked gauge reports no tips, and BoM's daily total is computed from those same tips, so
it also reports 0.0. The day therefore *verifies*, and 24 hours of `0.0` are written as
observed-dry rather than missing. The dead-gauge rule only catches runs of 90+ days.

This is the top risk in the current standard tables: it is silent, it looks like good data,
and it biases every method trained on it toward dry.

**Closes when:** a day-level check exists — verified-dry while most near neighbours recorded
rain — and the neighbour count, radius and rain threshold are agreed. The threshold is a
scientific choice; see ISS-12.
Source: [canonical/README.md](canonical/README.md).

### ISS-02 — Radar and satellite provenance unknown
**Severity:** Blocking · **Owner:** Component 01 · **Status:** Blocked

Which product, which units, and whether the radar is gauge-corrected are all unknown. If it
was corrected using gauges that include our test stations, every radar result leaks and is
not publishable.

**Closes when:** whoever produced the files states the product, units and correction method
in writing, and it is recorded in the Component 01 contract.

### ISS-03 — Satellite node coordinates do not exist
**Severity:** Blocking · **Owner:** Component 01 · **Status:** Blocked

726 satellite nodes have no latitude or longitude, so they cannot be placed on the map or
connected to gauges. Satellite fusion cannot run at all.

**Closes when:** the coordinate file is found or regenerated.

### ISS-04 — Radar and satellite cover 2022 only; radar missing 120 hours
**Severity:** Material · **Owner:** Component 01 · **Status:** Blocked

Gauges cover 2021–2022; radar and satellite cover 2022, and radar is missing 120 hours. The
graph builder fails on missing radar values rather than silently filling them, which is
correct, but it means fusion experiments are implicitly 2022-only.

**Closes when:** the team confirms in writing that fusion experiments are 2022-only, and the
missing 120 hours are either recovered or excluded by an agreed rule.

### ISS-05 — Within-day timestamp fidelity is assumed, not verified
**Severity:** Material · **Owner:** Component 02 · **Status:** Blocked

If any logger batch-reports its tips, the daily total is still correct but the hourly
distribution is wrong. Every hourly model would train on misplaced rain.

**Closes when:** BoM confirms reporting behaviour, or a detection heuristic is agreed.

### ISS-06 — Unexplained April–May 2021 gap in BoM daily totals
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

A network-wide gap of roughly three weeks across roughly 30 stations. A per-owner outage is
the working hypothesis. Until the cause is known it is unclear whether those days are missing
or genuinely unrecorded.

**Closes when:** the cause is identified, or the period is excluded by an agreed rule.

---

## Decisions needed

### ISS-07 — Donor policy differs between Components 04 and 06
**Severity:** Blocking · **Owner:** Kiran, Angus · **Status:** Open

Component 04 permits train **and validation** gauges as visible HGNN context at test time.
Component 06 specifies **training gauges only** for IDW and network mean. The implementation
honours both written contracts and flags the mismatch in every report.

Shared prediction targets alone do not make this a controlled comparison. Any claim that the
HGNN beats IDW is currently confounded by the two methods seeing different inputs.

**Closes when:** one donor policy is agreed and both contracts are edited to match.
Source: [models/README.md](models/README.md).

### ISS-08 — IDW exponent is unprescribed
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

`config/models/hgnn.yaml` sets `idw_power: 2.0`. That is a proposed setting, not a decision,
and not a claim to reproduce the inherited paper's IDW experiment.

**Closes when:** the exponent is agreed with the Component 03 reviewer and recorded in the
Component 06 contract, or a sensitivity sweep is run and reported.

### ISS-09 — No reversible normalization metadata in the graph contract
**Severity:** Material · **Owner:** Components 04 + 06 · **Status:** Open

The published graph shape carries physical rainfall in mm and binary validity, but nothing
that describes a reversible transform. Component 06 accepts `input_rainfall_scale:
physical_mm` and performs no private normalization; pre-normalized graphs fail validation.
This is safe but restrictive — no one can introduce normalization without a contract change.

**Closes when:** a normalization metadata field is agreed between 04 and 06, or the team
records that physical mm is permanent.

### ISS-10 — Spatial-block cross-validation decision unresolved
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

[spatial-cv-and-regional-partitioning.md](spatial-cv-and-regional-partitioning.md) works
through where spatial leakage actually lives in the current k-fold split and proposes a fix,
but the decision was never taken. The design discussion itself belongs in the archive once
the decision is recorded here.

**Closes when:** the team decides whether folds are spatially blocked, and the Component 03
contract states it.

### ISS-11 — Ingestion config claims Sydney but fetches Australia-wide
**Severity:** Blocking · **Owner:** Component 01 · **Status:** Open

[config/ingestion/sydney.yaml](../config/ingestion/sydney.yaml) has `region: sydney` and a
filename to match, but its bounds are `-44.0..-9.0` latitude and `112.0..154.0` longitude —
the whole continent. The surrounding comments still describe a hand-drawn Sydney basin box
from the Blue Mountains foothills to the coast, and still claim the Upper Blue Mountains are
excluded at `lon < 150.4`. [ingestion/README.md](ingestion/README.md) still estimates about
4,800 requests and roughly 90 minutes.

Running the documented fetch command today does not perform the documented Sydney job, and
the operator has no way to tell from the file that this changed.

**Closes when:** the scope expansion is either reverted, or given its own region identity and
config filename with corrected comments and operational estimates.

Note: the Sprint 2 experiment runner removes the "which config produced this result?"
problem by copying the config into each experiment directory. It does not remove this one,
which is a naming and expectations mismatch inside a single file.

### ISS-12 — Plausibility caps, tolerance and coverage threshold need a client decision
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Blocked

The verification tolerance is one tip increment, tight enough that 282 days fail on a single
tip's difference. The coverage threshold currently drops Gladesville at 89.85%. All of these
are config values, and all of them change which stations enter the study.

**Closes when:** the client states what they want, or the team records the defaults as its
own decision with the reasoning.

---

## Code and test gaps

### ISS-13 — `train_homo.py` and `train_st.py` cannot import
**Severity:** Immaterial · **Owner:** _unassigned_ · **Status:** Open

Both fail at import: they need `src.graph.cmlgraph` and `src.graph.radargraph`, neither of
which exists. Confirmed — `python -c "import train_homo"` raises
`ModuleNotFoundError: No module named 'src.graph.cmlgraph'`.

`cmlgraph` is commercial microwave link support, which is **permanently out of scope**, so
these two entry points cannot be restored as written. `train_fused.py` is unaffected; it
imports only `gaugegraphnew`.

**Immaterial because:** nothing in the production path calls them, and the replacement
(`src/models/run.py`) works. They are misleading rather than harmful.
**Closes when:** both files are deleted, or moved to the archive with a note that they are
non-runnable inherited reference.

### ISS-14 — Inherited-baseline graph fingerprint test missing
**Severity:** Material · **Owner:** Component 04 · **Status:** Open

The Component 04 contract requires a regression test pinning node counts, edge counts and
weights against a known `train_fused.py` run, so that a change to shared graph code cannot
move the inherited baseline unnoticed. The test that previously sat here only asserted
`median(weights) < 1e-3` — which the directed rebuild would still satisfy while changing the
edge count from 598 to 940 — so it gave no real protection and was removed on 2026-09-07.

Nothing replaced it. Shared graph code is currently unprotected against silent baseline drift.

**Closes when:** the fingerprint test exists and pins counts and weights.

### ISS-15 — No `.gitattributes`; NCI guard tests fail on Windows checkouts
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

The repository has no `.gitattributes`, so Git converts `scripts/nci/*.sh` and `*.pbs` to
CRLF on Windows. Bash then reads `set -o pipefail<CR>` and aborts before any assertion:

```
scripts/nci/test_guards.py: 8 failed
set: pipefail: invalid option name
```

All 8 guards pass on Linux and macOS. The main `tests/` suite excludes them, so this does not
appear in the headline 133 passed / 29 skipped figure — which means a Windows teammate sees a
green suite and a broken NCI path.

**Closes when:** a `.gitattributes` pins `*.sh` and `*.pbs` to LF and the guards pass on a
fresh Windows clone.

### ISS-16 — Failed raw downloads are excluded but not reported
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

Failed or unread raw downloads are correctly excluded from `stations.csv` and `days.csv`, but
nothing says so. A station whose coverage looks low cannot be told apart from a genuinely dry
one without manually cross-checking `index.csv`.

No wrong data results — this is an audit-trail gap against shared rule 7, "say what you
dropped" — but a coverage-based exclusion made on unreported grounds is not defensible.

**Closes when:** the Component 02 report lists excluded raw files with reasons.

### ISS-17 — `ruff check .` reports 692 findings in inherited files
**Severity:** Immaterial · **Owner:** _unassigned_ · **Status:** Open

All in inherited `benchmarks/`, `models/`, `src/graph/`, `training/` and the root runners;
identical on every branch and not a regression from any component. Two sit in
`tests/test_split_integrity.py`. New component modules pass focused Ruff checks.

**Immaterial because:** a clean repository-wide lint run is not the baseline, and mass-fixing
inherited files would destroy the diff against the inherited implementations before they are
classified file by file. Do not bulk-fix to silence it.
**Closes when:** inherited files are classified and either retained-and-cleaned or removed.

---

## Documentation gaps

### ISS-18 — Component 03 split summary and plot are not produced
**Severity:** Material · **Owner:** Component 03 · **Status:** Open

The Component 03 contract requires `config/splits/sydney_5fold.md` — stations per fold, plus
a scatter plot of the fold-0 assignment. Only `sydney_5fold.json` and `sydney_5fold.yaml`
exist. The contract's own reasoning was that "one picture stops a lot of arguments about
whether the split is sane"; right now no one can check a split by looking at it.

**Closes when:** the split CLI writes the summary and the plot.

### ISS-19 — Component 03 has no component README
**Severity:** Material · **Owner:** Component 03 · **Status:** Open

Every other component has one: `docs/ingestion/` (01), `docs/canonical/` (02),
`docs/graphs/` (04), `docs/models/` (06). There is no `docs/study/`. Under the Sprint 2
standard — no PR until your component has its own README — Component 03 does not meet it.

**Closes when:** `docs/study/README.md` exists with commands, inputs, outputs and known limits.

### ISS-20 — Canonical README's Sydney statistics table is stale
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

The "What the Sydney run produced (2026-09-08)" table predates the rain-day boundary fix, so
day verdicts and hourly counts have shifted. It is also pre-dated by the region-scope change
in ISS-11, so a rerun today would describe a different area entirely. The table is marked
stale in place, but stale numbers in a component README get quoted anyway.

**Closes when:** ISS-11 is resolved, the build is rerun, and the table is replaced.
