# Implementation handoff

**Audience:** component owners and reviewers  
**Branch:** `integration/sydney-gauge-baseline`  
**Baseline study period:** 2021-01-01 through 2022-12-31 inclusive

## 1. What has already been established

The archaeology phase is complete enough to begin component implementation:

- The Sydney/Niramay tree is gauge-only and its native pipeline was reproduced.
- The Jylow framework can execute with genuine 0D gauge support and no fabricated CML
  or radar nodes.
- A transitional adapter and imported split proved that Jylow can train on real Sydney
  station layouts.
- Both inherited trees used graph-wide edge normalisation instead of the paper's
  per-node equation. An explicit opt-in correction exists and was tested.
- The hand-exported Sydney rainfall values are distorted by a time- and station-varying
  export artefact. They cannot be repaired by one unit conversion and must not be used
  for scientific error magnitudes.
- BoM internal station ids resolve the public-number collisions and must be the
  canonical identity.

The current Jylow smoke runs prove **runnability**, not scientific validity. Their data,
training length, target coverage, validity handling, positional encoding, and random
seeding do not satisfy the final architecture.

## 2. Sources of truth

Use this precedence when documents differ:

1. [requirements/README.md](requirements/README.md) and the assigned component brief;
2. [requirements/AUTHORING_STANDARD.md](requirements/AUTHORING_STANDARD.md);
3. this handoff;
4. [baseline-reconciliation/STATUS.md](baseline-reconciliation/STATUS.md) and dated
   supersession notes;
5. older audit/handoff sections as historical evidence;
6. [archive/](archive/README.md) only for planning provenance.

`baseline-reconciliation/HANDOFF.md`, `SYSTEM_MAP.md`, and parts of `DATA_CONTRACT.md`
describe earlier points in the investigation. Their evidence remains useful, but their
old “not implemented” or “blocked” status lines are not current work instructions.

## 3. Component work order

| Component | Can start now? | Must wait for | Primary handoff |
|---|---|---|---|
| 01 source ingestion | Yes: implementation and small validation queries | Client approval before full baseline retrieval; NCI access for radar/satellite | acquisition index and source-faithful landing to 02 |
| 02 canonical data | Yes against a small accepted 01 sample | Full baseline build waits for 01 gauge landing | versioned observations/registry to 03/04/05 |
| 03 splits and seeds | Yes: schemas, seed loader, algorithms, synthetic tests | Production release waits for accepted 02 dataset | audited split and exact targets to 04/06/baselines |
| 04 graph samples | Yes against legacy/fixture adapters | Production release waits for 02+03 | typed samples and manifest to 06 |
| 06 model adapter | Yes against current graph objects/stubs | Exact prediction acceptance waits for 03+04 | validated predictions to future evaluation |
| 05 golden fixture | Start selection/build tooling now | Final expectations need accepted outputs from 01–04/06 | offline system test to everyone |

The table describes dependencies, not mandatory staffing. If the team has fewer owners,
combine 03+04 or 05+CI before combining source ingestion with modelling; data provenance
and model behaviour need independent review.

## 4. Recommended parallel lanes

### Lane A — Data foundation

Owner 01 implements config validation, landing layout, provenance receipts, progress,
resume, BoM discovery, representative series validation, and small test downloads.
Owner 02 agrees the acquisition-index fields early and implements preflight, hourly UTC
aggregation, daily-total verification, registry, exclusions, and dataset versioning.

Do not run the complete 2021–2022 station acquisition until the client approves it.
Small, rate-limited validation queries required by the component brief are permitted
only under the documented BoM pacing policy.

### Lane B — Study and graph integrity

Owner 03 lands `config/seeds.yaml`, target-index schema, split audit, reference protocol,
and corrected full-coverage protocol against a synthetic/legacy registry. Owner 04
removes import-time config from the graph boundary, carries station/time mappings, and
implements compatibility/corrected mode tests.

The owners must agree `station_id` dtype, target-index sorting, target hashes, and fold
role names before either public interface is merged.

### Lane C — Model/prediction boundary

Owner 06 wraps the existing HGNN training logic, defines the run directory, implements
restart checkpoints, and lands the prediction validator first. Use a stub target index
matching component 03's documented schema until the real artifact exists.

No evaluation logic should be embedded here. A valid row-level prediction artifact is
the deliverable.

### Lane D — Fixture and integration

Owner 05 builds fixture selection, manifest, external truth, mutation tests, and offline
bootstrap in parallel. Initially wire completed layers and skip unavailable layers with
the precise output named. Turn each skip into a failure once that output exists.

## 5. Repository boundaries

### Active implementation locations

New component code should use these packages and config roots:

```text
src/ingestion/             config/ingestion/
src/canonical/             config/canonical/
src/study/                 config/splits/ and config/seeds.yaml
src/graphs/                config/graphs/
src/models/adapters/       config/models/
src/fixtures/ or tools/fixtures/   config/fixtures/
```

Create a package only when its owner begins the component; do not scaffold empty
architectural layers solely to match the diagram.

### Transitional code

`train_fused.py`, `src/graph/gaugegraphnew.py`, `models/gnn.py`, and
`training/logic_hetero.py` are the useful inherited core. Wrap and regression-test them;
do not assume their current interfaces are the target architecture.

`adapters/niramay_to_jylow.py` is evidence for the transitional CSV and split shapes.
It reads the compromised hand export and therefore cannot become the new ingestion or
canonical pipeline by renaming it.

See [LEGACY_CODE.md](LEGACY_CODE.md) for retained research variants and removed stale
entry points. The complete imported source remains on `jylow-clone`.

## 6. First-day checklist for every owner

1. Read [requirements/README.md](requirements/README.md), the glossary, and the assigned
   component document fully.
2. Put owner and reviewer names at the document top without changing requirements.
3. Confirm the upstream/downstream schema with those owners; record a blocking mismatch
   before coding around it.
4. Run `git status --short --branch` and preserve unrelated local changes.
5. Run the existing test command and record passes, failures, and data-dependent skips.
6. Create a small implementation plan mapped to `FRxx`, `OUTxx`, `TESTxx`, and `DOCSxx`.
7. Implement the narrowest end-to-end slice that writes the real prescribed artifact.
8. Add component-level acceptance tests before source-specific edge cases.
9. Demonstrate failure behaviour and provenance, not just the happy path.
10. Ask the downstream reviewer to consume the output before declaring the component
    complete.

## 7. Branch and integration discipline

- Keep one component or contract change per branch/PR; do not mix model science with
  repository formatting.
- Do not rewrite `jylow-clone`, `niramay-clone`, or tag `niramay-inherited`.
- Generated data, checkpoints, and experiments stay ignored. Commit only small approved
  fixture data.
- Never delete or reset unrelated dirty-worktree changes.
- If an inherited numerical defect is corrected, preserve a labelled compatibility mode
  and add a before/after regression test.
- A requirements change that affects another component needs that downstream owner's
  review before implementation merges.
- Update documentation and artifact schemas in the same component change.

## 8. Environment and verification

The recorded environment is Python 3.13 with focused dependencies in root
`requirements.txt` and `requirements-dev.txt`. The inherited `requirements.txt` was a
non-portable conda freeze and has been replaced.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q -rx
```

GPU is not required for component onboarding or the golden fixture. Install a matching
official PyTorch CUDA wheel separately for full training and record the build/device.

The transitional Jylow smoke runner additionally needs generated ignored data under
`database/`; follow
[baseline-reconciliation/JYLOW_0D_CHANGES.md](baseline-reconciliation/JYLOW_0D_CHANGES.md)
only when reproducing that historical path.

## 9. Current decisions and blockers

### Decided

- Baseline period: 2021–2022.
- Gauge identity: BoM internal station id.
- Canonical units/time: millimetres and timezone-aware UTC.
- Source data: regenerate from quality-controlled BoM tip records.
- Comparison boundary: exact shared target index and row-level prediction artifacts.
- Corrected graph path: haversine distance, directed KNN, per-node incoming
  normalisation, preserved validity, physical-zero masking, full-graph LPE.
- Inherited behaviour remains available under explicit compatibility labels.

### External blockers

- Full BoM baseline retrieval awaits client approval.
- Radar product confirmation and access await NCI project access.
- Satellite product confirmation and access await NCI project access.
- The paper's exact 75-station Sydney list is unavailable; paper-table reproduction is
  not an acceptance criterion for this dataset.

### Not blockers

- NCI access does not block gauge ingestion/canonical design, splits, graph integrity,
  model adapter work, or the gauge-only golden fixture.
- The missing Singapore dataset does not block a Sydney integration, but it prevents a
  native Jylow paper-result reproduction and remains a stated limitation.

## 10. Definition of component handoff

A component is ready for downstream use only when:

- every required input and output is implemented with the documented schema/path;
- all `TESTxx` cases pass or an external blocker is explicitly accepted;
- produced artifacts contain provenance, config, ids/hashes, and exclusions/failures;
- the operator runbook works from a fresh supported environment;
- the downstream reviewer consumes a real artifact without private parsing or fixes;
- current technical findings are updated if implementation evidence overturned them;
- no official result depends on the distorted inherited rainfall export.

