# Temporary codebase review handoff

> **Temporary document:** use this brief for the next repository review, then delete it
> after the project lead has accepted the review. It is not a seventh requirements
> document and must not become permanent implementation authority.

**Review target:** `integration/sydney-gauge-baseline` at `dc1ad92`

**Review mode:** read-only audit. Do not implement a numbered component during this
review.

## 1. Purpose of this review

Decide whether the repository is safe and clear enough to hand to component owners.
The review must identify:

- code that is ready to reuse;
- code that contains useful ideas but needs extraction or correction;
- inherited code that should remain historical reference only;
- files proposed for removal because they are broken, misleading, duplicated, or
  permanently outside scope;
- contradictions between code, tests, dependencies, and current requirements; and
- blockers that must be resolved before each component can begin.

This review is not permission to delete or broadly rewrite files. A previous cleanup
incorrectly removed generic spatial and modelling code together with Singapore-specific
code. The deletions were rolled back. Classify conservatively and provide evidence before
the project lead decides what to remove.

## 2. Current project direction

The target system combines Australian:

1. Bureau of Meteorology (BoM) rain gauges;
2. validated Australian radar data from the National Computational Infrastructure
   (NCI); and
3. validated Himawari satellite data from NCI.

The first scientific milestone is gauge-only rainfall reconstruction for Greater Sydney.
The required baseline period is **2021-01-01 through 2022-12-31 inclusive**.

Singapore data, National Environment Agency formats, Singapore timezone assumptions,
and commercial microwave link (CML) data are permanently outside scope. They must be
removed from the eventual active system, not retained as optional features. However, a
file containing those assumptions may also contain a useful generic algorithm. Recommend
extraction before removal when that is the case.

The inherited Sydney rainfall export is regression evidence only. Its values are
scientifically distorted and cannot be treated as millimetres or used for official
results. The production path begins with new source-faithful BoM ingestion.

## 3. Authority and efficient reading order

Follow [AGENTS.md](../AGENTS.md). For this repository-wide review, use this focused order:

1. [README.md](../README.md)
2. [Implementation handoff](IMPLEMENTATION_HANDOFF.md)
3. [Component map and shared rules](requirements/README.md)
4. [Legacy-code guide](LEGACY_CODE.md)
5. this review brief
6. the requirements document related to each code area being assessed
7. the implementation and tests for that area

Do not read the complete archive by default. Open an archived audit only when an active
document or a specific finding cites it. Current instructions outrank archived material.

## 4. Starting state to verify

At the time this brief was written:

- the branch was `integration/sydney-gauge-baseline`;
- the tracked worktree was clean;
- recent commits were:
  - `dc1ad92 docs: publish component implementation handoff`
  - `743fb1b refactor: constrain transitional runner to Sydney gauges`
  - `661614c chore: isolate transitional Sydney regression paths`
- `.venv\Scripts\python.exe -m pytest tests -q -rx` reported **22 passed and
  7 expected failures**;
- the expected failures describe inherited data, split, identity, and edge-normalisation
  defects; they are not acceptance tests for the new components; and
- `git diff --check` reported no tracked whitespace errors.

Re-run these checks. Do not assume a passing focused test suite means every Python module
can import or compile: much of the restored reference tree is not exercised by the tests.

## 5. Confirmed boundaries

Treat these as different kinds of code:

| Area | Current status |
|---|---|
| `train_fused.py`, root `config.yaml` | Transitional Sydney gauge-only regression runner; not target architecture |
| `adapters/legacy_sydney_gauge_export.py` | Transitional conversion path; not Component 01 or 02 |
| `src/graph/gaugegraphnew.py` and retained gauge/model/training core | Potentially reusable behind Components 04 and 06, subject to explicit regression tests |
| `benchmarks/`, `src/radar/`, `src/visualization/`, `src/cml/`, extra graph/model files | Restored inherited reference code awaiting file-by-file classification |
| `docs/requirements/` | Current component contracts |
| `docs/archive/` | Historical evidence, not current authority |
| ignored `database/` and `experiments/` contents | Local/generated material; not production contracts and normally not committed |

Do not describe the restored inherited radar code as Australian radar support. Australian
radar and satellite implementation must start from the products and contracts validated
by Components 01 and 02.

## 6. Review work

### A. Classify the active code tree

Classify every top-level executable, package, and meaningful module outside
`docs/archive/` into one of these categories:

1. **Keep and use** — relevant, understandable, and sufficiently tested.
2. **Keep, then extract/refactor** — contains useful gauge, radar, satellite, spatial,
   temporal, modelling, or diagnostic logic mixed with unsafe assumptions.
3. **Historical reference only** — useful as evidence but should not be imported by the
   target system.
4. **Proposed removal** — broken, duplicated, misleading, or entirely out of scope.

For every category 2–4 decision, name the concrete evidence: imports, hard-coded bounds,
units, coordinates, timestamps, data paths, callers, tests, or lack of callers. Do not
classify a whole directory from its name alone.

Map reusable code to Components 01–06 where possible. Flag code with no current or
planned owner.

### B. Audit scope violations without losing useful logic

Search case-insensitively for Singapore, NEA, CML, Singapore-specific coordinates and
timezones, regional filenames, and old local paths. Separate:

- entirely out-of-scope implementations;
- generic functions with only stale comments or names; and
- mixed modules requiring useful code to be extracted first.

Also identify whether dependencies exist only to keep out-of-scope reference modules
importable. Do not equate a dependency's presence with supported capability.

### C. Check runnability and dependency truth

Review all apparent entry points, not only `train_fused.py`. Check:

- Python syntax and importability;
- imports of nonexistent functions or packages;
- functions whose signatures no longer match their callers;
- configuration read at import time;
- default arguments that immediately lead to `NotImplementedError`;
- machine-specific paths or assumptions;
- whether `requirements.txt` covers runtime code and `requirements-dev.txt` covers only
  development/test tools; and
- whether supposedly reusable modules have focused tests.

Report broken or orphaned entry points. Do not repair them in this review.

### D. Check scientific and interface safety

Trace coordinate, unit, timestamp, station-identity, missing-data, and split assumptions
through loaders, sampling, graphs, models, and outputs. In particular, look for:

- `(longitude, latitude)` mixed with `(latitude, longitude)`;
- raw degree distances treated as kilometres;
- hard-coded geographic bounds or grid resolution;
- naive timestamps, silent timezone conversion, or Singapore time;
- public station numbers used as unique keys instead of BoM internal `station_id`;
- missing rainfall converted to zero;
- rainfall rescaling without verified units;
- private or leaking train/validation/test splits;
- graph-wide versus per-node edge normalisation; and
- radar product assumptions unsupported by an identified Australian NCI product.

Distinguish a documented compatibility behaviour from an accidental defect. A successful
smoke run is evidence of runnability only, not scientific validity.

### E. Review tests and delegation readiness

Identify what the current suite actually covers and what it cannot prove. Compare it to
the acceptance tests in Components 01–06. Report:

- missing unit or integration coverage for retained code;
- expected failures that correctly preserve known evidence;
- stale, misleading, environment-dependent, or overfitted tests;
- the smallest offline test slice required before teammates work independently; and
- which components can begin now with stubs versus which require real outputs or access.

### F. Review repository clarity and metadata

Check the root, package layout, Git tracking/ignore rules, dependency files, active docs,
and archive index for stale or conflicting metadata. Look for misleading names,
duplicate entry points, obsolete instructions, generated files, cached outputs, notebooks,
logs, and documents that appear current but are historical.

Do not solve documentation discomfort by adding more general documents. Recommend a
change only when it removes ambiguity, fixes a contradiction, or makes a component
directly implementable.

## 7. Known leads that require confirmation

These are review leads, not a substitute for inspecting the files:

- `src/graph/gaugegraphnew.py` explicitly flags inconsistent coordinate order when
  connecting gauge nodes to other node types.
- `models/hgnn.py` appears syntactically incomplete.
- `src/radar/preprocessor.py` appears tied to Singapore bounds, paths, filenames, and
  fields; it must not be mistaken for the Australian radar pipeline.
- `src/visualization/main.py` appears to call the transitional station loader without
  its now-required years and imports a plotting dependency not listed in the focused
  environment.
- `src/sampling/main.py` deliberately blocks inherited Singapore plotting, and a plotting
  default may make an otherwise generic sampling call fail.
- several benchmark, graph, visualization, and CML modules contain out-of-scope logic,
  while some neighbouring spatial/temporal/model helpers may still be generic.
- the current tests focus on the transitional gauge path and inherited defects; restored
  reference modules may have no syntax/import coverage.
- root dependencies retained for restored reference code may be removable after useful
  logic is extracted and out-of-scope code is removed.

Confirm or reject each lead with file-and-line evidence. Add important findings that are
not listed here.

## 8. Required review output

Return one concise review report in the conversation. Do not create another handoff,
roadmap, or requirements document.

Use this structure:

1. **Delegation verdict** — ready, conditionally ready, or not ready, with a short reason.
2. **Critical findings** — correctness, scientific validity, security/data safety, or
   handoff blockers, ordered by severity and citing file/line locations.
3. **File classification table** — path/area, category 1–4, evidence, intended component,
   and recommended action.
4. **Requirements and documentation conflicts** — only concrete contradictions or
   missing decisions.
5. **Test and dependency gaps** — commands run, exact results, and uncovered risks.
6. **Safe removal proposal** — files that may be removed now versus files requiring
   extraction first. Do not perform the removals.
7. **Delegation matrix** — for Components 01–06, state what can start, required input,
   first sensible slice, and blocker.
8. **Next three actions** — smallest ordered actions that most improve handoff safety.
9. **Uncertainties** — facts that need project-lead, source-data, or NCI confirmation.

Prefer evidence and direct language over exhaustive narration. Explain unfamiliar terms
when first used. If a finding is only an inference, label it as such.

## 9. Review guardrails

During this review, do not:

- delete, rename, or rewrite code;
- edit requirements to match inherited behaviour;
- stage, commit, merge, or push;
- restore Singapore, NEA, or CML capability in any form;
- declare inherited radar code suitable for Australia without product evidence;
- treat ignored legacy data as an official component output;
- weaken or add expected failures merely to make tests green; or
- create new architecture to avoid a documented component contract.

If a destructive or scope-changing action appears necessary, report the exact proposed
change and wait for the project lead's decision.
