# COMP3888 Baseline Reconciliation Requirements

**Document status:** Initial requirements for code agent  
**Primary objective:** Establish a validated Australian/Sydney 0D gauge-only reference run before any project extensions.  
**Target integration direction:** Use the Jylow repository as the candidate extensible framework, and determine whether Niramay's Sydney gauge data can be adapted into it without altering the scientific protocol or breaking Jylow's existing 0D/1D/2D capability.

---

## 1. Context

The team has imported two previous researchers' implementations into separate branches of the team repository.

### Niramay implementation
- Australian/Sydney-focused.
- Known at present to use **0D rain-gauge data only**.
- Appears to include pre-downloaded gauge data.
- Its data provenance, schema, preprocessing, split logic, graph construction, target masking, model inputs, evaluation procedure, and reproducibility have **not yet been independently validated**.

### Jylow implementation
- Singapore-focused.
- More complete/extensible implementation.
- Supports **0D, 1D, and 2D spatial supports** in the model/research code.
- Its exact data contracts, support-selection behaviour, preprocessing, split logic, graph construction, model inputs, and reproducibility have **not yet been independently validated**.

### Current project constraint
The team does **not yet have NCI Gadi access**, although the client has indicated that relevant radar/satellite datasets already exist on NCI. Therefore, the first baseline-validation milestone must **not depend on NCI-only data**.

### Working hypothesis to test
A sensible first integration target is:

> **Jylow model/framework + Niramay Sydney gauge data + 0D-only execution + common baselines/evaluation.**

This is a hypothesis, not an assumption. The code agent must first prove that the data and model contracts are compatible or identify the minimal adapter required.

---

# 2. System-level objective

Produce a reproducible experiment in which:

1. Niramay's Sydney gauge dataset is understood and validated;
2. Niramay's native implementation can be executed, if technically possible, to establish a local reference result;
3. Jylow's implementation can execute in a genuine **0D gauge-only mode**;
4. the same Sydney gauge observations and held-out targets can be supplied to Jylow through a clearly documented adapter;
5. statistical baselines and the neural model are evaluated on the **same target records**;
6. results, splits, configuration, seeds, logs, and predictions are persisted as reproducible artifacts;
7. the agent produces a precise inventory of additional data required for later 2D radar/satellite fusion, without blocking the gauge-only baseline on those datasets.

No temporal, radar, satellite, elevation, wind, attention, architecture-scaling, Australia-wide, or other research extension is in scope for this task.

---

# 3. Critical engineering principle

**Do not redesign either repository before understanding and reproducing it.**

The order of work is:

```text
AUDIT -> NATIVE REPRODUCTION -> DATA CONTRACT -> ADAPTER -> JYLOW 0D RUN
      -> COMMON BASELINES/EVALUATION -> VALIDATION REPORT -> ONLY THEN REFACTOR
```

A clean architecture is desirable later, but premature refactoring would make it impossible to distinguish:

- inherited behaviour;
- integration behaviour;
- bugs introduced by the team;
- genuine model differences.

---

# 4. Repository and branch rules

## BR-001 — Preserve source branches
Treat the two imported researcher branches as **read-only reference branches**.

Do not:
- rewrite their history;
- make cleanup commits directly on them;
- merge one wholesale into the other;
- delete apparently unused files before the audit is complete.

## BR-002 — Create a dedicated integration branch
Create a new integration branch, preferably from the Jylow branch because Jylow is the current candidate extensible framework.

Suggested name:

```text
integration/sydney-gauge-baseline
```

If the repository already has naming conventions, follow them instead.

## BR-003 — Use both branches side-by-side during the audit
Use Git worktrees or equivalent so both implementations can be inspected and executed independently.

## BR-004 — Keep integration commits separable
Prefer commits in this order:

1. audit/documentation only;
2. Sydney gauge-data adapter and golden fixture;
3. Jylow 0D-only execution support/configuration;
4. common prediction/evaluation harness;
5. validation fixes that are demonstrably necessary.

Do not mix broad formatting/refactoring with scientific changes.

---

# 5. Phase 0 — Baseline archaeology / repository audit

Before modifying code, produce an evidence-based system map of both repositories.

## FR-001 — Identify the native entry point
For each repository identify the exact command or sequence used to:

- preprocess data;
- create graphs/samples;
- train the model;
- load a checkpoint;
- run inference;
- run statistical baselines;
- evaluate results.

Record whether execution is CLI-, script-, notebook-, or configuration-driven.

## FR-002 — Trace the end-to-end call chain
For each repository, document:

```text
entry point
  -> configuration
  -> data files
  -> preprocessing
  -> split construction
  -> graph/sample construction
  -> model input
  -> training
  -> prediction
  -> evaluation
```

Include actual source file names, functions/classes, and line references where practical.

## FR-003 — Audit all file-system data dependencies
Search the repository for data reads and writes, including but not limited to:

- `open(...)`
- `read_csv`, `read_parquet`, `read_excel`
- `np.load`, `torch.load`
- NetCDF/xarray/Zarr/HDF readers
- `glob`/path patterns
- hard-coded absolute paths
- environment variables
- configuration files
- notebook paths
- ignored data directories

Classify every data asset as:

1. **bundled in repository**;
2. **generated by preprocessing**;
3. **downloaded from a public source**;
4. **expected at an external/local path**;
5. **NCI-only or believed to be NCI-hosted**;
6. **missing/unknown**.

## FR-004 — Produce a data dictionary
Do not infer feature meaning merely from variable names. Trace transformations back to source data where possible.

For every model input/target, document:

- support type: 0D / 1D / 2D;
- source/product;
- variable name;
- physical meaning;
- units;
- raw dtype/shape;
- processed dtype/shape;
- spatial reference / latitude-longitude representation;
- temporal resolution;
- missing-value representation;
- normalization/transformation;
- whether it is an input, target, mask, coordinate, edge feature, or metadata field.

## FR-005 — Audit split and leakage behaviour
For both repositories determine exactly:

- train/validation/test definition;
- whether splitting is spatial, temporal, random, or some combination;
- whether the same station appears across partitions;
- whether the same timestamp appears across partitions;
- when normalization statistics are computed;
- how target gauge values are masked;
- whether any target-derived values can leak into inputs;
- what observations IDW/Kriging use when predicting a held-out gauge.

Do not modify this behaviour during the initial audit.

## FR-006 — Audit graph construction
For both repositories document:

- node types;
- supported spatial supports;
- edge/relation types;
- directionality;
- neighbour-selection algorithm;
- `k` or radius parameters;
- distance calculation;
- edge weights/features;
- self loops;
- target masking;
- whether graph topology changes with time or target selection.

## FR-007 — Audit model contract
For each model document the exact conceptual contract:

```text
input object/tensors + shapes
        -> model forward pass
        -> output tensor + shape + semantic meaning
```

Also document:

- loss function;
- optimizer;
- learning rate;
- epochs;
- early stopping;
- batch size;
- random seed behaviour;
- checkpointing;
- metric calculation.

### Required Phase 0 outputs
Create:

```text
docs/baseline-reconciliation/
    AUDIT_NIRAMAY.md
    AUDIT_JYLOW.md
    DATA_INVENTORY.md
    SYSTEM_MAP.md
```

If repository conventions differ, equivalent locations are acceptable.

---

# 6. Phase 1 — Reproduce Niramay natively

Niramay's implementation is the local Australian control because it already targets Sydney and appears to contain gauge data.

## FR-101 — Confirm what the included Sydney data actually are
Do not assume the files are raw observations.

Determine whether Niramay includes:

- raw gauge observations;
- cleaned/aggregated observations;
- precomputed tensors;
- graph objects;
- train/test split files;
- some subset of the above.

Trace provenance as far back as repository documentation/code permits.

## FR-102 — Validate basic data integrity
At minimum check:

- number of stations;
- time range;
- temporal frequency;
- rainfall units;
- coordinate range;
- missing-value rate;
- duplicate station-time records;
- impossible/negative rainfall values unless explicitly encoded;
- number of observations in each split.

Do not silently remove suspicious records. Report them first.

## FR-103 — Execute the native Niramay pipeline unchanged
Attempt the documented/original run before changing model or preprocessing code.

Persist:

- exact command;
- Python/package environment;
- random seed(s);
- configuration;
- stdout/stderr;
- predictions if available;
- baseline results;
- neural-model results;
- runtime;
- commit hash.

If reproduction fails, document the **first blocking failure** and resolve only the minimum necessary compatibility issue. Every compatibility fix must be isolated and explained.

## FR-104 — Establish the Niramay reference result
Record, without reinterpretation:

- baseline metric(s);
- model metric(s);
- relative model improvement/degradation against each baseline;
- exact evaluation sample count.

For an error metric such as RMSE, report relative improvement as:

```text
100 * (baseline_error - model_error) / baseline_error
```

Only use this formula where lower metric values are better.

---

# 7. Phase 2 — Determine whether Jylow genuinely supports 0D-only execution

## FR-201 — Discover support-selection mechanism
Identify how Jylow determines which spatial supports/node types are present.

Possible outcomes:

1. Jylow already has an explicit 0D-only configuration;
2. Jylow accepts absent 1D/2D node stores/relations naturally;
3. Jylow assumes all support types exist and needs a minimal compatibility change.

Document which outcome is true.

## FR-202 — Do not fabricate sensor data
To achieve gauge-only execution:

- do **not** create fake CML/1D observations;
- do **not** create fake radar/satellite values;
- do **not** duplicate gauge values into 2D features.

If the graph library requires empty stores for unsupported relation types, empty tensors/edge sets are acceptable only if they contribute no messages and the behaviour is tested.

## FR-203 — Preserve multi-support capability
Any change made to support 0D-only execution must leave Jylow's existing 0D/1D/2D path intact.

A support set should be configurable rather than achieved by deleting 1D/2D code.

Suggested conceptual configuration:

```yaml
supports:
  point_0d: true
  line_1d: false
  area_2d: false
```

Use Jylow's native configuration conventions if they already exist.

## FR-204 — Smoke-test one 0D forward pass before training
Before full training, prove that Jylow can:

1. load a tiny gauge-only sample;
2. build a valid 0D graph;
3. perform a forward pass;
4. calculate finite loss;
5. backpropagate one step;
6. produce finite predictions.

---

# 8. Phase 3 — Reconcile Niramay Sydney data with Jylow

The preferred approach is an **adapter**, not a one-off rewrite of the dataset.

## FR-301 — Define the Niramay -> Jylow mapping
Produce an explicit mapping table:

| Niramay field/artifact | Meaning | Transformation | Jylow field/artifact | Notes |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

Every transformation must be explicit, including:

- units;
- coordinate order;
- normalization;
- station IDs;
- timestamps/time zones;
- target masks;
- split labels;
- graph-neighbour inputs;
- missing-value masks.

## FR-302 — Separate raw/source data from derived Jylow artifacts
Do not overwrite Niramay's original files.

Conceptually maintain:

```text
source Sydney gauge data
        -> adapter/preprocessing
        -> Jylow-compatible 0D representation
```

If the repository stores data outside Git, follow that convention.

## FR-303 — Preserve split identity for cross-model comparison
The first Jylow-on-Sydney validation experiment must use the **same held-out station-time targets** as the Niramay control whenever the Niramay split can be reconstructed.

If the exact split cannot be reconstructed:

- do not claim direct reproduction;
- create one explicit common split;
- run both Niramay and Jylow on that split if technically possible;
- label the experiment as a controlled re-evaluation rather than paper reproduction.

## FR-304 — Preserve training-only transformations
Any normalization statistics or learned preprocessing parameters must be fit using the training partition only unless the inherited scientific protocol explicitly does otherwise.

If inherited behaviour violates this rule, do not silently change it in the reproduction run. Report the issue and create a separate corrected experiment later.

---

# 9. Golden fixture requirements

Create a tiny deterministic Sydney gauge-only fixture from the validated source dataset.

## FR-401 — Fixture purpose
The fixture exists to test the pipeline, **not** to estimate scientific performance.

It must be small enough to run locally on CPU in a short test.

## FR-402 — Fixture content
Include enough data to exercise:

- multiple gauges;
- at least several timestamps;
- train and evaluation targets;
- non-zero rainfall where available;
- zero-rain observations;
- masking;
- graph construction;
- prediction output.

## FR-403 — Fixture stability
The fixture selection must be deterministic and versioned. Record source record IDs/ranges or a reproducible selection rule.

Do not regenerate a different fixture on each test run.

---

# 10. Phase 4 — Common baseline and evaluation protocol

The purpose is not yet to build the final semester-wide benchmark system. Build the **minimum fair evaluator needed to validate the port**.

## FR-501 — Same targets for every method
For a given experiment, Niramay, Jylow, IDW, and any other baseline being compared must be evaluated on exactly the same target station-time records.

## FR-502 — Baseline inputs must respect the split
A baseline may use only observations it would legitimately have under the experiment's holdout protocol.

A held-out target rainfall value must never be available to IDW/Kriging or the neural model.

## FR-503 — Persist row-level predictions
Do not store only summary RMSE.

Persist a prediction artifact containing at minimum:

```text
experiment_id
method
station_id
timestamp
latitude
longitude
y_true
y_pred
split
```

Add useful metadata where available.

## FR-504 — Persist run metadata
For each run record:

- git commit;
- source branch provenance;
- dataset/fixture version;
- split definition/version;
- model configuration;
- random seed;
- package/environment information;
- device;
- runtime;
- output paths.

## FR-505 — Initial metrics
At minimum report the metrics already used by the inherited implementations/papers, plus RMSE if available/appropriate.

Do not change the historical metric definition and then call the result a reproduction.

Where both methods use the same error metric, calculate the relative delta over baseline.

---

# 11. Validation logic

The scientific objective of this task is **not** to force Jylow's Sydney result to equal a paper result from another domain.

Three levels of evidence must be kept distinct.

## Level A — Native reproduction
Can Niramay reproduce its own reported/expected Sydney behaviour on its own dataset?

## Level B — Controlled implementation comparison
When Niramay and Jylow receive the same Sydney gauge data, targets, split, and baseline protocol, how do their results compare?

This is the most informative validation of the port.

## Level C — Paper-level sanity check
Does the gauge-only neural approach show an improvement/degradation over statistical interpolation that is qualitatively plausible relative to the relevant paper result?

**Do not make matching a Singapore paper's percentage improvement a hard acceptance criterion for Sydney.** Different geography, gauge density, rainfall regime, preprocessing, split, graph architecture, and hyperparameters can legitimately produce a different delta.

### Validation acceptance criteria
The integration passes this milestone when:

1. Jylow completes a 0D-only Sydney run without synthetic 1D/2D data;
2. outputs are finite and correctly aligned to held-out targets;
3. baselines and Jylow use identical target records;
4. the split and leakage rules are documented;
5. Jylow's relative performance versus baseline is reported;
6. Niramay's relative performance versus the same baseline is reported where possible;
7. differences are explained or converted into explicit investigation items;
8. no result is declared "reproduced" unless the relevant data, split, preprocessing, model configuration, and metric definition genuinely match.

---

# 12. Data acquisition and NCI gap analysis

The team does not yet have Gadi access. This must not block the gauge-only milestone.

## FR-601 — Build a machine-readable data manifest
Create a manifest for every known/required dataset.

Suggested fields:

```yaml
name:
support_type:          # 0D / 1D / 2D
role:                  # input / target / covariate
region:
time_range:
temporal_resolution:
spatial_resolution:
variables:
units:
source_authority:
public_url:
nci_path:              # UNKNOWN until verified if necessary
repository_path:
access_status:          # available / missing / NCI-pending / download-needed
raw_or_processed:
processing_required:
approx_size:
licence_or_terms:
checksum:
notes:
```

## FR-602 — Do not bulk-download speculative datasets
Before downloading any large radar/satellite dataset, the agent must identify:

1. exact product;
2. variable required by the model;
3. expected spatial/temporal resolution;
4. study time range;
5. authoritative source;
6. whether the same product is already expected on NCI;
7. estimated download/storage size.

Return this information for human approval first.

## FR-603 — Sydney gauge data are the immediate data priority
If Niramay's repository already contains the required Sydney gauge observations, use them for the first baseline milestone after provenance/integrity validation.

Do not block baseline reproduction on radar, satellite, wind, elevation, or Gadi access.

## FR-604 — Prepare, but do not implement, the 2D acquisition plan
After auditing Jylow, identify exactly what its 2D interface expects and create a gap list for future Australian radar/satellite integration.

For each 2D source report:

- expected variable(s);
- tensor/data shape expected by Jylow;
- preprocessing assumptions;
- whether Singapore-specific logic must be replaced;
- likely Australian source/NCI dependency if known;
- information still requiring NCI access/client confirmation.

Do **not** port Singapore radar/satellite assumptions blindly to Australia.

---

# 13. Tests required

Implement tests around the integration boundary. Exact filenames may follow project conventions.

## T-001 — Sydney gauge loader
Verify:
- loader returns non-empty records;
- timestamps parse consistently;
- coordinates are finite;
- rainfall values obey documented validity rules;
- station IDs are stable.

## T-002 — Split integrity
Verify:
- every target belongs to the intended split;
- no target value leaks into permitted predictors;
- comparison methods use identical target IDs/timestamps.

## T-003 — Adapter mapping
Given fixed Niramay records, verify Jylow-compatible fields/shapes/units match the documented mapping.

## T-004 — Jylow 0D graph smoke test
Verify a gauge-only graph builds without real or dummy 1D/2D observations.

## T-005 — Forward/backward smoke test
Verify one batch produces finite loss, gradients, and predictions.

## T-006 — Prediction-schema test
Verify all methods emit the required prediction fields and that `y_true` aligns exactly by station and timestamp.

## T-007 — Golden fixture end-to-end test
Run:

```text
fixture -> Jylow 0D graph -> model -> predictions -> evaluator
```

The test should be runnable without NCI/Gadi.

---

# 14. Non-functional requirements

## NFR-001 — Reproducibility
Every scientific result must be attributable to:

```text
code commit + config + data version + split + seed
```

## NFR-002 — No hidden machine-specific paths
New integration code must not depend on one developer's absolute filesystem paths.

Use config/environment variables/path resolution consistent with repository conventions.

## NFR-003 — Fail loudly
Missing files, unsupported support combinations, invalid shapes, and split mismatches must produce explicit errors rather than silently dropping data.

## NFR-004 — Minimal change principle
Until validation is complete, prefer adapters/configuration changes over rewrites of inherited model code.

## NFR-005 — Preserve provenance
Whenever code or data logic is copied/adapted from one researcher branch into the integration branch, record its origin in commit messages and/or module documentation.

## NFR-006 — Local-first baseline
The golden fixture and basic 0D smoke test must run without Gadi. Gadi/PBS support is a later execution concern, not a prerequisite for understanding the inherited baseline.

---

# 15. Explicit non-goals

The code agent must **not** implement these during this reconciliation task unless separately authorised:

- radar fusion;
- satellite fusion;
- CML/1D Australian data;
- temporal GNN-LSTM/GRU extensions;
- elevation;
- wind;
- temperature;
- distance-to-coast features;
- attention/GAT experimentation;
- GraphSAGE/GIN/transformer experiments;
- hyperparameter sweeps;
- Australia-wide scaling;
- production PBS orchestration;
- architecture-wide refactors unrelated to the baseline port;
- performance tuning intended to improve the result.

The purpose is to understand and validate the inherited baseline before extending it.

---

# 16. Required final deliverables from the code agent

At completion, provide:

### A. Repository audit
```text
AUDIT_NIRAMAY.md
AUDIT_JYLOW.md
SYSTEM_MAP.md
```

### B. Data documentation
```text
DATA_INVENTORY.md
data_manifest.yaml
NIRAMAY_TO_JYLOW_MAPPING.md
```

### C. Reproducible baseline runs
Provide exact commands/configurations for:

```text
1. Niramay native Sydney run, if reproducible
2. Jylow Sydney gauge-only smoke run
3. Jylow Sydney gauge-only full/reference run
4. Common statistical baseline evaluation
5. Golden-fixture end-to-end test
```

### D. Persisted artifacts
At minimum:

```text
config
run metadata
logs
predictions
metrics
checkpoint where applicable
```

### E. Validation report
Create `BASELINE_RECONCILIATION_REPORT.md` answering:

1. What data does Niramay actually contain?
2. What are the gauge features and target?
3. What preprocessing does Niramay perform?
4. What split does Niramay use?
5. What graph does Niramay construct?
6. What does Jylow expect for a 0D node/sample?
7. Can Jylow run with only 0D support without fake sensor data?
8. What transformations were required to feed Sydney data into Jylow?
9. On an identical evaluation set, what are the IDW/Kriging/native/Jylow metrics where available?
10. What are the relative deltas over baseline?
11. Are differences attributable to data, splits, preprocessing, graph construction, architecture, optimization, or still unknown?
12. What exact datasets will be required next for Australian 2D radar/satellite fusion?
13. Which of those datasets are locally/publicly accessible and which remain pending NCI access?
14. What should the team do next, and what should explicitly **not** be changed yet?

---

# 17. Stop / escalation conditions

Stop and report before making a scientific substitution if any of the following occur:

- the Niramay gauge dataset's meaning or provenance cannot be established;
- the two repositories use materially different prediction targets;
- the split required for the claimed comparison cannot be reconstructed;
- Jylow fundamentally cannot execute without a 1D or 2D support;
- obtaining 0D-only behaviour requires changing the mathematical architecture rather than only handling missing support types;
- a required dataset is missing and multiple plausible products could substitute for it;
- reproduction would require changing preprocessing, target masking, metric definitions, or evaluation targets;
- a large external download is required;
- a suspected data-leakage issue is discovered.

In these cases, return:

```text
BLOCKER
Evidence
Why it matters
Options
Recommended minimal next action
```

Do not choose a scientifically consequential workaround silently.

---

# 18. Definition of done

This reconciliation milestone is complete only when the team can truthfully say:

> We understand the inherited Sydney gauge data and both repositories well enough to trace one observation from source file to model input to prediction. We can execute Jylow's framework in a real 0D gauge-only mode using Sydney gauge data, evaluate it and the statistical baselines on the same held-out targets, reproduce the run from recorded configuration and artifacts, and clearly identify which additional 2D Australian datasets are required next. No extension work was needed to achieve this baseline.
