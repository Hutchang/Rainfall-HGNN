# 04 — Graph and sample construction

**Owner:** _unassigned_ · **Reviewer:** model-adapter owner (component 06)
**Status:** interface work can start now; production samples wait for accepted files
from components 02 and 03

**Baseline period:** 2021-01-01 through 2022-12-31 inclusive
**First milestone:** gauge-only (0D) graph samples; no fabricated non-gauge nodes

## 1. Summary

This component converts Component 02's hourly station observations and Component 03's
accepted split into graph samples that models can read. A **graph** represents each rain
gauge as a node and a selected spatial connection between gauges as an edge. A **sample**
is the rainfall, validity, graph connections, and prediction targets for a defined time
slice. PyTorch Geometric is the Python library used to store and batch those graphs.

This component owns node features, spatial edges, feature scaling, target hiding,
station-to-node mappings, and the saved record explaining how each sample set was made.

The first implementation wraps and tests the useful core of inherited
`GaugeGraphNew`, while retiring hidden settings reads and making every choice that can
affect results explicit. Compatibility modes reproduce inherited behaviour; corrected modes
fix known validity, positional-encoding, distance, and edge-direction defects.

Responsibility ends at validated fold samples with documented Python types and shapes.
This component does not choose stations, train a model, calculate metrics, collect raw
data, or prepare the standard rainfall dataset.

## 2. System Position

```text
02 hourly observations + station table      03 split + target list
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                         GRAPH / SAMPLE BUILDER
                                   │
                                   ▼
                         06 model adapter
```

### Worked example — fold 0, gauge-only

1. Load Component 02 observations and stations for the split's `dataset_version`.
2. Verify the split check report and exact fold-0 target-list fingerprint (hash).
3. Build the fold-0 train, validation, and test station graphs.
4. Compute full-graph Laplacian positional encodings or reproduce per-split encodings,
   according to the required graph settings.
5. Fit continuous-feature statistics on train stations only.
6. Create samples whose target station rainfall is hidden while validity becomes 0.
7. Attach explicit node-index-to-station-id and target-row mappings.
8. Save the samples and a `manifest.json` record under a calculated `sample_version`.

Component 06 loads those files; it does not recreate edges or reinterpret features.
The `manifest.json` mentioned above is the sample directory's machine-readable record;
OUT03 lists its required contents and why it is needed.

## 3. Inputs

### 3.1 Settings owned by Component 04

#### IN01 — Graph/sample settings

The Component 04 owner creates and maintains
`config/graphs/gauge_baseline.yaml`. It must supply every field below; none has a silent
default because each can change numerical model input:

```yaml
schema_version: 1
k_neighbours: 5
distance_mode: haversine_km          # euclidean_degrees for inherited compatibility
edge_direction_mode: directed_knn    # legacy_undirected for compatibility
edge_normalisation: per_node         # global for inherited compatibility
lpe_mode: full_graph                 # per_split for inherited compatibility
lpe_dimensions: 4
validity_mode: preserved             # inherited_normalised for compatibility
mask_value_mode: physical_zero       # normalised_zero for compatibility
feature_normalisation: train_zscore
```

The complete settings used are copied into the sample directory.

### 3.2 Data from Components 02 and 03

#### IN10 — Standard rainfall dataset

Component 02 OUT01/OUT02/OUT05, loaded through its Python loader and checked against the
recorded `dataset_version` and file fingerprints.

#### IN11 — Accepted split and prediction target list

Component 03 OUT01/OUT02/OUT04. Sample creation refuses a failed check report, dataset
mismatch, or target-list mismatch.

#### IN12 — Project seeds

Component 03 OUT05. Used for Laplacian positional-encoding sign handling and any sample
ordering. The seed-file hash is part of `sample_version`.

A hash in this document means a SHA-256 content fingerprint. If an input file or saved
sample changes, its hash changes too.

## 4. Functionality Requirements

### FR01 — Define the gauge node and target contract

Each station node has `x` shaped `[nodes, time, features]` and `y` shaped
`[nodes, time, 1]`. These are **tensors**, meaning multidimensional arrays of numbers
used by PyTorch. Feature order is fixed:

| Channel | Name | Meaning before continuous-feature normalisation |
|---|---|---|
| 0 | `rainfall_mm` | observed rainfall; physical 0 for dry or missing/masked |
| 1 | `validity` | 1 for an available observation; 0 for missing or masked |
| 2–5 | `lpe_0..3` | four numbers describing a station's position in the graph |

- `y` is a separate tensor copied from Component 02 rainfall before target hiding.
- Timestamps and node-index-to-station-id mappings are explicit metadata.
- New features append under a new feature-layout version; existing channels never move.
- Samples contain only accepted 2021–2022 hours allowed by Component 03's target list.

### FR02 — Build role-specific station graphs

For each fold:

- train graph contains train stations;
- validation graph contains train plus validation stations;
- test graph contains train, validation, and test stations;
- gauge edges are recomputed for each role's available station set;
- every station has at most `k_neighbours` outgoing K-nearest-neighbour (KNN) edges and
  no self-edge. KNN connects a station to the configured number of closest stations;
- graph role, station ordering, node/edge counts, and target masks are recorded.

The denser evaluation graph reproduces the inherited inductive method. A future
fixed-density method requires a new labelled mode; do not change it inside this work.

### FR03 — Make the distance calculation explicit

Two modes are required:

- `euclidean_degrees`: inherited compatibility, KNN over raw longitude/latitude.
- `haversine_km`: corrected mode, great-circle distance in kilometres.

`haversine_km` is the default in corrected experiment settings. The selected mode is
required, validated, and recorded. Public functions always accept longitude followed by
latitude; conversion to radians happens inside the implementation.

**Why**

- Raw longitude/latitude degrees have unequal physical scale at Sydney's latitude.
- Retaining the inherited mode keeps historical graph comparisons possible.

### FR04 — Preserve directed KNN semantics when required

Two modes are required:

- `legacy_undirected`: reproduce the inherited NetworkX/`ToUndirected` representation.
- `directed_knn`: store every target station's KNN relation as a directed edge and, when
  both directions exist, preserve both weights independently.

The corrected per-node equation must use `directed_knn`. Reject
`edge_normalisation=per_node` with `legacy_undirected` for corrected runs; permit it only
when `manifest.json` labels it `legacy_per_node_overwrite` for regression testing of the
2026-08-31 run.

### FR05 — Calculate and normalise inverse-distance edge weights

Raw weight for edge `source → target` is `1 / distance`, with an explicit minimum
distance guard recorded in the graph settings.

Required modes:

- `global`: divide every role graph's weights by its single maximum; inherited.
- `per_node`: divide each target node's incoming weights by that target's maximum;
  paper Equation 1.

In corrected `per_node + directed_knn` mode, every node with neighbours has maximum
incoming weight exactly 1. Store raw/normalised distribution summaries and zero-distance
guards in `manifest.json`.

### FR06 — Construct and apply the validity channel

Required modes:

- `preserved`: validity remains exactly 0 or 1 and bypasses z-score normalisation.
- `inherited_normalised`: reproduce the inherited all-channel normalisation defect.

`preserved` is required for corrected experiments. A missing observation and masked
target both have validity 0; a genuine dry observation has rainfall 0 and validity 1.
The implementation checks these meanings before saving a `.pt` file.

### FR07 — Define masking in physical and normalised space

Required modes:

- `physical_zero`: set target rainfall to physical 0 and validity to 0 **before**
  applying train-fitted continuous statistics. Corrected mode.
- `normalised_zero`: z-score first, then replace rainfall and validity with numeric 0.
  Inherited compatibility; numeric 0 corresponds to the training mean.

Mask only feature channels 0–1. LPE remains visible. Every sample contains an explicit
boolean target mask and the exact target-list rows it represents.

### FR08 — Describe each station's graph position consistently

A **Laplacian positional encoding (LPE)** gives each station a small set of numbers
derived from the graph connections. It helps a model distinguish where a station sits
in the network even when stations have similar rainfall features.

Required modes:

- `full_graph`: compute four LPE columns once on the all-station graph for the fold's
  configured topology, apply deterministic sign orientation, then subset by station id.
- `per_split`: reproduce the inherited behaviour by computing LPE independently on each
  role graph.

`full_graph` is corrected mode. It exposes held-out station coordinates/topology but
never held-out rainfall. Record this use of held-out station positions in
`manifest.json`.

### FR09 — Fit feature normalisation on training data only

- Fit means/standard deviations only from the fold's train graph and available values.
- Apply unchanged statistics to validation/test.
- In `preserved` mode, exclude validity from fitted channels.
- Prevent zero variance with a documented epsilon and report affected channels.
- Save statistics by fold and feature name, not as unexplained array positions.

### FR10 — Enforce leakage guards

**Required**

- `x` and `y` do not share mutable storage.
- Before a sample is returned, assert every target row has rainfall hidden and validity
  0 in `x`, while `y` retains Component 02 rainfall truth.
- Assert no test station is present in a train graph.
- Mutation test: changing a held-out station's Component 02 rainfall must not change any
  training graph feature or normalisation statistic.
- Fail on a target-list row that cannot be represented; never drop it.

### FR11 — Carry identity and time through batching

Each graph release contains:

- ordered node `station_id` list per role;
- ordered UTC timestamp list or range/index reference;
- batch/sample target mapping to `(fold, station_id, timestamp)`;
- target-list hash.

PyTorch Geometric batching and unbatching must preserve or reconstruct these mappings
without relying on a positional id as a station identity.

### FR12 — Produce reproducible sample releases

Derive `sample_version` from dataset/split ids, graph settings, seed-file hash, feature
layout, code revision, fitted statistics, and saved-sample hashes. Creating samples
again from the same inputs produces equivalent tensors and the same sample record.
Existing different content is never overwritten.

### FR13 — Preserve the validated inherited regression combination

The compatibility combination is:

```text
distance_mode=euclidean_degrees
edge_direction_mode=legacy_undirected
edge_normalisation=global
lpe_mode=per_split
validity_mode=inherited_normalised
mask_value_mode=normalised_zero
```

On the golden fixture, it must match the validated transitional pipeline's node/edge
counts, feature tensors, masks, and weight summaries within declared numeric tolerance.

### FR14 — Provide the corrected gauge settings

The corrected combination is:

```text
distance_mode=haversine_km
edge_direction_mode=directed_knn
edge_normalisation=per_node
lpe_mode=full_graph
validity_mode=preserved
mask_value_mode=physical_zero
```

This mode must pass all integrity tests before it is used for a scientific comparison.
No result is labelled corrected if only a subset of these switches was applied; partial
combinations are named explicitly.

### FR15 — Leave future modalities attachable but absent

The gauge-only milestone creates no dummy radar or satellite nodes or edges. The public functions may later accept source-specific node
builders, but their required fields and shapes need an approved revision after the NCI
products are known.

## 5. Non-negotiable Rules

1. The split file, not graph code, decides station roles and target rows.
2. No held-out rainfall reaches a model input or train-fitted statistic.
3. Every graph choice that can affect results is required in the settings and recorded
   in `manifest.json`.
4. Validity 0/1 is preserved in corrected mode.
5. Corrected per-node weights preserve direction; undirected overwrite is not accepted.
6. `x`, `y`, station identity, target identity, and UTC time remain separately traceable.
7. No synthetic 1D/2D data is created for a gauge-only run.
8. Compatibility and corrected sample sets have distinct `sample_version`s and labels.
9. Samples use only accepted 2021–2022 Component 02 data and a passed Component 03 split.

## 6. Outputs

Each release lives under:

```text
database/graph_samples/<sample_version>/
  manifest.json
  settings-used.yaml
  normalisation.json
  fold-0/train.pt
  fold-0/validation.pt
  fold-0/test.pt
  fold-0/stations.json
  ...
```

A `.pt` file is PyTorch's saved-data format for tensors and graph data. Record the exact
Python, PyTorch, and PyTorch Geometric versions used to write it. The loader may recreate
samples instead of opening an incompatible `.pt` file, but it must verify the same
`sample_version` and tensor summaries.

### OUT01 — Graph/sample creation functions

`src/graphs/` exposes type-hinted `create_samples(dataset_version, split_id, settings_path)`,
`load_fold(sample_version, fold, role)`, and sample iteration interfaces.

### OUT02 — Saved graph samples for each fold

Role-specific saved gauge-only data with explicit features, targets, edges, masks,
station ids, timestamps/target mapping, and fitted-statistic references.

### OUT03 — Sample-set record (`manifest.json`)

A **manifest** is a machine-readable record for the complete sample directory. Here,
`manifest.json` records the sample version, Component 02/03 versions and hashes, graph
settings, seed hash, feature layout, graph modes, station order, node/edge/target counts,
weight summaries, normalisation hashes, library versions, code revision, and hashes of
the saved output files.

### OUT04 — Normalisation statistics

`normalisation.json` stores per fold/type/feature training mean, standard deviation,
count, epsilon use, and excluded channels.

### OUT05 — Mode comparison report

`MODE_REPORT.md` records fixture evidence for compatibility and corrected modes,
including the legacy per-node overwrite regression if retained.

## 7. Testing

### 7.1 Component-level acceptance tests

#### TEST01 — Release round trip

Build, load, and validate every fold/role. Rebuilding from the same inputs produces the
same sample version, `manifest.json`, tensor summaries, mappings, and target counts.

#### TEST02 — Reject invalid Component 02/03 inputs

Failed split check, mismatched dataset, changed target hash, ineligible station, missing
Component 02 row, and wrong UTC type each fail before graph construction.

#### TEST03 — Exact target handoff

The union of sample target mappings equals component 03 `targets.parquet` exactly—no
extra, missing, or duplicate `(fold, station_id, timestamp)` rows.

#### TEST04 — Leakage mutation drill

Changing held-out rainfall does not alter train tensors/statistics. Shared `x`/`y`
storage and an unmasked target are each detected by planted-fault tests.

#### TEST05 — Identity through batching

Batch/unbatch known samples and prove predictions can be joined back to exact internal
station ids and UTC timestamps without positional assumptions.

### 7.2 Mode-specific tests

#### TEST10 — Distance modes

Known coordinate pairs produce expected haversine kilometres; inherited degree mode
matches the regression fixture and yields the documented Sydney topology differences.

#### TEST11 — Edge direction and normalisation

Corrected directed mode preserves reciprocal edges with independent weights; every
node's maximum incoming per-node weight is 1. Global mode matches the fixture defect.

#### TEST12 — Validity and masking

Corrected masked, missing, dry, and wet examples have distinct required semantics.
Inherited modes reproduce validity collapse and normalised-zero behaviour for regression.

#### TEST13 — LPE consistency

`full_graph` gives the same station the same oriented LPE across roles; `per_split`
reproduces and reports divergence.

#### TEST14 — Regression anchor

FR13 matches frozen fixture tensors and summaries. Any accepted difference updates the
fixture through component 05's expectation-review workflow.

#### TEST15 — Corrected configuration gate

Invalid combinations—especially `per_node + legacy_undirected` without its explicit
legacy label—are rejected.

## 8. Documentation Requirements

All documents are Markdown under `docs/graphs/`.

### DOCS01 — Graph shapes and feature guide

Explain shapes, features, targets, masks, node/edge types, mappings, and saved `.pt` files
with a five-station, two-hour worked example.

### DOCS02 — PyTorch Geometric walkthrough

Explain how Component 02 tables become PyTorch Geometric `Data` values and batches for
a reader who has never used the library. Define every library-specific class on first
use.

### DOCS03 — Science-bearing mode guide

Explain every settings field, inherited/corrected combinations, distance decision,
per-node direction requirement, validity/masking semantics, LPE trade-off, and labels.

### DOCS04 — Build/load runbook

Commands to build, validate, inspect, compare modes, load a fold, and hand it to
component 06.

### DOCS05 — Sample changelog

One entry per sample version with input dataset/split ids, mode combination, reason, and
previous version.

## 9. Dependencies / Blockers

- Requires accepted component 02 and 03 releases.
- Production corrected graphs are blocked until regenerated BoM standard data exists.
- Gauge-only implementation does not wait for NCI.
- Radar/satellite graph fields and shapes are blocked until their source data and
  Component 02 formats are validated.
- Changes to role graph density or masking rules require a revised split/model study,
  not an unlabelled graph refactor.

## 10. Current Technical Findings

- `GaugeGraphNew` can execute with only `raingauge` nodes; top-level drivers, not model
  mathematics, were the original blocker.
- The existing class reads root `config.yaml` at import time and mixes graph building,
  feature construction, normalization assumptions, and dataset iteration. New code must
  receive explicit arguments and avoid import-time config reads.
- Both inherited implementations use raw-degree KNN and graph-wide edge normalization.
- The existing opt-in `per_node` implementation is vulnerable to direction-dependent
  weights being overwritten in an undirected NetworkX graph.
- The inherited graph code currently z-scores validity, masks after normalization, and
  computes LPE per role graph. These are inherited behaviours, not corrected requirements.
- Existing predictions carry positional station indexes; FR11 replaces that boundary.

## 11. Resources / Glossary

- [GLOSSARY.md](GLOSSARY.md) defines KNN, LPE, masking, validity, PyTorch Geometric,
  edge normalization, and leakage.
- Previous component requirements:
  [02_CANONICAL_DATA_REQUIREMENTS.md](02_CANONICAL_DATA_REQUIREMENTS.md) and
  [03_STUDY_SPLIT_REQUIREMENTS.md](03_STUDY_SPLIT_REQUIREMENTS.md).
- Inherited core: `src/graph/gaugegraphnew.py` and the transitional `train_fused.py`.
- Evidence: the passing graph regression tests and
  [../archive/baseline-reconciliation/AUDIT_NIRAMAY.md](../archive/baseline-reconciliation/AUDIT_NIRAMAY.md).
