# Glossary

Plain-language definitions for every domain term and acronym used in the requirements
documents. Alphabetical within theme.

## Rainfall data & BoM

- **BoM** — Australia's Bureau of Meteorology, the national weather agency and the
  source of our gauge data.
- **WaterData** — BoM's public portal for water observations (rainfall, river levels…):
  http://www.bom.gov.au/waterdata/
- **KiWIS** — "Kisters Web Interoperability Solution", the HTTP API behind WaterData.
  You call URLs like `…/waterdata/services?service=kisters&type=queryServices&request=X`
  and get JSON back. The three requests we use: `getStationList` (find stations),
  `getTimeseriesList` (find data series for a station), `getTimeseriesValues` (download
  the values).
- **Rain gauge / TBRG** — "Tipping Bucket Rain Gauge". A funnel fills a tiny see-saw
  bucket; each time it tips, a fixed depth of rain (here **0.5 mm**) has fallen and a
  timestamped "tip" record is logged. Rainfall between tips is not measured — you know
  0.5 mm fell at each tip time, nothing more granular.
- **Tip** — one such record: a timestamp plus the value 0.5 (mm).
- **station_no vs station_id** — BoM stations have a *public number* (`station_no`,
  e.g. `563083`) shown on websites, and an *internal id* (`station_id`, e.g. `7730155`)
  used inside their database. **Public numbers are not unique** — we found two real,
  distant sites sharing `563083` and two sharing `567163`. Internal ids ARE unique.
  That's why the project keys everything on internal id (RULE04).
- **Time series / ts_id** — one stream of measurements for one station. A station has
  many (raw, quality-controlled, daily totals…), each with its own id (`ts_id`).
- **AsStored** — a series containing the raw stored records (for rainfall: the
  individual tips, irregular timestamps), as opposed to aggregated series.
- **DMQaQc** — "Data Management Quality assured / Quality controlled" — the prefix BoM
  puts on series that have been through their full quality-control process. We use
  `DMQaQc.Merged.AsStored` (quality-controlled tip records) as the trusted source.
  Other prefixes you'll see: `Received` (raw as telemetered), `Harmonised`
  (standardised), `Provisional` (not yet fully checked).
- **DailyTotal.09HR** — a BoM series of daily rainfall totals for the standard
  "rain day" ending 9am local time. We use it as an independent cross-check: our summed
  tips for a day must match BoM's own daily total.
- **QC (quality control)** — automated/manual checks a data provider runs to remove
  sensor glitches (stuck sensors, impossible spikes) before publishing.

## Time & coordinates

- **UTC** — Coordinated Universal Time, the global reference clock (no daylight
  saving). Canonical timestamps in this project are UTC (RULE03).
- **AEST / AEDT** — Australian Eastern Standard Time (UTC+10) / Daylight Time (UTC+11,
  summer). BoM's API serves fixed UTC+10 year-round. The inherited dataset used naive
  local clock time, which jumps an hour at daylight-saving transitions — banned.
- **DST seam** — the discontinuity in a time series at a daylight-saving changeover.
- **Hour-beginning convention** — an hourly value stamped `14:00` covers 14:00–15:00.
- **WGS84** — the standard latitude/longitude coordinate system (what GPS uses).
- **CRS** — Coordinate Reference System; WGS84 is one.
- **EPSG:3577 / Australian Albers** — a coordinate system measured in metres and
  designed for Australian mapping. The corrected split protocol uses it so east-west
  and north-south distances have comparable physical meaning.

## Datasets & experiments

- **Standard dataset / canonical data** — the single cleaned, checked dataset all
  components read (Component 02's output). "Canonical" is retained in code paths such as
  `database/canonical/`; it simply means "the one authoritative version".
- **Raw source data** — files saved by Component 01 before scientific preprocessing.
  The requirements use this concrete phrase instead of the data-platform term "landing".
- **Source-faithful data** — source records may be parsed, compressed or packaged for
  reliable storage, but their scientific values and meaning are not transformed.
  Hourly aggregation, interpolation, normalisation and feature engineering are not
  source-faithful ingestion operations.
- **Acquisition id** — the stable name of one attempt to collect a specified region,
  period, and source set, for example `sydney_2021_2022_v1`.
- **Manifest** — the conventional filename `manifest.json` used for a machine-readable
  dataset/sample/run record. It lists what the directory contains, which inputs and
  settings produced it, row counts, versions, and file hashes. Each component document
  explains why its own manifest is needed.
- **dataset_version / split_id** — short identifier strings for "which exact data" and
  "which exact split" a result used. Recorded everywhere so results are traceable.
- **sample_version** — the version id of graph samples created from a particular dataset,
  split, graph settings, and seeds.
- **experiment_id / run_id** — `experiment_id` identifies one method/settings/input
  combination; `run_id` identifies one concrete execution attempt that
  may be resumed.
- **Content hash / SHA-256** — a deterministic fingerprint of bytes. If content changes,
  the hash changes; manifests use hashes to detect corruption and hidden revisions.
- **Acquisition file list / target list** — machine-readable rows defining,
  respectively, every raw file expected from Component 01 and every held-out
  station-hour a method must predict. The actual filenames retain `index` for brevity.
- **Split / fold** — the division of stations into training / validation / test sets.
  With 5-fold cross-validation there are 5 such divisions ("folds"); each station
  ideally appears in exactly one fold's test set.
- **Spatial cross-validation** — folds split by *station* (place), not by time: the
  model is tested on gauges it never saw, at locations it never saw.
- **Leakage** — any path by which information about a held-out target sneaks into
  training (e.g. the same station appearing in both train and test under duplicate
  metadata — a real inherited bug we fixed).
- **Golden fixture** — a small frozen dataset with known-correct expected outputs, used
  to test the whole system quickly and catch regressions.
- **Held-out** — data hidden from the model during training and used only to test it.

## Graphs & the model

- **Node / edge** — stations are graph *nodes*; *edges* connect nearby stations so the
  model can pass information between them.
- **KNN** — "k nearest neighbours": each station connects to its k closest stations
  (project default k=5).
- **Edge weight** — a number on each edge scaling how much influence flows across it;
  here 1/distance, then normalised (see next).
- **Edge-weight normalisation, `global` vs `per_node`** — `global` divides every weight
  by the single largest weight in the whole graph (inherited behaviour); `per_node`
  divides each station's incoming weights by that station's own largest (the paper's
  Eq. 1). Both are implemented; which one a run uses must always be stated.
- **Validity flag** — a 0/1 feature per station-hour meaning "this reading exists".
  Distinguishes a genuinely dry hour (value 0, flag 1) from a missing/hidden one
  (value 0, flag 0). An inherited bug destroys it — see doc 04 FR04.
- **Masking** — hiding a station's value from the model and asking it to predict it;
  how the model learns to estimate rainfall at unobserved places.
- **LPE (Laplacian Positional Encoding)** — four numbers per node computed from the
  graph's structure, acting as a "structural address" so the model can tell nodes
  apart. Must be computed consistently between train and test graphs — see Component 04
  FR08.
- **z-score normalisation** — rescaling a feature to mean 0, standard deviation 1,
  using statistics from the *training* stations only.
- **PyG / HeteroData** — PyTorch Geometric, the graph deep-learning library;
  `HeteroData` is its container for graphs with multiple node/edge types.
- **HGNN** — Heterogeneous Graph Neural Network, the inherited model family.
- **Checkpoint** — a `.pt` file containing model, optimiser, and random state so training
  can resume or predictions can be reproduced without training again.
- **Prediction table** — `predictions.parquet`, containing one prediction row per
  held-out station-hour with the required columns in Component 06 FR06. Every method
  writes one.

## Metrics & baselines

- **RMSE** — root mean square error; lower is better; same units as the data (mm).
- **Pearson r** — linear correlation between predictions and truth, −1…1; ~0 means the
  model learned nothing; unit-free.
- **IDW** — Inverse Distance Weighting, the classical interpolation baseline: estimate
  a location as the distance-weighted average of nearby gauges.
- **Kriging / KED** — geostatistical interpolation baselines (KED = Kriging with
  External Drift, i.e. with auxiliary predictors).

## Infrastructure

- **NCI / Gadi / PBS** — Australia's National Computational Infrastructure; Gadi is its
  supercomputer; PBS is Gadi's job-queue system. Radar/satellite products live in NCI
  project storage (`rq0`, `rv74`).
- **Rainfields3** — BoM's operational radar rainfall product (note: *gauge-corrected*,
  which matters for leakage — see doc 01 FR09).
- **Himawari-8/9 CRRPH** — the satellite rainfall product used by the paper
  (Convective Rain Rate — Physically based, from the Himawari geostationary satellite).
- **CI** — continuous integration: tests that run automatically on every change.
