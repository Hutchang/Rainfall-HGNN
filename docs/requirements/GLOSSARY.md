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
  That's why the project keys everything on internal id (rule 3 in the README).
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
  saving). Canonical timestamps in this project are UTC (rule 2 in the README).
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
  components read (Component 02's output, in `database/standard/`). "Canonical" survives
  in the package name `src/canonical/`; it just means "the one authoritative version".
- **Raw source data** — files saved by Component 01 before scientific preprocessing.
  The requirements use this concrete phrase instead of the data-platform term "landing".
- **Source-faithful data** — source records may be parsed, compressed or packaged for
  reliable storage, but their scientific values and meaning are not transformed.
  Hourly aggregation, interpolation, normalisation and feature engineering are not
  source-faithful ingestion operations.
- **Index (`index/<fetcher>.csv`)** — Component 01's list of every raw file it wrote,
  with coverage, status and content hash; one file per fetcher (`index/gauge.csv`,
  `index/radar.csv`). Component 02 reads it instead of listing the directory, so a failed
  download cannot be mistaken for a month with no rain.
- **Content hash / SHA-256** — a deterministic fingerprint of bytes. If content changes,
  the hash changes. Recorded in `index.csv` so you can tell whether a changed result came
  from changed data.
- **Target rows** — the station-hours a method must predict: every row of
  `gauge_hourly.csv` belonging to a fold's test stations. Defined by that rule rather
  than by a file, so two people always get the same set.
- **Split / fold** — the division of stations into training / validation / test sets.
  A *fold* is one round of the test: some stations are held out and predicted, the rest
  are used for training. With 5-fold cross-validation there are 5 such rounds; each
  station is held out in exactly one of them.
- **Blocked fold** — a fold whose held-out stations are whole areas of the map, so a
  held-out station has no training station right next to it. The opposite — folds that
  each take a few stations from every area — makes prediction easy because a neighbour a
  few kilometres away is always in training. The paper uses blocked folds; closing that
  gap is D08.1.
- **Spatial cross-validation** — folds split by *station* (place), not by time: the
  model is tested on gauges it never saw, at locations it never saw. Done with blocked
  folds so that "never saw" also means "nothing right next to it".
- **Visible gauge** — a gauge a method is allowed to read when predicting a held-out
  station. Which gauges are visible (training only, or training plus validation) must be
  the same for every method being compared; closing that gap is D08.2. Older documents,
  and the archived
  Sprint 1 contracts, call these *donors*; the two words mean the same thing.
- **Validation gauges** — the stations set aside, inside the training side of a fold,
  whose values are used only to decide when to stop training. They are not test
  stations.
- **Correlation length** — how far apart two gauges can be before their rainfall stops
  looking alike. Measured by plotting the correlation of every pair's hourly series
  against their distance and reading off where it falls away. The paper's explanation of
  its Sydney result rests on Sydney's gauge spacing being small compared with this
  length.
- **Leakage** — any path by which information about a held-out target sneaks into
  training (e.g. the same station appearing in both train and test under duplicate
  metadata — a real inherited bug we fixed).
- **Test sample** — the few kilobytes of real data each component commits under
  `tests/data/` so its tests run on a fresh clone with no network.
- **Held-out** — data hidden from the model during training and used only to test it.
- **Component** — one of the numbered parts of the pipeline, each with its own code
  (`src/<name>/`), config (`config/<name>/`), docs and owner: 01 fetches raw data, 02
  builds the standard tables, 03 picks which stations are held out, 04 builds the
  graphs, 06 trains and predicts. There is no 05 (cut). A *run* is Components 02, 03, 04
  and 06 in order, each reading the files the one before it wrote; Component 01 is not
  part of a run — it fills the shared raw cache once, ahead of any run. Component 02 has
  two programs (gauge tables and, when radar is on, grid tables), which the runner
  treats as one step.
- **Run** — one execution of the pipeline from the config files as they were at that
  moment. Every run gets its own folder under `experiments/`, named by date and time
  plus an optional label (`2026-09-19_1432_blocked-folds`), holding copies of those
  config files; the folder is never written to again (D07). The folder name is the
  run's id in the registry.
- **Fingerprint (hash, SHA-256)** — a short string computed from a file's bytes. The
  same bytes always give the same string; any change to the file gives a different one.
  Recorded so a run can later prove it read exactly this file, and so a rerun can tell
  when a file has been rewritten since.
- **Experiment file** — `config/experiments/<name>.yaml`: five paths, one per component
  config, plus an optional label. It names *which* config files a run uses and holds no
  settings of its own; the runner copies it and the five files into the run folder. Two
  runs differ first by which files they named, then by the keys inside those files.
- **Provenance record** — `provenance.json` in a run folder: for each component, every
  input file and its fingerprint, every output file and its fingerprint, and the code
  revision.
  Written once per run. It is how a result is tied to exactly the data and code that
  produced it.
- **Manifest (per source)** — a file written once per raw data source declaring what
  that source *is*: units, coordinate system, time step, covered period, and where it
  came from (ISS-24). Not the same thing as the provenance record: the manifest says
  what the source is, the record says what one run used.
- **Registry** — `experiments/registry.csv`: one row per run, never edited, so the list
  of every experiment ever run is one file.
- **Component cache** — a store of component outputs keyed by the fingerprint of the
  component's settings and inputs, so a component that has already been run with the
  same settings on the same data is reused rather than run again (D07, backlog).
- **Region** — one named study area with its own bounding box and, later, its own radar
  site: `config/ingestion/<region>.yaml`. Sydney is the first. D07 and D10 make the
  pipeline work per region without hard-coded Sydney paths.
- **Deliverable / subtask / DEC** — Sprint 2's units of work. A *deliverable* is one
  line of work with its own requirements document, numbered on from the Sprint 1
  components (D07–D13); a *subtask* is one piece of a deliverable (D08.1, D09.2…) with
  its own "done when"; a *DEC* is a decision recorded in the requirements README.

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
  (value 0, flag 0).
- **Scaling (z-score)** — before training, rescale an input so its values sit roughly
  between −3 and 3: subtract the training set's average and divide by its spread
  (standard deviation). Done so no one input dwarfs the others. The paper does it; D08.2
  says how we do it.
- **Masking** — hiding a station's value from the model and asking it to predict it;
  how the model learns to estimate rainfall at unobserved places.
- **LPE (Laplacian Positional Encoding)** — four numbers per node computed from the
  graph's structure, acting as a "structural address" so the model can tell nodes
  apart. They are kept visible when a node's rainfall is hidden, so the model knows which
  node it is predicting. Must be computed consistently between train and test graphs —
  Computed per graph by design. The sign of each one is drawn at random at build time,
  which is ISS-27.
- **PyG / HeteroData** — PyTorch Geometric, the graph deep-learning library;
  `HeteroData` is its container for graphs with multiple node/edge types.
- **HGNN** — Heterogeneous Graph Neural Network, the inherited model family.
- **Checkpoint** — a `.pt` file containing model, optimiser, and random state so training
  can resume or predictions can be reproduced without training again.
- **Prediction table** — `predictions.csv`, containing one prediction row per
  held-out station-hour with the columns in `src/models/contracts.py`. Every method
  writes one.
- **Weighted MSE / α** — the training loss: each squared error is multiplied by
  `1 + α × log(1 + rainfall)`, so hours with heavy rain count for more. α = 0 is plain
  mean squared error. We use α = 0.18, the paper's first author's own value; the paper
  itself never gives one.

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
  which matters for leakage — see D09).
- **prcp-m15 / prcp-crate** — two Rainfields3 products. `prcp-m15` is rain
  *accumulated* over each 15-minute window, corrected using nearby gauges; `prcp-crate`
  is an instantaneous rain *rate*, not corrected. We fetch `prcp-m15`.
- **Contributing gauges** — the list, inside each `prcp-m15` file, of the BoM gauges
  used to correct that file. A held-out test station in that list means the radar input
  already contains its reading; D09 states that caveat on every radar result.
- **Accumulation vs rate** — an accumulation is millimetres that fell during a window
  (sum four 15-minute windows to get an hour); a rate is millimetres per hour at an
  instant. They are not interchangeable, and a column named `rain_rate` is not proof of
  either.
- **Himawari-8/9 CRRPH** — the satellite rainfall product used by the paper
  (Convective Rain Rate — Physically based, from the Himawari geostationary satellite).
- **CI** — continuous integration: tests that run automatically on every change.
