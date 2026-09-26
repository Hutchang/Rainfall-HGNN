# Known issues

Outstanding bugs and gaps. An entry here is a **recorded, deliberate
choice**: either it is not worth fixing yet given time and materiality, or it is blocked on
someone else's answer. Anything not written down here is not deferred — it is forgotten.

This is the only place issues are recorded. [CURRENT_STATE.md](CURRENT_STATE.md) describes
project state and sprint progress and cites `ISS-xx` ids; component READMEs under `docs/`
may cite ids too. Neither restates an issue's severity, owner or status — there would then
be two answers and no way to tell which is current. If you find an issue written up
somewhere else, move it here and leave the id behind.

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

## Index (By Severity)

| ID | Title | Severity | Owner | Status |
|---|---|---|---|---|
| [ISS-01](#iss-01) | Short gauge blockages produce confident false zeros | Blocking | Component 02 | Open |
| [ISS-02](#iss-02) | Radar and satellite provenance unknown | Blocking | D13 | Closed |
| [ISS-03](#iss-03) | Satellite node coordinates do not exist | Blocking | D13 | Open |
| [ISS-07](#iss-07) | Methods are allowed to read different gauges when predicting | Blocking | Components 04 + 06 | Open |
| [ISS-11](#iss-11) | Ingestion config claims Sydney but fetches Australia-wide | Blocking | Component 01 | Closed |
| [ISS-30](#iss-30) | Fetched radar files cannot be read by the grid adapter | Blocking | D13 | Closed |
| [ISS-04](#iss-04) | Radar and satellite cover 2022 only; radar missing 120 hours | Material | D13 | Open |
| [ISS-05](#iss-05) | Within-day timestamp fidelity is assumed, not verified | Material | Component 02 | Blocked |
| [ISS-06](#iss-06) | Unexplained April–May 2021 gap in BoM daily totals | Material | Component 02 | Open |
| [ISS-08](#iss-08) | IDW exponent is unprescribed | Material | _unassigned_ | Open |
| [ISS-09](#iss-09) | No reversible normalization metadata in the graph contract | Material | Components 04 + 06 | Open |
| [ISS-10](#iss-10) | Spatial-block cross-validation decision unresolved | Material | _unassigned_ | Open |
| [ISS-12](#iss-12) | Plausibility caps, tolerance and coverage threshold need a client decision | Material | _unassigned_ | Blocked |
| [ISS-14](#iss-14) | Inherited-baseline graph fingerprint test missing | Material | Component 04 | Open |
| [ISS-15](#iss-15) | No `.gitattributes`; NCI guard tests fail on Windows checkouts | Material | _unassigned_ | Open |
| [ISS-16](#iss-16) | Failed raw downloads are excluded but not reported | Material | Component 02 | Open |
| [ISS-18](#iss-18) | Component 03 split summary and plot are not produced | Material | Component 03 | Open |
| [ISS-19](#iss-19) | Component 03 has no component README | Material | Component 03 | Open |
| [ISS-20](#iss-20) | Canonical README's Sydney statistics table is stale | Material | Component 02 | Open |
| [ISS-21](#iss-21) | `docs/requirements/README.md` is empty but three documents route readers to it | Material | Requirements | Closed |
| [ISS-22](#iss-22) | "Greater Sydney" scope for the upper Blue Mountains is undecided | Material | D08 | Closed |
| [ISS-24](#iss-24) | No per-source data manifest for units, CRS and provenance | Material | _unassigned_ | Open |
| [ISS-25](#iss-25) | Gauge edges run one way per pick; some gauges receive from no gauge | Material | Component 04 | Closed |
| [ISS-27](#iss-27) | Positional encoding may differ for the same station between train and test graphs | Material | Component 04 | Open |
| [ISS-28](#iss-28) | Component 02 silently overwrites `database/standard/` | Material | D07 | Open |
| [ISS-29](#iss-29) | Fetch keeps a stale station list and rewrites the index wholesale | Material | D07 | Open |
| [ISS-32](#iss-32) | Fold count is hard-coded and nothing below Component 01 knows about regions | Material | D07 | Open |
| [ISS-33](#iss-33) | Test coverage is not measured | Material | D07 | Open |
| [ISS-34](#iss-34) | Validation gauges are placed at random and can sit next to a hidden gauge | Material | _unassigned_ | Open |
| [ISS-13](#iss-13) | `train_homo.py` and `train_st.py` cannot import | Immaterial | _unassigned_ | Open |
| [ISS-17](#iss-17) | `ruff check .` reports 692 findings in inherited files | Immaterial | _unassigned_ | Open |
| [ISS-23](#iss-23) | The paper's exact 75-station Sydney list is unavailable | Immaterial | _unassigned_ | Open |
| [ISS-26](#iss-26) | Weighted-loss α = 0.18 was an unrecorded choice | Immaterial | Component 06 | Closed |
| [ISS-31](#iss-31) | `train_fused.py` raises `TypeError` as committed | Immaterial | _unassigned_ | Open |

---

## Scientific correctness

<a id="iss-01"></a>

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

<a id="iss-02"></a>

### ISS-02 — Radar and satellite provenance unknown
**Severity:** Blocking · **Owner:** D13 · **Status:** Closed 2026-09-26 (D13.4)

Which product, which units, and whether the radar is gauge-corrected are all unknown. If it
was corrected using gauges that include our test stations, every radar result leaks and is
not publishable.

The earlier source investigation found that the inherited Rainfields `prcp-m15` stream
merges gauges. Holding a gauge out of the graph does not remove information already
blended into the input, so if the supplied file comes from that stream the leak is real
rather than theoretical.

No longer waiting on a person: the archive files carry the answers. The radar fetcher
confirmed from real files that `prcp-m15` is a 15-minute accumulation in millimetres,
gauge-corrected, with the correcting gauges listed per file. The leak this implies is
accepted for now and stated on every result by D09; D13.4 reports its size.

**Closes when:** D13 writes product, units, time step, coordinate system and correction
status for both radar and satellite into the grid report, taken from the files' own
attributes.

Checked 2026-09-26 whether an uncorrected radar alternative exists in the same NCI
project (`rq0`): it does not. `rq0/level_2/` holds reflectivity and derived products, but
none is a rainfall accumulation, so using it would mean designing and validating a Z-R
conversion — new scientific work, not a parsing change. `rainfields3`'s `prcp-m15`
remains the only ready rainfall product available, and D13.2 stays targeting it, carrying
this issue's caveat. See [grids/RADAR_SOURCE.md](grids/RADAR_SOURCE.md) for the full
comparison.

D13.2 (2026-09-26) closes the radar half of this issue's "Closes when" line: `grids.yaml`
now declares `product`, `units`, `timestamp_convention` and `gauge_correction` for radar
from the archive files' own attributes, with no `UNKNOWN` remaining (see ISS-30, closed).
D13.3 does the same for satellite.

**Closed 2026-09-26 (D13.4).** `grid_report.md` now states coordinate system and native
time step for both sources too, computed from the files themselves
(`grid_mapping_name`, and `valid_time - start_time` for radar / the gap between
consecutive `nominal_product_time` values for satellite) — the last two fields this
issue's "Closes when" line named. Confirmed against synthetic fixtures for both sources;
real-archive verification still needs Gadi access, tracked separately.

<a id="iss-03"></a>

### ISS-03 — Satellite node coordinates do not exist
**Severity:** Blocking · **Owner:** D13 · **Status:** Open

The satellite input Sprint 1 planned around was a hand-exported pickle from the previous
team: an array of 726 grid points per hour (rain value and a valid flag each) for the
Sydney area, with no file saying where those points are. The 726 are whoever's subsample
of the Himawari 2 km grid, and the choice was never written down, so the file cannot be
recreated. Without it the nodes cannot be placed on the map or connected to gauges, and
satellite fusion cannot run.

The way out is not to find the file but to stop using the export: the archive product
itself (Himawari CRRPH on NCI, project `rv74`) lets coordinates be derived for every
cell from the file's own projection metadata, so coordinates come from the source and
the subsample becomes a rule we write down.

Checked 2026-09-26, project `rv74` access confirmed and the archive browsed via the
public THREDDS catalog and OPeNDAP metadata (no Gadi shell needed): the file does not
carry a stored `lat`/`lon` array — the correction to the line above — but its CF grid
mapping (`+proj=geos`, the geostationary projection) is exactly the same situation as
radar's Albers grid, decodable with `pyproj` the same way
`src/ingestion/radar/coords.py` already does. The grid is the full 5500x5500 Himawari
disk, not pre-cropped to Sydney the way radar's 512x512 already was, so D13.3 needs a
crop-to-study-area step radar didn't. See
[grids/SATELLITE_SOURCE.md](grids/SATELLITE_SOURCE.md) for the full investigation,
including the `crr`-vs-`crrph` and `intensity`-vs-`accum` decisions.

**Closes when:** D13.3 reads the satellite product from the archive and writes
`satellite_nodes.csv` with coordinates taken from the files.

<a id="iss-04"></a>

### ISS-04 — Radar and satellite cover 2022 only; radar missing 120 hours
**Severity:** Material · **Owner:** D13 · **Status:** Open

Gauges cover 2021–2022; the hand-exported radar and satellite files cover 2022, and the
radar export is missing 120 hours. The graph builder fails on missing radar values rather
than silently filling them, which is correct, but it means fusion experiments were
implicitly 2022-only.

The archive is a different matter: the radar fetcher found both 2021 and 2022 present for
site 71, so the "2022 only" limit belonged to the export. Whether the 120 missing hours
are missing in the archive too is not yet known.

**Closes when:** D13.4's grid report states, per source, the days and hours present and
missing in the archive for 2021–2022, and any remaining gap is excluded by a written rule.

D13.4 (2026-09-26) lands the mechanism: `grid_report.md` now states, per source, hours
possible/present/missing against the configured period, so the 120-hour question is
answerable by comparison once a run reads the real archive. Not yet closed: nobody has
run this against Gadi, so whether the 120 hours are actually present or absent in the
archive is still unmeasured.

<a id="iss-05"></a>

### ISS-05 — Within-day timestamp fidelity is assumed, not verified
**Severity:** Material · **Owner:** Component 02 · **Status:** Blocked

If any logger batch-reports its tips, the daily total is still correct but the hourly
distribution is wrong. Every hourly model would train on misplaced rain.

**Closes when:** BoM confirms reporting behaviour, or a detection heuristic is agreed.

<a id="iss-06"></a>

### ISS-06 — Unexplained April–May 2021 gap in BoM daily totals
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

A network-wide gap of roughly three weeks across roughly 30 stations. A per-owner outage is
the working hypothesis. Until the cause is known it is unclear whether those days are missing
or genuinely unrecorded.

**Closes when:** the cause is identified, or the period is excluded by an agreed rule.

<a id="iss-25"></a>

### ISS-25 — Gauge edges run one way per pick; some gauges receive from no gauge
**Severity:** Material · **Owner:** Component 04 · **Status:** Closed 2026-09-20, commit `4600a04`

The graph builder finds each gauge's five nearest gauges and, for gauge `i` and each
nearest `j`, adds one edge from `i` to `j` (`src/graph/gaugegraphnew.py:145-160`). Nothing
adds the reverse (`:77`). In the model, an edge from `i` to `j` means `j` receives `i`'s
value, so a gauge receives from the gauges that picked it — not necessarily from its own
nearest. A gauge that is nobody's nearest receives nothing; the builder already prints
such gauges (`src/graph/builder_new.py:188-197`). The paper connects each gauge to its own
nearest and scales weights at the receiving gauge so its nearest has weight 1.

The Sprint 1 directed rebuild fixed the weights (each direction scaled at its own
receiver) but kept one direction per pick. The paper settles whether that was intended:
sec 3.2 connects each point node to its `k` nearest neighbours, and Eq. 1 normalises at
the receiving node over `N(j)`, the set `j` is connected to. The connectivity is
symmetric; only the weights depend on direction. Directedness itself is still required —
the same pair carries weight 1.0 one way and 0.111 the other in the three-gauge example
above, which one undirected number cannot hold — so the fix is the second half of the
directed rebuild rather than a separate fix.

Applied on 2026-09-20 behind a `both_directions` setting in
`src/graph/gaugegraphnew.py`: false, the inherited one-direction behaviour, by default,
and true in `config/graphs/gauge_baseline.yaml`, so the inherited baseline stays
reproducible alongside the corrected one. On the committed four-gauge sample, one
direction per pick leaves two of the four gauges receiving nothing; both directions leaves
none, and every receiving gauge still has a strongest incoming weight of exactly 1.

**Closed 2026-09-20.** The paper connects each gauge to its own nearest and normalises
at the receiving gauge (sec 3.2, Eq. 1), so the connectivity is symmetric; our code had
one direction per pick. The defect is
fixed in code and the corrected behaviour is the one `config/graphs/gauge_baseline.yaml`
uses. One check remains and cannot run yet, because no Sydney tables exist on any machine:
rebuilding the graphs on real data and confirming the report shows no gauge without an
incoming edge. That check is D08.2's "done when" line and is tracked there, not here.

<a id="iss-26"></a>

### ISS-26 — Weighted-loss α = 0.18 was an unrecorded choice
**Severity:** Immaterial · **Owner:** Component 06 · **Status:** Closed 2026-09-19

The training loss multiplies each squared error by `1 + α × log(1 + rainfall)`. The paper
says "weighted MSE" without the weights; `config/models/hgnn.yaml:14` sets α = 0.18 with
no source. Traced: it is the paper's first author's own config value (`config.yaml:21`,
commit `dc5db56`, jylow, 2026-06-10); the function default of 1.0
(`training/logic_hetero.py:483`) is not what they ran. Recorded in D08's protocol; kept.

<a id="iss-27"></a>

### ISS-27 — Positional encoding may differ for the same station between train and test graphs
**Severity:** Material · **Owner:** Component 04 · **Status:** Open

Each gauge carries four numbers that describe where it sits in the graph's structure (the
positional encoding). They are computed from the graph, and the builder computes them
separately for the train, validation and test graphs
(`src/graph/gaugegraphnew.py:73-75, 272-290`), which hold different station sets (100, 118
and 148 stations). So the same station may get different numbers in training and in
testing. The paper keeps these numbers visible when a gauge's rainfall is hidden so the
model knows which node it is predicting; if the numbers change between graphs, that
identity does not carry over. Not yet checked by hand; this is a subagent's reading.

**Read by hand on 2026-09-20. What is now known, with nothing changed yet:**

- **The filed worry is not by itself a defect.** The paper rebuilds the graph per split on
  purpose — held-out gauges never enter the training graph's topology (sec 3.2) — and the
  model is inductive: at inference it inserts target nodes that sat in no training graph
  at all (sec 3.4). Different station sets must give different numbers.
- **The real question is the sign.** An eigenvector is fixed only up to its sign, and
  PyTorch Geometric multiplies each one by a random +1 or −1 on every call (the
  `torch.randint` at the end of `AddLaplacianEigenvectorPE.forward`,
  `add_positional_encoding.py:109-110`). It is drawn once per graph build and frozen into
  the `.pt` file, so the model trains against one arbitrary orientation for its whole run.
  The lowest-frequency eigenvectors of graphs over the same geography are approximately
  the same smooth spatial gradient, so train and test can carry opposite orientations of
  the same thing, and the model has no way to learn that the sign should not matter. The
  method the paper cites for the encoding (reference [9], Dwivedi & Bresson) re-draws the
  sign every epoch, which is what teaches that invariance — TO VALIDATE against [9].
- **The asymmetry half is already fixed.** `torch_geometric.utils.get_laplacian` uses
  the edge list exactly as given, so one direction per pick (ISS-25) gave a matrix that
  was not symmetric, whose eigenvectors can be complex, of which only the real part was
  kept. With both-direction edges the matrix is symmetric and the symmetric solver is now
  used.
- **Still unmeasured:** the gap between the fourth and fifth smallest eigenvalues on the
  real Sydney graph. Where two eigenvalues are equal, any rotation within that pair is
  equally valid and the solver's pick is arbitrary, which no sign convention fixes.

**Closes when:** the planned experiment on the legacy Sydney data has run — whether the
encoding changes the result at all, then whether the random sign costs anything — and a
sign treatment is chosen on that evidence or recorded as not worth changing.

---

## Decisions needed

<a id="iss-07"></a>

### ISS-07 — Methods are allowed to read different gauges when predicting
**Severity:** Blocking · **Owner:** Components 04 + 06 · **Status:** Open

Component 04 permits train **and validation** gauges as visible HGNN context at test time.
Component 06 specifies **training gauges only** for IDW and network mean. The implementation
honours both written contracts and flags the mismatch in every report.

Shared prediction targets alone do not make this a controlled comparison. Any claim that the
HGNN beats IDW is currently confounded by the two methods seeing different inputs.

This is D08's Gap 2 — every
method sees training and validation gauges at test time.

**Closes when:** D08.2 changes `src/models/baselines.py` to match the paper.
Source: [models/README.md](models/README.md).

<a id="iss-08"></a>

### ISS-08 — IDW exponent is unprescribed
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

`config/models/hgnn.yaml` sets `idw_power: 2.0`. That is a proposed setting, not a decision,
and not a claim to reproduce the inherited paper's IDW experiment.

**Closes when:** the exponent is agreed with the Component 03 reviewer and recorded in the
Component 06 contract, or a sensitivity sweep is run and reported.

<a id="iss-09"></a>

### ISS-09 — No reversible normalization metadata in the graph contract
**Severity:** Material · **Owner:** Components 04 + 06 · **Status:** Open

The published graph shape carries physical rainfall in mm and binary validity, but nothing
that describes a reversible transform. Component 06 accepts `input_rainfall_scale:
physical_mm` and performs no private normalization; pre-normalized graphs fail validation.
This is safe but restrictive — no one can introduce normalization without a contract change.

This is D08's Gap 3. Graph files
stay in physical mm; Component 06 scales the rainfall input on training-graph statistics
and records the two numbers in each run's config.

**Closes when:** D08.2 applies the scaling in Component 06 and records the two numbers.

<a id="iss-10"></a>

### ISS-10 — Spatial-block cross-validation decision unresolved
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

A design discussion worked through where spatial leakage actually lives in the current
k-fold split and proposed a fix, but the decision was never taken and the document was
never committed — `docs/spatial-cv-and-regional-partitioning.md` has never existed on any
ref. The reasoning therefore exists only in whoever's local notes.

What the code does today, checked by hand: `src/study/split.py:173-178` groups stations
into 10 clusters by location, then hands each cluster's stations out one at a time to
fold 0, 1, 2, 3, 4, 0, 1… so every fold holds a fifth of every cluster and every held-out
station has training stations from its own cluster a few km away. This is D08's Gap 1: each fold takes whole
clusters, as the paper's sec 4.2 requires.

**Closes when:** D08.1 changes `create_test_folds`, the new split is
released (subtask D08.1), and D08.7's measurement shows that hidden gauges no longer have
training gauges next door (D08.7 attaches the figure and the table). Whoever holds the
original design note should still commit it to `docs/archive/`.

<a id="iss-34"></a>

### ISS-34 — Validation gauges are placed at random and can sit next to a hidden gauge
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

What the code does: `assign_fold_roles` (`src/study/split.py:239-248`) shuffles the
non-test stations of each fold and takes the first 15 % as validation gauges — 18 per
fold — with no regard to where they are. What the paper says: §4.2 hides whole areas so
that held-out stations are "not adjacent to training stations"; §4.3 trains with early
stopping, so validation gauges exist, but the paper gives no rule for where they go. Why
it matters: once D08's Gap 2 is fixed every method may read the validation gauges when guessing a
hidden gauge, so a validation gauge that lands inside a hidden area is a next-door
neighbour again, and for that gauge the whole-area rule of Gap 1 is undone. How often this
happens has not been measured.

Any fix — drawing validation gauges as whole clusters, or requiring a minimum distance
from every hidden gauge — adds a rule the paper does not state. That is an extension of
the paper, not a return to it, and it is not in Sprint 2 scope. D08.7 produces the number
this issue waits for: how many hidden gauges have a validation gauge as their nearest
visible gauge, and how close.

**Closes when:** D08.7's number is recorded here and the team either accepts the effect as
small enough to leave, with the number, or opens a new DEC for a rule, labelled as going
past the paper.

<a id="iss-11"></a>

### ISS-11 — Ingestion config claims Sydney but fetches Australia-wide
**Severity:** Blocking · **Owner:** Component 01 · **Status:** Closed — not reproducible (2026-09-19)

As filed on 2026-09-18: `config/ingestion/sydney.yaml` had bounds of `-44.0..-9.0` latitude
and `112.0..154.0` longitude — the whole continent — under comments describing a Sydney
box.

Checked on 2026-09-19: the file has `-34.3..-33.4` and `150.4..151.4`, the Sydney box, and
no commit on any branch or remote ref has ever held `-44.0` in that file. The continent
box exists only as the `AUS` constant in `src/ingestion/gauge/survey.py:26`, which draws the
national map behind the Sydney box in the survey plot. Either the constant was read as the
config, or an uncommitted local edit was seen and later discarded. Either way the config
fetches Sydney, the request and time estimates in the ingestion README are right, and no
Australia-wide fetch has ever run.

A national fetch config, if wanted, is D10 work (`config/ingestion/<region>.yaml`), not
a defect in the Sydney file.

<a id="iss-12"></a>

### ISS-12 — Plausibility caps, tolerance and coverage threshold need a client decision
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Blocked

The verification tolerance is one tip increment, tight enough that 282 days fail on a single
tip's difference. The coverage threshold currently drops Gladesville at 89.85%. All of these
are config values, and all of them change which stations enter the study.

**Closes when:** the client states what they want, or the team records the defaults as its
own decision with the reasoning.

<a id="iss-22"></a>

### ISS-22 — "Greater Sydney" scope for the upper Blue Mountains is undecided
**Severity:** Material · **Owner:** D08 · **Status:** Closed — not a decision (2026-09-19)

As filed: whether Katoomba, Blackheath and the rest of the range above longitude 150.4
are inside Greater Sydney had never been decided, and the config comment said "pending a
decision from the project lead".

Closed because the fetch box is a setting, not a project-wide decision. With the
experiment runner (D07) every run records the box it used, so a run with the mountains in
and a run with them out are two labelled runs, not a choice to make once. D08 names the
box its Sydney baseline uses (the current one, matching the paper's extent); any other box
is another region under D10. The "pending a decision" comment in
`config/ingestion/sydney.yaml` should be replaced with that sentence when the file is
next touched (D07.1 or D10.2).
Source: [ingestion/README.md](ingestion/README.md).

<a id="iss-23"></a>

### ISS-23 — The paper's exact 75-station Sydney list is unavailable
**Severity:** Immaterial · **Owner:** _unassigned_ · **Status:** Open

The inherited paper reports results over 75 Sydney stations and the list of which 75 was
never supplied. Our station set is built from the BoM station list and the Component 02
rules instead, so it will not match theirs.

**Immaterial because:** the team has already decided that reproducing that table is not an
acceptance criterion. Recorded so nobody rediscovers it and treats a mismatch as a bug.
**Closes when:** the list is found, or a contract records permanently that station-level
reproduction of the paper is out of scope.

---

## Code and test gaps

<a id="iss-13"></a>

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

<a id="iss-14"></a>

### ISS-14 — Inherited-baseline graph fingerprint test missing
**Severity:** Material · **Owner:** Component 04 · **Status:** Open

The Component 04 contract requires a regression test pinning node counts, edge counts and
weights against a known `train_fused.py` run, so that a change to shared graph code cannot
move the inherited baseline unnoticed. The test that previously sat here only asserted
`median(weights) < 1e-3` — which the directed rebuild would still satisfy while changing the
edge count from 598 to 940 — so it gave no real protection and was removed on 2026-09-07.

Nothing replaced it. Shared graph code is currently unprotected against silent baseline drift.

**Closes when:** the fingerprint test exists and pins counts and weights.

<a id="iss-15"></a>

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

<a id="iss-16"></a>

### ISS-16 — Failed raw downloads are excluded but not reported
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

Failed or unread raw downloads are correctly excluded from `stations.csv` and `days.csv`, but
nothing says so. A station whose coverage looks low cannot be told apart from a genuinely dry
one without manually cross-checking `index.csv`.

No wrong data results — this is an audit-trail gap against shared rule 7, "say what you
dropped" — but a coverage-based exclusion made on unreported grounds is not defensible.

**Closes when:** the Component 02 report lists excluded raw files with reasons.

<a id="iss-17"></a>

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

<a id="iss-24"></a>

### ISS-24 — No per-source data manifest for units, CRS and provenance
**Severity:** Material · **Owner:** _unassigned_ · **Status:** Open

Every raw source carries assumptions that exist only inside the fetcher that wrote it:
what units the values are in, what coordinate basis positions are expressed against, what
the temporal resolution is, and where the data came from. Nothing declares them where a
consumer can check, so Component 02 either restates the assumption or inherits it silently.

This is the class behind several existing entries rather than a new symptom. ISS-02 and
ISS-03 are radar and satellite provenance and the missing satellite coordinates; ISS-04 is
coverage that differs per source. The radar reader assumed its projection axes were in
kilometres and multiplied by 1000 without checking: that instance is now an explicit
assertion in `src/ingestion/radar/coords.py`, so a units change raises instead of silently
scaling every node coordinate by 1000, but the next source gets no such protection for
free.

A manifest per raw source — units, coordinate reference system, temporal resolution,
covered period, and provenance including the exact product and retrieval date — would make
these declared and checkable at the boundary instead of discovered downstream. The index
already carries per-file status; this is the per-source counterpart.

Deferred deliberately: no current result is wrong because of it, and the fetchers that
exist have had their individual assumptions checked. It will bite when a fourth source is
added or when a source changes format.

**Closes when:** each raw source writes a machine-readable manifest declaring units, CRS,
temporal resolution, coverage and provenance, and Component 02 validates against it rather
than assuming. D07's per-run provenance record (what one run read) is the
counterpart of this, not a replacement; the per-source manifest (what a source is) is
still open.

<a id="iss-28"></a>

### ISS-28 — Component 02 silently overwrites `database/standard/`
**Severity:** Material · **Owner:** D07 · **Status:** Open

`src/canonical/gauge/build.py:97-136` creates the output directory with `exist_ok=True`
and writes the four tables with plain `to_csv`. A second build with different settings
replaces them with no warning. Every graph built from the old tables, and every past
model run's recorded input hashes, then point at bytes that no longer exist: the mismatch
is detectable but the old tables are gone, and any old `best.pt` can never be replayed.
The graph builder (`src/graph/builder_new.py:46-47`) and the grid adapter
(`src/canonical/grids.py:236-244`) already refuse an existing output directory; 02 is the
odd one out.

**Closes when:** the build refuses an existing output directory (D07, FR08).

<a id="iss-29"></a>

### ISS-29 — Fetch keeps a stale station list and rewrites the index wholesale
**Severity:** Material · **Owner:** D07 · **Status:** Open

Two resume behaviours in `src/ingestion/gauge/fetch.py` give a wrong result silently when
the config changes. (1) `load_or_fetch_stations` (`:110-124`) reads `gauge/stations.json`
back if it exists and never fetches again, so changing the bounding box and rerunning
keeps the old station set while the report prints the new box next to the old counts.
(2) `write_index` (`src/ingestion/common.py:65-77`) rewrites the whole index each flush
from the current run's rows, so a narrower fetch drops the rows for files an earlier,
wider fetch left on disk; Component 02 reads only `ok` index rows, so those files vanish
from the pipeline's view while still occupying disk.

**Closes when:** the station list is fetched again whenever the bounds or period differ
from those recorded beside it, and the index never drops a row for a file still on disk
(D07, FR09).

<a id="iss-30"></a>

### ISS-30 — Fetched radar files cannot be read by the grid adapter
**Severity:** Blocking · **Owner:** D13 · **Status:** Closed 2026-09-26 (D13.2)

The radar fetcher (`src/ingestion/radar/`) writes one zip per day holding 96 NetCDF files
of 15-minute accumulations on a 512 × 512 grid of 0.5 km cells (262,144 cells). The grid
adapter (`src/canonical/grids.py:98-106`) reads one pickle file of hourly values for
about 952 nodes, refuses values not on an hourly boundary (`:15-25`) and refuses a
coordinate file whose row count does not match (`:138-143`). Nothing decodes the zips,
nothing sums four 15-minute files into an hour, and nothing subsamples the grid; the
subsample rule is a scientific choice not yet written down. `config/canonical/grids.yaml`
still declares product, units and gauge correction as `UNKNOWN` although the fetcher now
knows all three.

Blocking for D09 only: no radar result can be produced until it is closed. Gauge-only
work is unaffected.

**Closes when:** D13.2 lands: a reader for the archive's daily zips, hourly sums, a
recorded subsample rule, and a `grids.yaml` with no `UNKNOWN`.

**Closed 2026-09-26.** `src/canonical/radar_archive.py` reads the archive's daily
zips in place, sums complete four-file hours (a cell missing in any one of the four
propagates as NaN, not a partial value), and subsamples the native grid by a
recorded, deterministic rule (`layers.radar.subsample_stride`). `grids.yaml`'s radar
block declares `product`, `units`, `timestamp_convention` and `gauge_correction` from
the archive files' own attributes, no `UNKNOWN` remaining. Wired behind
`src.canonical.grids.build()`, so validation, CSV shapes and the report are
unchanged. Satellite is untouched and stays on D13.3.

<a id="iss-31"></a>

### ISS-31 — `train_fused.py` raises `TypeError` as committed
**Severity:** Immaterial · **Owner:** _unassigned_ · **Status:** Open

`train_fold` is defined at `train_fused.py:178` without a `stats` argument and called at
`:266-273` with `stats=...`, so the transitional runner stops with a `TypeError` before
training. Even with that fixed, the `stats` name used inside `train_fold` (`:214, :219`)
refers to the last fold's statistics from the loop at `:149-156`, not the current fold's.
`docs/LEGACY_CODE.md` still calls it a validated runner.

**Immaterial because:** nothing in the production path calls it; `src/models/run.py` is the
runner. It matters only as the reference run for the inherited-baseline fingerprint
(ISS-14), which would need it working.
**Closes when:** it runs, or it is moved to the archive with a note that it does not.

<a id="iss-32"></a>

### ISS-32 — Fold count is hard-coded and nothing below Component 01 knows about regions
**Severity:** Material · **Owner:** D07 · **Status:** Open

The number of folds is fixed at five in `src/graph/builder_new.py:89`,
`src/models/contracts.py:69-71`, `src/models/evaluate.py:18` and `src/models/run.py:99,235`;
a split with any other count is rejected. "Sydney" is written into the output paths of
`config/canonical/standard.yaml`, `config/splits/sydney_5fold.yaml`,
`config/graphs/gauge_baseline.yaml` and `config/models/hgnn.yaml`, and the `region` key in
the ingestion config is used only to name files and report headings. A second region
therefore collides with Sydney in `database/standard/` unless every path is edited by
hand.

**Closes when:** the fold count is read from the split file, and a `region` key drives every
component's paths (D07, FR01).

<a id="iss-33"></a>

### ISS-33 — Test coverage is not measured
**Severity:** Material · **Owner:** D07 · **Status:** Open

The suite has 153 tests, but `pytest-cov` is not a dependency and no coverage number has
ever been produced, so nobody knows which lines the tests reach. Rule 10 of
[requirements/README.md](requirements/README.md) sets floors — 80 % over the production
packages, 90 % over the modules that write files other work depends on — and none of them
can be checked today. The modules in the 90 % class are the ones whose bugs corrupt data
silently rather than crash; ISS-28 and ISS-29 are both examples, and both are missing
tests as much as missing fixes.

Measured over all of `src/`, inherited review-only code would swamp the number, so the
measurement must be scoped to the production packages.

**Closes when:** D07.7 lands — `pytest-cov` installed, measurement scoped, both floors
failing the run, and the baseline recorded in `CURRENT_STATE.md`.


## Documentation gaps

<a id="iss-18"></a>

### ISS-18 — Component 03 split summary and plot are not produced
**Severity:** Material · **Owner:** Component 03 · **Status:** Open

The Component 03 contract requires `config/splits/sydney_5fold.md` — stations per fold, plus
a scatter plot of the fold-0 assignment. Only `sydney_5fold.json` and `sydney_5fold.yaml`
exist. The contract's own reasoning was that "one picture stops a lot of arguments about
whether the split is sane"; right now no one can check a split by looking at it.

**Closes when:** the split CLI writes the summary and the plot.

<a id="iss-19"></a>

### ISS-19 — Component 03 has no component README
**Severity:** Material · **Owner:** Component 03 · **Status:** Open

Every other component has one: `docs/ingestion/` (01), `docs/canonical/` (02),
`docs/graphs/` (04), `docs/models/` (06). There is no `docs/study/`. Under the Sprint 2
standard — no PR until your component has its own README — Component 03 does not meet it.

**Closes when:** `docs/study/README.md` exists with commands, inputs, outputs and known limits.

<a id="iss-20"></a>

### ISS-20 — Canonical README's Sydney statistics table is stale
**Severity:** Material · **Owner:** Component 02 · **Status:** Open

The "What the Sydney run produced (2026-09-08)" table predates the rain-day boundary fix, so
day verdicts and hourly counts have shifted. The table is marked stale in place, but stale
numbers in a component README get quoted anyway. (An earlier version of this entry also
blamed a region-scope change; that turned out not to exist — see ISS-11.)

**Closes when:** the build is rerun and the table is replaced.

<a id="iss-21"></a>

### ISS-21 — `docs/requirements/README.md` is empty but three documents route readers to it
**Severity:** Material · **Owner:** Requirements · **Status:** Closed — Sprint 2 requirements set written (2026-09-19)

The file is tracked and zero bytes. `597c0a2` emptied it when the Sprint 1 contracts moved
to `docs/archive/sprint1/`, and the Sprint 2 set that replaces it is still being written.

Three places send people there and none of them say it is empty: `README.md`'s "Start
here" calls it the shared component map, work packages, contracts and team rules;
`AGENTS.md` makes it item 3 of the required reading order; and `AGENTS.md`'s authority
order ranks it third, above `CURRENT_STATE.md` and the root `README.md`. A new owner or
agent following the reading order opens nothing and has no way to tell whether the file is
missing content or the rule is void.

**Closes when:** the Sprint 2 requirements set lands, or the file states in one line that
it is a placeholder and where the current contracts actually are.
