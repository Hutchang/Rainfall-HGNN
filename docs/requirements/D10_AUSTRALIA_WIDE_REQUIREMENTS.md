# D10 — Australia-wide

Template: study (see [AUTHORING_STANDARD.md](AUTHORING_STANDARD.md)). What counts as
work, and the rules, are in [README.md](README.md).

## Summary

Sydney was a null result. Under the paper's own protocol the model did not beat plain
inverse-distance weighting: IDW scores RMSE 0.667 and correlation 0.854, the gauge-only
model 0.686 and 0.843, and adding radar (0.688) then satellite (0.698) makes it slightly
worse each time. The paper explains this rather than burying it. What a learned model can
add appears to depend on how far apart the gauges are *compared with* how far rainfall
stays alike — its **correlation length**. Sydney's gauges sit about 3 km apart, rainfall
at two gauges 3 km apart is still about 0.85 correlated, and the correlation length is
near 14 km. A held-out gauge is already well predicted by the gauges around it, so there
is nothing left for a model to recover.

That is the argument for going national, and it is a much stronger one than "more data".
Gauge spacing across Australia runs from about 3 km in Sydney to hundreds of kilometres
in the interior, while rainfall in the tropical north comes apart over far shorter
distances than it does in Sydney. The ratio the paper says governs the result flips right
over. Australia-wide is the first time this method gets run in the regime where the paper
predicts it helps.

It also sets the trap. Most of Australia's hourly gauges sit on the east coast in
Sydney-like conditions. Train and score nationally without separating regions by that
ratio, and the headline number is dominated by the places where no gain is possible — and
we reproduce Sydney's null result at national scale.

This is research with a small amount of code, not a code deliverable: the questions come
first, the regions are chosen to answer them, and the code is whatever those experiments
need.

## Questions

**Q1 — the mechanism.** Does the gain over IDW rise with gauge spacing divided by
correlation length, as the paper says? Two ways, and we want both. **Across regions**
chosen to sit at different points on that ratio (D10.5). And **by thinning** Sydney's own
dense network — dropping gauges until the survivors sit 4, 5, 6 and 8 km apart — and
watching whether the gain appears as the network gets sparser (D10.2). The paper itself
calls the mechanism "consistent with the evidence rather than proven", because its two
testbeds differ in more than one way at once: Singapore also has microwave links, and it
also decorrelates faster. Thinning changes one thing only, so it turns their two-point
comparison into a curve. That is a contribution, not a scale-up.

**Q2 — one model or several.** Does one model trained across Australia, given features
that describe what kind of place each gauge sits in, match or beat one model fitted per
climate zone? The worry behind "model degradation" is that one set of weights spread
across very different climates fits none of them well. We expect it not to happen here,
for reasons set out below, and Q2 is how we find out rather than assume.

**Q3 — a region the model never saw.** Can a model trained on some regions predict a
region it was never trained on? This is the real Australia-wide question: a national
product has to work where no model was fitted.

## Three questions hiding inside "partition Australia"

Splitting Australia into subgraphs is three different questions with three different
answers, and they are usually discussed as one. Keep them apart.

1. **Splitting to fit the computer.** The graph or its data will not fit in GPU memory,
   or an epoch takes a week. This is engineering. The answer is cheaper training and
   batching, not separate models. (D10.6.)
2. **Splitting the learned weights.** Do different parts of Australia need their own
   trained model? A statistical question with a testable answer. (D10.7.)
3. **Routing a query.** Given a place to predict, which model answers? This question only
   exists if the answer to (2) is yes.

Once separated, (1) is ordinary work, (2) is an experiment, and (3) most likely never
comes up.

## Why one national graph is probably fine

A gauge connects to its 5 nearest gauges. A prediction node connects to its 9 nearest
grid cells. There are 5 message-passing layers. Nothing in the model looks at every node
at once — there is no global attention and no pooling over the whole graph. So a gauge in
Perth and a gauge in Cairns can never exchange a message: no chain of five short hops
joins them. The graph is already cut into pieces by its own geometry, and holding it as
one object costs almost nothing.

The intuition that one big model must be worse would be right for a model that sees the
whole country at once — a transformer with global attention, or a convolutional network
over a continental image. It is not right for nearest-neighbour message passing.

### The one part that genuinely breaks: the structural address

Each gauge carries four numbers computed from the shape of the whole graph — the
Laplacian positional encoding, or structural address, which lets the model tell apart two
gauges that happen to read the same value (`src/graph/gaugegraphnew.py:271-290`, which
calls `AddLaplacianEigenvectorPE(k=4)`). Four numbers is a choice made for a 70 km by
55 km graph of 148 gauges, where they describe position within a city. On a national
graph the same four numbers describe the largest shapes in the country — roughly "east or
west" and "north or south" — and every local distinction is gone.

Worse, they may not be well defined at all. The four numbers are the eigenvectors of the
graph Laplacian with the smallest non-zero eigenvalues. A graph in one connected piece
has exactly one zero eigenvalue, so "the smallest non-zero" picks out something definite.
A graph in several disconnected pieces has one zero eigenvalue per piece, and the phrase
stops picking out what it is meant to. A national 5-nearest-neighbour gauge graph is
almost certainly in several pieces: no gauge in Tasmania has a mainland gauge among its
five nearest, so Tasmania is already its own piece before the island stations are
counted. D10.1 reports how many pieces there actually are.

This needs fixing whichever way Q2 comes out, and nothing stops it today: the builder does
not check whether the graph is in one piece, so a national build would return arbitrary
numbers rather than fail. ISS-27 is measuring the related sign question on Sydney first;
this one is D10's, because it only bites at national scale. Three options, in rising order
of departure from the
paper: keep the method and use many more than four numbers; replace it with a *local*
structural address (a random-walk encoding, which only looks a few hops out and so does
not care how big the graph is); or drop it for plain coordinates plus terrain. D10.6
picks one and records why.

## Why regional variation is a feature, not a separate model

The things that vary across the country — how much it rains and in what form, how far
apart the gauges are, whether there is a radar in range and how far away it is, terrain —
can each be a number attached to a node or an edge. That is cheaper than separate models
and strictly better fed: the model learns "trust radar less at 250 km range" from every
radar in Australia at once, instead of from one region's worth of data.

It is also the argument this project set out to make. BRAIN divides Australia into
tropical, subtropical and temperate subregions and fits its error and correlation curves
per subregion (TO CONFIRM against the preprint — this is D11's to settle). If our answer
to regional variation is "we also cut Australia into regions, only with neural networks",
we have adopted the thing we are critiquing. If our answer is "we condition on the
geometry that drives the variation, continuously", the critique stands.

There is a practical reason too. A storm crossing the boundary between two separately
trained models produces a visible seam in the output map — the same storm predicted by
two sets of weights, meeting along a line. The headline figure of this project is a
reconstructed rainfall map, and a seam in it is hard to defend. Overlapping tiles
produced by one model have no seam, because the weights on both sides are identical.

## What actually does not scale

Five things, with the code. None of them matter at Sydney's size, and none of them is a
reason for separate models — they are all question (1) above. `N` below is the number of
gauges in the graph; D10.1 settles what `N` is nationally, and the worked numbers use
2,000 and 7,000 as brackets.

- **The training loop is the real cost.** Training hides one gauge, runs the whole graph
  forward, and repeats that once for every gauge before a single weight update
  (`src/models/trainer.py:74`). Sydney: 148 forward passes over a 148-node graph per
  batch of hours. At `N` gauges it is `N` passes over an `N`-node graph, so the work per
  epoch grows with the *square* of the gauge count: about 180 times Sydney's at 2,000
  gauges and about 2,200 times at 7,000.
  **The fix:** hide a random tenth to fifth of the gauges at once and score all of them
  in one pass. The paper's own justification is already written down — at evaluation time
  it masks every held-out node simultaneously, "since their predictions are mutually
  independent given the observed nodes". With 15 % hidden per pass, covering every gauge
  once per epoch takes about seven passes per batch instead of `N`, so the cost grows
  with `N` rather than `N` squared. The side effect is the one we want: hiding more
  gauges at once *is* a sparser network, which is the condition this whole deliverable is
  about.
- **The data is held as one dense block.** Every gauge times every hour times 6 features
  (the value, the validity flag, and the four structural-address numbers), held three
  times per fold — train, validation and test graphs
  (`src/graph/gaugegraphnew.py:201-211` and `271-290`). Sydney's 148 gauges over 17,520
  hours is about 190 MB per fold. At 2,000 gauges it is about 2.5 GB per fold and 13 GB
  across five folds; at 7,000 it is about 9 GB per fold and 44 GB.
- **Component 02 builds the full station-by-station distance table.** Every station
  against every station (`src/canonical/gauge/stations.py:42-45`), for two rules that
  only need neighbours within a few tens of kilometres. 150 stations is nothing; 17,000
  stations is about 2 GB held as one array.
- **Fetching is one request per second.** About 26 requests per station over the study
  period, so a 300-station region is about two hours and a national fetch is days. This
  is the thing to start early: it is wall-clock waiting, not effort.
- **Dry hours outnumber wet ones, and more so inland.** The authors already hit this in
  Sydney — they leave the model's output unconstrained during training specifically to
  stop the gradients vanishing over the many dry periods in the record (paper section
  3.3). With the arid interior in the training set it gets worse. Expect to have to
  sample wet hours preferentially, and to record that as a choice rather than let it
  happen quietly.

**Grid nodes are less of a problem than they look.** Himawari at 2 km over the national
box (112 to 154 degrees east, 9 to 44 degrees south — roughly 4,200 km by 3,900 km) is
about four million cells. We do not need them during training. Cross-support edges only
reach cells near a prediction node, and during training the prediction nodes are just the
gauges: 9 cells attached per gauge, and of the 5 message-passing layers one is the
cross-support hop, so information travels at most 4 steps across the 8-neighbour grid
before it arrives. That is a patch of a few hundred cells around each gauge, not the
continent. Full-resolution inference does need every cell, but that is one forward pass
with no gradients and it can be done a tile at a time. **The size of the training graph
is set by the gauge count, not by the resolution of the inputs** — which removes most of
the compute argument for subgraphs. D10.6 measures the real patch size rather than
trusting this estimate.

Radar is not one national grid in any case: it is one 256 km by 256 km footprint per BoM
radar site (D13). "Radar for a region" means choosing the site, which is why the region
rule asks regions to fit inside one footprint where they can.

**The number of hours is not a problem.** 17,520 hourly steps over 2021–2022 sounds
large, but the graph's shape never changes — gauges and grid cells do not move — so the
graph is built once and the hours are streamed through it as features. Our study period
is already twice the paper's Sydney run, which used 2022 only (8,760 hours).

## Region rule

Regions are chosen after D10.1 shows where the gauges with usable hourly data actually
are. The rule, in order of priority:

1. **Spread across the ratio.** The regions must sit at different points on gauge spacing
   divided by correlation length, because that is the axis Q1 tests. Sydney is one end
   (about 3 km spacing against a 14 km correlation length). The other end is somewhere
   the gauges are far apart and the rainfall decorrelates fast — the tropical north or
   the inland. A set of regions that all land near Sydney's ratio answers nothing.
2. **Different rainfall types**, as a secondary spread: tropical storms in the north,
   steady winter rain in the south-west, rain forced by terrain along the ranges, the dry
   interior.
3. **Inside one radar footprint** (256 km by 256 km around a BoM radar site) *where the
   region can be*, so D10.9 can add radar later. A genuinely sparse region may have to be
   larger than one footprint to hold enough gauges; that region is then gauge-only, and
   the document says so.
4. **Enough held-out stations to score.** Five blocked folds need enough stations that
   each fold holds a usable handful. Where a sparse region cannot manage five, use fewer
   folds and record the count — this needs the fold count to come from the split file
   (ISS-32, D07.1).

Three to five regions. Candidate names are not fixed here; Darwin and Sydney are the
obvious two ends, and D10.1 decides the rest.

**"City" is not the rule.** The five capitals are five dense, well-resolved networks —
all on the same side of the paper's line as Sydney. They would show that Sydney's result
holds in four similar places, and little else.

**Sparse regions come with a small-sample problem, and we say so.** A region with few
gauges gives a score resting on few held-out stations, so its error bars are wide.
BRAIN's own evaluation outside radar coverage rests on 33 of about 2,155 hourly gauges
(TO CONFIRM — D11). This is exactly why D10.2's thinning study matters: it studies the
sparse regime using a dense region's ground truth, so the sparse-region result has a
controlled counterpart rather than standing alone.

**One calibration point.** BRAIN — a fully engineered operational system — found that
outside radar coverage its blended product reached higher correlation but *higher* daily
RMSE than IDW (TO CONFIRM — D11). Beating IDW where gauges are sparse is hard for
everyone. It is not a low bar, and a result that fails to clear it is still a result.

## Protocol

- **Data.** Per region: the same study period, 2021-01-01 to 2022-12-31, fetched with
  that region's `config/ingestion/<region>.yaml` and built into standard tables by
  Component 02 with the same rules in `config/canonical/standard.yaml`.
- **Split.** D08's, unchanged: five folds each hiding whole areas (D08 Gap 1), 15 % of
  the remaining stations for validation. Fewer folds only where a region cannot support
  five, with the count recorded.
- **Known gauges at test time.** Training and validation gauges, for every method
  (D08 Gap 2).
- **Methods.** IDW with power 2.0; network mean; HGNN(G) with `config/models/hgnn.yaml`
  as it stands, plus input scaling (D08 Gap 3) and both-direction gauge edges (already
  in the code).
- **Metrics.** RMSE in millimetres and Pearson correlation, computed per fold then
  averaged across folds. Each region also reports its median gauge spacing, its
  correlation at that spacing, and its correlation length (the D08.5 figure, per region).
- **What is held fixed.** Everything except the region. Where a scale fix from D10.6
  changes how training works — the masking scheme above all — **Sydney is re-run under
  the same fix**, so every row in every table comes from the same method. D10.6 also
  reports Sydney scored both ways, so we know whether the cheaper masking changed the
  answer.

## Inputs

| Input | From | State today |
|---|---|---|
| national station list and its period coverage | BoM's KiWIS service, through `src.ingestion.gauge.survey` | the program works; nobody has run it nationally, so the count is unknown |
| Sydney standard tables, split, graphs and results | D08 | not built on any machine yet |
| the Sydney correlation-versus-distance figure | D08.5 | does not exist |
| `config/ingestion/<region>.yaml`, one per region | D10.3 | only `sydney.yaml` exists |
| per-region raw gauge files | D10.4 | none fetched |
| per-region radar tables | D13.2 | not produced |
| Gadi, for the runs that outgrow a laptop | [../nci/README.md](../nci/README.md) | access per person TO CONFIRM |

## Subtasks

| Subtask | What | Needs | Done when |
|---|---|---|---|
| **D10.1** | National station survey: `python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml --no-box` draws every BoM rainfall gauge in Australia in one request, and with a config also marks which of them cover the study period. Record the national count, the count with usable hourly data, a density map, and — building the 5-nearest-neighbour gauge graph on that national set — how many disconnected pieces it falls into. | nothing | A figure and three counts in the findings: all stations, stations covering 2021–2022, and graph pieces. Nobody has any of these numbers today. |
| **D10.2** | **Thinning study on Sydney (Q1, controlled).** Thin Sydney's gauge network to target spacings of about 4, 5, 6 and 8 km and run IDW and HGNN(G) at each, scoring the *same* held-out stations and the same target rows every time. Plot gain over IDW against gauge spacing divided by correlation length. Method and limits are below this table. | D08.1, D08.2, D08.3, D08.5 | A curve with at least four thinning levels plus the full network, each point labelled with the surviving gauge count, and a sentence on whether the gain appears as the network thins. |
| **D10.3** | Write the region rule down with the chosen regions and the reason for each. One `config/ingestion/<region>.yaml` per region: a bounding box, the shared period, and the radar `site_id` for that footprint where one applies. | D10.1 | Configs exist; each has a one-paragraph reason naming where it sits on the ratio. |
| **D10.4** | Fetch each region's gauges. Serial at one request per second: about 26 requests per station, so a 300-station region is about two hours. Start early; it is wall-clock time, not effort. Use the runner's `region` key (D07.1) or hand-set `output_dir` so regions do not collide in `database/raw/`. | D10.3; D07.1 or hand-set paths | Each region has its raw files, index and report. |
| **D10.5** | **Q1 across regions.** Build tables, split and graphs per region and run IDW, network mean and HGNN(G) under the protocol above. Compute the D08.5 correlation-versus-distance figure per region — that is the x-axis. | D10.4, D10.6, D08.1, D08.2 | One D08-style table per region, plus median spacing, correlation at that spacing, and correlation length for each. |
| **D10.6** | **The scale fixes.** (a) Mask a random subset of gauges per pass instead of one gauge at a time (`src/models/trainer.py:74`), and report Sydney scored both ways. (b) Decide and implement what replaces the four structural-address numbers on a large or disconnected graph, and measure the gap between the fourth and fifth smallest eigenvalues on the Sydney graph, because if they sit close together the fourth number is noise. Coordinate with ISS-27, which is measuring the sign treatment on Sydney first. (c) Replace the full station-by-station distance table with a neighbour search (`src/canonical/gauge/stations.py:42-45`). (d) Measure the grid patch a gauge actually reaches, to check the "a few hundred cells" estimate above. | D10.1 says how big the problem is; (a) and (b) can start now | Sydney scores the same within noise under subset masking; the graph builder runs on a disconnected national graph without an ill-defined structural address; the distance table no longer grows with the square of the station count; the measured patch size is in the findings. |
| **D10.7** | **Q2 — how many models.** Same regions, same targets, three strategies: (a) one model per region; (b) one model trained on all regions pooled, with the regional features described above; (c) one model per BRAIN zone, using BRAIN's own tropical/subtropical/temperate split. Score per region, never only pooled. | D10.5, D10.6, D07 (to keep the runs straight) | A table with all three strategies scored per region, and a sentence on whether pooled matches or beats zoned. |
| **D10.8** | **Q3 — a region the model never saw.** Train on all regions but one and score the held-out region, once per region. | D10.7 | One row per held-out region, beside that region's own within-region score from D10.5. |
| **D10.9** | Radar per region, using D13's readers and tables and D09.2's edges. | D13.2, D13.4, D09.2, D10.4 | G+R rows per region, each stating the gauge-correction caveat D09 describes. |

### D10.2 — how to thin, and where it runs out

**How.** Random subsets leave clumps and holes, so their median spacing does not describe
the network. Instead: for a target spacing `d`, lay a grid of step `d` over the region
and keep the gauge nearest each grid point. That gives an evenly spread network at
spacing `d`. Repeat with a few offsets of the grid so each point on the curve has a
spread rather than a single number.

**What is held fixed.** The held-out test stations and the target rows stay the same at
every thinning level, so RMSE is comparable across the curve. What thins is the set of
gauges a method may read, which is also the training set for the model. That means
thinning changes two things at once: the field is sampled more coarsely, and the model
has less data to learn from. Both are true of a genuinely sparse region, so thinning both
is the honest main run. Name the confound in the write-up, and if there is time add one
variant that keeps the full training set and thins only what the method may read at test
time, which separates the two effects.

**Where it runs out.** Sydney's box is only about 70 km by 55 km. At 3 km spacing it
holds 148 study gauges; at 8 km, roughly 15; beyond about 10 km there are too few left to
score five folds. So Sydney can move the ratio from about 0.2 (3 km over 14 km) to about
0.6, which covers a useful part of the range but not the sparse end. Reaching 30 km and
100 km spacings needs a larger dense area — the wider east-coast network — which is a
fetch, so it rides on the first large region D10.4 brings in and is written up as an
extension of the same curve.

## Deliverable

`docs/results/D_australia_wide.md`, holding:

- the national survey figure and its three counts (D10.1);
- the thinning curve (D10.2), with the confound named;
- the region rule and the chosen regions, each with its reason;
- a D08-style table per region (D10.5);
- the Q1 figure: gain over IDW against gauge spacing divided by correlation length, with
  the thinning points and the region points on the same axes;
- the Q2 table (D10.7) and the Q3 table (D10.8);
- what the scale fixes changed, including Sydney scored under both masking schemes
  (D10.6);
- the run id for every number.

The region configs live under `config/ingestion/`.

## Acceptance

- Every region run uses D08's protocol unchanged; the only things that vary are the
  region and the D10.6 scale fixes, and Sydney is re-run under those same fixes so its
  row is comparable with everyone else's.
- Q1 has both figures — the thinning curve and the across-regions scatter with at least
  three regions — and a sentence saying whether the paper's account held. A curve that
  shows no gain as the network thins is a result and is reported as one.
- Every count in the document about gauges, graph pieces, or region size comes from
  D10.1's survey, not from an estimate in this requirements document.
- Q2 reports scores per region and states how many models each strategy fits. A national
  average alone is not acceptable: it hides the regime where gain is possible.
- Any result resting on few held-out stations states how many.
- No region result is quoted without its run id.
- The coverage floors (README rule 10) hold for what D10.6 touches: `src/models/` and
  `src/graph/` at 80 %, and `src/canonical/gauge/stations.py` at 90 %, since Component 02
  writes tables other people's work depends on.

## Dependencies

D08 (D08.1 and D08.2 for the protocol; D08.3 for the Sydney row D10.2 thins away from;
D08.5 for the correlation figure). D07.1 for the `region` key and the fold count, without
which regions collide in `database/standard/`; D07 generally for keeping D10.7's runs
straight, which is not credible without it. D13 (readers) and D09.2 (edges) for D10.9.
D11 for every BRAIN fact this document marks TO CONFIRM — the subregion split, the 33
gauges, and the correlation-versus-RMSE result outside radar coverage. D08.6 for the tool
that measures distance to the nearest visible gauge, which is Q1's x-axis. Issues: ISS-32
(no region concept below
Component 01; fold count fixed at five), ISS-27 (the structural address, whose Sydney
sign experiment runs first and whose national answer is D10.6's to decide).

## This week and backlog

**This week:** D10.1 (one request, a few hours' work, and it unblocks D10.3); D10.3 once
D10.1 lands; D10.2 started if D08.3 lands, since it needs no new data; D10.4 started for
the first region, because the fetch is wall-clock time.
**Backlog (week 7+):** D10.5, D10.6, D10.7, D10.8, D10.9.

The ordering changed from the first draft of this document. D10.2 moved early because it
answers Q1 — the question the paper leaves open — using only data D08 already produces,
and it needs no fetch and no Gadi.

## Findings

*(filled in as the work runs)*
