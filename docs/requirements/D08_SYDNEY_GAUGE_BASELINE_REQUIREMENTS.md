# D08 — Sydney gauge-only baseline

Template: study (see [AUTHORING_STANDARD.md](AUTHORING_STANDARD.md)). The rule for what
counts as work, and the rules everyone follows, are in [README.md](README.md).

## Summary

Run the paper's Sydney comparison — IDW, network mean and the gauge-only HGNN — under
the paper's protocol, on our own 2021–2022 data, and put our table beside the paper's
Table 2 with every difference explained. "Reproduce" here means the same method and the
same evaluation; it does not mean matching their numbers, which we cannot do (different
years, different station set, our own folds).

## Question

Under the paper's protocol, do IDW, network mean and HGNN(G) rank on our Sydney data as
they do in Table 2 (IDW best; HGNN close behind), and does the size of the gap fit the
paper's explanation — that Sydney's gauges are dense enough to resolve the rainfall field
on their own, so a learned model has little to add?

## Three gaps between the paper and our code

These are the whole job. Each one is what the code does, what the paper says, and why the
difference matters. Nothing here is a choice to argue about — the paper did it and we
don't.

### Gap 1 — The split doesn't hide whole areas

**What the code does.** `src/study/split.py` groups the study stations into 10 clusters
by location, then hands each cluster's stations out one at a time to fold 0, 1, 2, 3, 4,
0, 1 and so on (`create_test_folds`; the counter never resets between clusters). Every
fold therefore hides a fifth of every cluster, and a hidden gauge keeps gauges from its
own cluster visible a few kilometres away in every direction.

**What the paper says** (§4.2): *"Stations are partitioned into spatially coherent folds
so that held-out stations are not adjacent to training stations, preventing a model from
succeeding through trivial spatial proximity rather than genuine generalisation."*

**Why it matters.** Guessing a hidden gauge that has a visible neighbour next door is
close to copying that neighbour. IDW does it well, the model does it well, and the gap
between them shrinks — so our table cannot be set beside Table 2.

**Fixed in D08.1.** Give each fold two whole clusters (10 clusters over 5 folds). Keep
the clustering step and the 15 % validation share. Build the split from
`config/splits/sydney_5fold_blocked.yaml`, the config D08.1 also writes, where every
setting is traced to the paper or marked as our choice; write the result to
`config/splits/sydney_5fold_blocked.json`, and keep the old one so the effect of the
change can itself be measured — that is D08.7's job.

### Gap 2 — IDW reads fewer gauges than the neural net

**What the code does.** In each fold the visible gauges split into *training* gauges,
used to fit the model, and *validation* gauges, used only to decide when to stop training
(18 per fold). IDW and network mean read the training gauges only
(`src/models/baselines.py:66`). The HGNN's test graph holds training **and** validation
gauges (`src/models/graphs.py:68-73`). The neural net guesses from 18 more gauges.

**What the paper says** (§4.2): held-out stations are *"masked and predicted from the
remaining observations"*, and *"the same held-out stations are used across all methods
and all support configurations to ensure comparability."* Every gauge that is not hidden
is a remaining observation, for every method.

**Why it matters.** Any gap between IDW and the HGNN is partly the extra 18 gauges, not
the method. The comparison this project exists to make isn't fair until it's fixed.

**Fixed in D08.2.** Every method reads training and validation gauges at test time. The
validation gauges were never used to fit weights, only to stop training, so letting the
baselines see them gives nothing away. One line in `src/models/baselines.py:66`, no graph
rebuild.

### Gap 3 — The model is fed raw millimetres

**What the code does.** Rainfall reaches the model in physical millimetres, 0 to 60 and
beyond, sitting next to four positional numbers around ±0.1. `src/models/run.py:75-78`
refuses anything else.

**What the paper says** (§4.3): *"Features are z-score normalized per node type using
training-split statistics only"* — subtract the training average, divide by the spread.

**Why it matters.** When one input is hundreds of times larger than another, the large
one dominates early training, and a learning rate chosen for scaled inputs is wrong for
the small ones. Our config copies the paper's learning rate, which was chosen for scaled
inputs, so training is probably worse than it should be and we can't claim we followed
the recipe.

**Fixed in D08.2.** Graph files stay in physical millimetres. Component 06 scales the
rainfall channel per node type when it loads the graphs, using the training graph's
average and spread only, and writes both numbers into the run's `config.yaml`. The
validity flag and positional numbers are left alone; targets and predictions stay in
millimetres.

### Already closed — gauge edges now run both ways

The graph builder used to add one edge per pick, so a gauge received from whoever picked
it rather than from its own nearest, and a gauge nobody picked received nothing. The
paper connects each gauge to its own nearest and normalises at the receiving gauge
(§3.2, Eq. 1), which makes the connectivity symmetric. Fixed 2026-09-20 in commit
`4600a04`: `src/graph/gaugegraphnew.py` gained a `both_directions` setting and
`config/graphs/gauge_baseline.yaml` turns it on. **You are not writing this.** All that
is left is rebuilding the graphs on real data, which is part of D08.2.

## Protocol

- **Data.** Sydney gauges, 2021-01-01 to 2022-12-31, fetched with the box in
  `config/ingestion/sydney.yaml` as it stands — it matches the paper's roughly 70 km by
  55 km extent; a different box is a different region (D10), not a change to this
  baseline — built into standard tables by Component 02 with the current rules in
  `config/canonical/standard.yaml`.
- **Split.** Five folds, each hiding whole areas (Gap 1). 15 % of the remaining stations
  are validation, used only to decide when to stop training.
- **Visible gauges at test time.** Training and validation gauges, for every method
  (Gap 2).
- **Methods.** IDW with power 2.0; network mean; the HGNN with `config/models/hgnn.yaml`
  as it is (hidden size 32, 5 layers, dropout 0.15, learning rate 0.0003, weight decay
  0.0001, early stopping after 10 epochs without improvement, α = 0.18), plus input
  scaling (Gap 3) and both-direction gauge edges (already done).
- **Metrics.** RMSE in millimetres and Pearson correlation, computed per fold, then
  averaged across the five folds — that is Table 2's statistic. Also report the pooled
  value that `score_predictions` already gives, labelled as pooled.
- **Targets.** Every method predicts the same hidden station-hours (the existing contract
  in `src/models/contracts.py`).

**Two settings, recorded not argued.** The paper says "weighted MSE" and never gives the
weights, so there is nothing to match: α = 0.18 comes from the paper's first author's own
config (`config.yaml:21`, commit `dc5db56`) and stays. The IDW exponent of 2.0 is our own
default; nobody has agreed it formally, so say so wherever it is quoted.

## Inputs

| Input | From | State today |
|---|---|---|
| `database/standard/gauge_hourly.csv`, `stations.csv` | Component 02 | not built on any machine yet; one fetch (~90 min) and one build (~15 s) |
| `config/splits/sydney_5fold_blocked.yaml` — the paper-matching split config | subtask D08.1 | does not exist |
| `config/splits/sydney_5fold_blocked.json` — what that config produces | subtask D08.1 | does not exist |
| `config/splits/sydney_5fold.json` — today's split, kept as the "before" | Component 03 | exists and is git-tracked |
| `database/graphs/<name>/fold-<i>/*.pt` | Component 04, rebuilt after D08.2 | not built |
| `config/models/hgnn.yaml` | Component 06 | matches the paper's settings already |

## Subtasks

| Subtask | What | Needs | Done when |
|---|---|---|---|
| **D08.1** | Make each fold hide whole clusters, instead of a scattered fifth of every cluster. Change `create_test_folds` in `src/study/split.py`. Save the result as `config/splits/sydney_5fold_blocked.json` and keep the old file. | nothing; start now | No hidden station has a training station in its own cluster. |
| **D08.2** | Let IDW and network mean read the validation gauges too (`src/models/baselines.py:66`). Scale the rainfall input when Component 06 loads the graphs, and write the two numbers it used into the run's `config.yaml`. Rebuild the graphs. | D08.1 | Both changes have tests. The rebuilt graph report shows no gauge without an incoming edge. |
| **D08.3** | Run IDW, network mean and HGNN(G) on all five folds. Write our table next to the paper's Table 2. | D08.1, D08.2, the Sydney tables | The table exists, and each difference from the paper has a sentence. |
| **D08.4** | Add ordinary kriging as a second classical baseline (`uv add pykrige`). Optional. | D08.1 | A `kriging` row in the same table, on the same targets. |
| **D08.5** | Plot how correlated two gauges are against how far apart they are. This is the paper's explanation for Table 2, and D10 needs the same plot per region. Watch out: a trial on the legacy data matched the paper at 3 km (0.815 vs 0.85) but gave 43 km for the correlation length against their 14 km, so agree what "correlation length" means before quoting one. | the Sydney tables only | A figure, plus the two numbers with the definition stated. |
| **D08.6** | Build the tool that measures how far each hidden gauge is from the nearest gauge a method can read. Works on a made-up station table, so it needs nothing else. D10 reuses it. | nothing; start now | TEST01 and TEST02 pass. |
| **D08.7** | Measure that distance on the old split and the new one, and run all three methods on both. Produce the histogram and the before/after table. Close ISS-10 with them, and put the validation-gauge count in ISS-34. | D08.1, D08.2, D08.3, D08.6, the Sydney tables | Both splits measured and scored on identical target rows, with run ids. ISS-10 closed. |

Two jobs go with D08.1 because whoever does it is already in that code: the split summary
and fold map Component 03 still owes (ISS-18), and the missing `docs/study/README.md`,
the only component README that does not exist (ISS-19).

Dropped from this deliverable, with reasons: the support-agnostic HGNN (the paper ran it
on Singapore only, Table 1 — not a Sydney reproduction item); matching the paper's exact
75-station list, which was never published, so our station set is built from the BoM list
and the Component 02 rules instead and will not match theirs. Also out of scope:
**where the 18 validation gauges are placed.** They are picked by shuffling the non-hidden
stations and taking the first 15 % (`split.py:239-248`), so one can land right beside a
hidden gauge inside an area Gap 1 just hid. The paper gives no rule for where validation
gauges go, so placing them deliberately would *add* to the paper rather than return to it.
D08.7 measures how often it happens; the question itself is [ISS-34](../ISSUES.md#iss-34).

### The split config, for D08.1

D08.1 builds the `.json` from a config file, `config/splits/sydney_5fold_blocked.yaml`.
Write that config first, because it is where a reviewer checks whether we really used the
paper's protocol. Every setting in it needs a comment saying where it came from:

- Carry one setting for each thing the paper states about the split, with a comment
  quoting the sentence it comes from
  ([§4.2](../source_paper.md#42-evaluation-protocol)): five folds; each fold hides whole
  areas, so that hidden stations are not next to training stations; every station is
  hidden exactly once across the folds; the same hidden stations for every method (this
  one is kept by there being one split file per config, rule 6, and the comment says so).
- Carry one setting for each thing the paper leaves open, with a comment that says
  "paper silent" and the reason for our value: how the areas are drawn (today's method,
  location clusters on projected coordinates); how many areas (10, today's value); areas
  per fold (2, because 10 areas over 5 folds); the validation share (15 %, today's value;
  the paper only says early stopping is used,
  [§4.3](../source_paper.md#43-implementation-details)); the seed (from
  `config/seeds.yaml`, rule 5).
- Say at the top which of these settings `src/study/split.py` cannot honour yet — today it
  can only deal a fifth of every cluster to each fold — so the code half of D08.1 knows
  what to build. The key name for the whole-areas rule is yours to choose; the comment
  states the behaviour, not the key.
- Note, in the same file, the one difference no config can remove: the paper's Sydney
  station list was never published ([ISS-23](../ISSUES.md#iss-23)), so the split runs over
  all 148 study stations rather than the paper's set.

Without this, "we used the paper's split" is a claim nobody can check.

**Done when:** every setting has one of those two comments, the unsupported ones are
listed at the top, and the split can be built from the file without anyone having to
work anything else out.

### The distance tool, for D08.6

One new file, `src/study/held_out_distance.py`, with one command. It reads the `.json` a
split config produced:

```console
uv run python -m src.study.held_out_distance config/splits/sydney_5fold.json \
    --stations database/standard/stations.csv --out-dir database/study/
```

It must:

- For each fold, for each hidden (test) station, find the nearest gauge the method may
  read — any training or validation station in that same fold — by great-circle distance,
  reusing `haversine_km` from `src/canonical/gauge/stations.py:33-39` rather than writing
  a second copy.
- Write `<split name>_held_out_distance.csv` with one row per fold and hidden station:
  `fold_index`, `station_id`, `nearest_visible_id`, `nearest_visible_role` (`train` or
  `validation`), `distance_km`.
- Write a summary block — count, minimum, 10th percentile, median and maximum, per fold
  and pooled, plus the share of rows whose nearest visible gauge is a validation station —
  into `<split name>_held_out_distance.md`, and append the same block to the split summary
  D08.1 writes, so a reviewer sees it beside the fold map.
- Refuse, naming the ids, if any station in the split is absent from `stations.csv` or has
  `use_in_study` false. Do not drop it quietly (rule 7: say what you dropped).

This number says how hard the test actually is. A hidden gauge with a visible one
500 m away is close to copying its neighbour, so every method scores well for a reason
that has nothing to do with the method. D10 reads the same tool per region, where the
number becomes the x-axis of its first question.

**Done when:** TEST01 and TEST02 pass.

## Testing

| Test | Checks | Subtask |
|---|---|---|
| TEST01 | Three gauges on a line at 0 km, 1 km and 10 km, with the gauge at 10 km hidden and the other two visible: the tool reports `distance_km` 9.0 and names the gauge at 1 km. A second case puts the nearest visible gauge in the validation role and checks `nearest_visible_role` says so. | D08.6 |
| TEST02 | A split naming a station absent from `stations.csv`, and one naming a station with `use_in_study` false, are both refused with the id in the message and a non-zero exit code; nothing is written. | D08.6 |

## Deliverable

`docs/results/B_sydney_gauge_baseline.md`: our Table 2 beside theirs; a paragraph per
difference; the D08.5 figure; the run ids in the registry that produced every number.
Every number in the document traces to a run folder.

It also carries the split evidence, because the table cannot be read without it: the
distance histogram for today's split and the new one with the minimum and 10th percentile
marked; the share of nearest visible gauges that were validation stations; the
before/after table of all three methods under both splits; and one paragraph saying
whether Sydney's ranking held when the split changed and by how much the gap moved.

Beside each split config, committed with it: `<split name>_held_out_distance.csv` and
`<split name>_held_out_distance.md`.

## Acceptance

- Every method scored on identical target rows (the existing checker enforces this).
- Metrics reported as mean across folds, with the per-fold values in an appendix.
- The document states the split file, which gauges each method could read, the scaling
  numbers and α used.
- **The table is not reported without the distance beside it.** How far a hidden gauge was
  from the nearest gauge a method could read is what says whether the table measured the
  model or measured the neighbours.
- The distance is reported as a distribution — minimum, 10th percentile, median,
  histogram — never as a mean on its own. A mean hides exactly the short distances that
  matter.
- Every setting in `sydney_5fold_blocked.yaml` traces to a paper sentence or to a stated
  choice. A reviewer who reads only that file can say which parts are the paper's and
  which are ours.
- Each `.json` compared was rebuilt from its config and seed (rule 5). No split is
  generated inside a results script (rule 6), and no split rule beyond Gap 1 and Gap 2 is
  introduced.
- The comparison is honest about what differs from the paper; it does not claim
  reproduction of their numbers.
- ISS-10 closes with the figure and the table attached. "D08.1 was applied" does not close
  it: D08.1 is the change, and D08.7 is the evidence the change did what it claimed.
- The ISS-34 count is recorded whichever way it came out.
- The coverage floors (rule 10 in [README.md](README.md#rules-everyone-follows)) hold for
  what this deliverable touches: `src/study/split.py` and the new
  `src/study/held_out_distance.py` at 90 % because both write files other work depends on,
  `src/graph/` and `src/models/` at 80 %, and `src/models/run.py` at 90 %.

## Who to ask

| If you need | Go to | Owner |
|---|---|---|
| The Sydney gauge data to exist at all | Component 01's fetch, then Component 02's build | TO CONFIRM |
| Your runs kept apart and recorded | D07, the experiment runner | TO CONFIRM |
| Radar added on top of this baseline | D09 | Rick |
| The distance tool for a region that is not Sydney | D10 reads D08.6's tool unchanged | TO CONFIRM |

D07 is not required — D08.3 can run component by component with hand-set output
directories — but D08.3's results should be re-run under the runner once it lands so they
sit in the registry.

**Everything in D10 waits on this deliverable.** Extending to the rest of Australia only
means something once the Sydney baseline is trustworthy, so D10 is the last thing to
start, not the first.

## This week and backlog

**This week:** D08.1, D08.6 (neither needs a fetch — one is a config and a function, the
other builds and tests on a synthetic station table), then D08.2, D08.3 (a first table)
and D08.5 once the Sydney tables exist.
**Backlog (week 7+):** D08.4; D08.7, which needs everything above it; a fingerprint test
pinning the inherited graph's node and edge counts so shared graph code cannot drift
unnoticed; the experiment on whether a station's four structural-address numbers stay
consistent between the train and test graphs.

## Findings

*(filled in as the work runs)*
