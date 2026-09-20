# Component requirements — Sprint 2

Sprint 2 is one week (end date TO CONFIRM), six people. The work is split into eight
deliverables, D07 to D13, that run at the same time; the numbering continues from the
Sprint 1 components 01 to 06. Each deliverable has its own requirements document and a
set of numbered subtasks (D08.1, D08.2, ...). Three gaps between the paper and our code
have to close first, because every training run in D08, D09 and D10 waits on them;
all three are D08’s. Whatever is not finished this week is written down as backlog and
carries into week 7.

Sprint 1's five component contracts are archived in [../archive/sprint1/](../archive/sprint1/).
They still describe what each component implements today; they are reference, not
assignments. Where the project stands is in [../CURRENT_STATE.md](../CURRENT_STATE.md).
What is broken, missing or undecided is in [../ISSUES.md](../ISSUES.md). Unfamiliar words
are in [GLOSSARY.md](GLOSSARY.md).

## The deliverables

| Deliverable | Name | Document | This week | Backlog (week 7+) | Owner |
|---|---|---|---|---|---|
| D07 | Experiment runner | [D07_EXPERIMENT_RUNNER_REQUIREMENTS.md](D07_EXPERIMENT_RUNNER_REQUIREMENTS.md) | D07.1, D07.2, D07.6, D07.7; D07.5 if a Sydney fetch runs | D07.3, D07.4 | TO CONFIRM |
| D08 | Sydney gauge-only baseline | [D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md](D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md) | D08.1, D08.6, then D08.2, D08.3 (first table), D08.5 | D08.4, D08.7, ISS-14 | TO CONFIRM |
| D09 | Radar fusion | [D09_RADAR_FUSION_REQUIREMENTS.md](D09_RADAR_FUSION_REQUIREMENTS.md) | D09.2 | D09.4, D09.5 | Rick |
| D10 | Australia-wide | [D10_AUSTRALIA_WIDE_REQUIREMENTS.md](D10_AUSTRALIA_WIDE_REQUIREMENTS.md) | D10.1; D10.3 once D10.1 lands; D10.2 started if D08.3 lands; D10.4 started for the first region | D10.5 to D10.9 | TO CONFIRM |
| D11 | BRAIN paper report | [D11_BRAIN_REPORT_REQUIREMENTS.md](D11_BRAIN_REPORT_REQUIREMENTS.md) | TO CONFIRM | TO CONFIRM | TO CONFIRM |
| D13 | Source grids from NCI: radar and satellite | [D13_SOURCE_GRIDS_REQUIREMENTS.md](D13_SOURCE_GRIDS_REQUIREMENTS.md) | D13.1, D13.2 | D13.3, D13.4, D13.5 | TO CONFIRM |

Suggested split of six people: D07 one person; D08 two (one on D08.1 and D08.5, one on
D08.2 and D08.3); D09 Rick; D10 one; D11 one. D10 starts last: extending to the rest of
Australia only means something once the Sydney baseline is trustworthy.

How the deliverables depend on each other:

```text
        D08.1 (new split) --> D08.2 (graph and model changes) --> D08.3 (Sydney table)
                              |                                   |
        D13.2, D09.2 ---------+--> D09.4                          |
                                                                  |
        D10.1 (survey) --> D10.3 (regions) --> D10.4 (fetch, slow) -+--> D10.5
                                                                    |
        D08.3 -----> D10.2 (thin Sydney; no fetch needed) ----------+

        D08.6 (distance tool) --> D08.7 (before/after; closes ISS-10)
                 |                  ^
                 +--> read by D10.5 |
        D08.3 -------------------------+

        D07.1, D07.2, D07.6 (runner) ---- needs nothing; everyone adopts it when it lands
        D07.5 (golden run) ---- needs one Sydney fetch
```

Subtasks with no arrow into them can start on day one: D07.1, D07.2, D07.6, D07.7, D09.2,
D10.1, D13.1, D13.2, D08.1, D08.6. D08.5 starts as soon as the Sydney tables exist. The
one real choke point is D08.1 + D08.2: every training run in D08, D09 and D10 wants the
settled protocol.

## What counts as work

One rule decides whether something belongs in this sprint:

> **The paper does it, and our code doesn't.**

That is the whole test. We are reproducing [the source paper](../source_paper.md), not
improving on it. If you find yourself arguing about which approach is better, you are off
track — the only question is what the paper did.

Every gap is written the same way, inside the deliverable that fixes it:

1. **What the code does today**, with the file and line.
2. **What the paper says**, with the section and a quote.
3. **Why the difference matters** to the result.

There is no separate register and nothing to look up. The gap lives in the deliverable
that closes it, and whoever owns that deliverable owns the gap.

**If your work needs someone else's gap closed first**, the deliverable's *Needs* column
names the deliverable and its owner. Go and ask that person. Don't wait for a meeting.

### The gaps that are open

All three are D08's, and every training run in D08, D09 and D10 waits on them.

| The code does | The paper does | Who fixes it |
|---|---|---|
| Hands a fifth of every cluster to each fold, so a hidden gauge keeps neighbours a few km away | Hides whole areas (§4.2) | [D08.1](D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md) |
| Lets the neural net read 18 more gauges than IDW | Gives every method the same gauges (§4.2) | [D08.2](D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md) |
| Feeds the model raw millimetres | Scales inputs on training statistics (§4.3) | [D08.2](D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md) |

One is already closed: gauge edges now run both ways (commit `4600a04`), which is what
the paper's Eq. 1 requires. It still needs a rebuild on real data — part of D08.2.

### Two things that are not gaps

Neither is an argument to have. They are settings, recorded where they are used.

- **The heavy-rain loss weight, α = 0.18.** The paper says "weighted MSE" and never gives
  a value, so there is nothing to match. Ours is the first author's own
  (`config.yaml:21`, commit `dc5db56`). Recorded in D08's protocol.
- **The radar product is gauge-corrected.** So was the paper's (§4.1), so we match it
  already. It does mean a hidden station may be among the gauges that corrected the
  radar, so every radar result says so. Recorded in D09.

## Rules everyone follows

1. **Australian data only.** Singapore, NEA layouts, and commercial microwave links are
   permanently out of scope and are not restored behind a flag.
2. **Millimetres and UTC after 02.** Every timestamp downstream of Component 02 is
   timezone-aware UTC. Rainfall is millimetres.
3. **`station_id` is the key.** BoM's internal `station_id`. The public `station_no` is a
   display label and is not unique — two real Sydney stations share `563083`, and two
   share `567163`.
4. **Missing is not zero.** A missing observation is an absent row. `0.0` means the gauge
   was working and no rain fell. Never fill a gap with an estimate.
5. **One seed file.** All randomness comes from `config/seeds.yaml`. No literal seed in a
   module.
6. **One split.** Every method predicts the same rows, defined by the same split file. No
   component invents a private split.
7. **Say what you dropped.** If you exclude a station, an hour, or a file, it appears in
   your component's report with a reason. Not only in console output.
8. **Don't commit data.** `database/` and `experiments/` are gitignored. The only
   exception is the small test sample each component commits under `tests/data/`.
9. **Every run is a folder.** A result is only a result if it sits in its own run folder
   with the config and the input fingerprints that produced it. Nothing overwrites a
   result.
10. **Tests cover the code, and we measure it.** Line coverage of the production packages
    (`src/ingestion`, `src/canonical`, `src/study`, `src/graph`, `src/models`,
    `src/experiments`) stays at or above **80 %**. Any module that writes files other
    people's work depends on stays at or above **90 %**, because a bug there corrupts or
    overwrites data silently instead of crashing — ISS-28 and ISS-29 are that class. Those
    modules today: `src/ingestion/common.py`, `src/ingestion/gauge/fetch.py`,
    `src/ingestion/radar/fetch.py`, `src/canonical/gauge/build.py`,
    `src/canonical/grids.py`, `src/study/split.py`, `src/graph/builder_new.py`,
    `src/models/run.py`, and the runner `src/experiments/run.py` when it exists. Coverage
    is measured with `pytest-cov` on every run of the suite and both floors fail the run
    (D07.7 wires this in). Inherited review-only code (`benchmarks/`, `models/`,
    `training/`, `src/cml`, `src/radar`, `src/sampling`, `src/visualization`) is outside
    the measurement until it is classified. A number is a floor, not a goal: a test that
    reaches a line without checking what it did counts toward the number and proves
    nothing, so reviewers still read the tests.

## Working agreement

- Branch from `integration/sydney-gauge-baseline` as `feature/<subtask id>-<name>`,
  for example `feature/D08.1-blocked-folds`.
- Update `../CURRENT_STATE.md` and `../ISSUES.md` in the same pull request as the change
  that makes them wrong. The rules are in [../../AGENTS.md](../../AGENTS.md).
- Change a shared column or file path only after telling the owner who reads it.
- Requirements change when implementation proves them wrong. Edit the document in the
  same pull request and say what changed.
- Mark real uncertainty `TO CONFIRM` in the document rather than guessing.
- A component does not ship without a README under `docs/`.

## Reading order for a deliverable

1. This page: what counts as work, and the rules.
2. Your deliverable document, in full.
3. In the other deliverables, only the acceptance lines of the subtasks yours needs.
4. [GLOSSARY.md](GLOSSARY.md) whenever a word is unfamiliar.
5. [AUTHORING_STANDARD.md](AUTHORING_STANDARD.md) before you change any of these documents.
