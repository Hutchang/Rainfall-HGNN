# Decisions

Choices the project lead has made, with the evidence behind each. This is the only place
a decision is recorded.

[ISSUES.md](ISSUES.md) holds what is broken, missing or undecided. When a decision
settles an issue, the issue points here and is updated or closed; it does not repeat the
decision. [CURRENT_STATE.md](CURRENT_STATE.md) says whether the code applies it yet.

## Conventions

- **ID** is permanent (`DEC-xx`). Cite it in commits, PRs and other documents.
- An entry is not edited to change its meaning. A changed decision is a new entry that
  names the one it replaces, and the old entry gets one line saying so.
- Each entry gives the date, who decided, the decision, why, what it costs, and what it
  changes.

## Index

| ID | Decision | Date |
|---|---|---|
| [DEC-01](#dec-01) | Keep radar; drop hours that have no radar, for every method | 2026-10-11 |
| [DEC-02](#dec-02) | Satellite hour = mean of the 10-minute images that have a value, if at least 3 of 6 do | 2026-10-11 |
| [DEC-03](#dec-03) | Sydney uses the round-robin split | 2026-10-11 |
| [DEC-04](#dec-04) | Sydney 2022 is the baseline; 2021-2022 and raw radar are extensions | 2026-10-19 |

---

<a id="dec-01"></a>

### DEC-01 — Keep radar; drop hours that have no radar, for every method
**Date:** 2026-10-11 · **Decided by:** project lead · **Settles:** the radar part of
[ISS-04](ISSUES.md#iss-04)

**Decision.** Radar stays a model input. Any hour with no radar value is taken out of the
shared timeline for every method, gauge-only HGNN, IDW and network mean included, and the
run report lists each dropped hour with its cause. Missing radar is never filled with 0.

**Why.** NCI's file listing has no radar file for site 71 on 11 days of 2021–2022
(ISS-04). Each missing daily file loses 25 hours, that UTC day and the 23:00 hour before
it: 272 of 17,496 hours (1.55 %). Measured on the D08 gauge-only run (blocked split,
code `6f967be`), those hours are ordinary:

- they hold 1.29 % of observed rain, and 1.62 % of station-hours with 10 mm or more;
- dropping them moves each method's RMSE (mean over folds) by at most 0.002 mm: HGNN
  0.9026 → 0.9025, IDW 0.7305 → 0.7285, network mean 0.8388 → 0.8371. The ranking
  does not change.

The alternative, a radar validity flag ([ISS-36](ISSUES.md#iss-36)), keeps those hours
but changes the paper's inputs and needs Components 04 and 06 to accept a second radar
feature, for hours that carry nothing special.

**Cost.** About 1.5 % fewer hours. If the first real-archive run also finds radar
missing inside the files that are present, those hours are dropped too, and the count
may grow; that would be a reason to revisit ISS-36.

**Changes.** The timeline Components 03 and 04 work from. The gauge-only table must be
scored on the same, shorter set of hours before it sits beside a G+R row.

---

<a id="dec-02"></a>

### DEC-02 — Satellite hour = mean of the 10-minute images that have a value, if at least 3 of 6 do
**Date:** 2026-10-11 · **Decided by:** project lead · **Settles:** the rule in
[ISS-37](ISSUES.md#iss-37) and the satellite part of [ISS-04](ISSUES.md#iss-04)

**Decision.** For each cell and hour, the hourly value is the mean of the 10-minute
`crrph_intensity` images that have a value at that cell. If fewer than 3 of the six do,
less than half the hour was seen, and the cell-hour is invalid. A missing image counts as
an image with no value at every cell. "No value" is never treated as 0: the product
writes 0 itself for clear sky and for cloud too thin to rain.

**Why.**

- The archive has no image at 02:40 or 14:40 UTC on any day checked (142 of 144 files on
  2021-03-15, 2022-01-03 and 2022-07-20). Under the old rule, which needs all six, the
  02:00 and 14:00 UTC hours are never written, about 1,460 hours, and Component 04
  refuses a timeline with any hour missing.
- By day about a fifth of the cells in each image have no value, because the product
  cannot measure that cloud (ISS-37). Under the old rule about half the cells of a dry
  daytime hour are invalid. These cells follow cloud, so the old rule throws away cloudy,
  rainy hours more than clear ones.
- Three of six means at least half the hour was seen. An invalid cell is not lost from
  the model: its validity flag tells the model to ignore it for that hour.
- The paper says only that it used CRRPh at 2 km, hourly (§4.1); it does not describe
  how images become hours or what it did with gaps.

**Cost.** A cell-hour may rest on 3 to 5 images. If the moments without a value were the
rainy ones, the hour reads too dry. Neither the size nor the direction of this is
measured. Every satellite result states this rule.

**Changes.** The satellite reader (`src/canonical/satellite_archive.py`), the threshold
recorded in `config/canonical/grids.yaml`, and the grid report's hourly rule.

---

<a id="dec-03"></a>

### DEC-03 — Sydney uses the round-robin split
**Date:** 2026-10-11 · **Decided by:** project lead · **Settles:**
[ISS-10](ISSUES.md#iss-10) · **Replaces:** D08's Gap 1 fix

**Decision.** Sydney experiments (D08, D09) use the split built from
`config/splits/sydney_5fold_round_robin.yaml`: each location cluster's gauges are dealt
one at a time across the five folds, so each fold hides about a fifth of every area. The
blocked split (`sydney_5fold_blocked.yaml`, whole areas hidden) stays as a comparison, not
the reported protocol. How to split for Australia-wide work (D10) is decided later.

**Why.** The paper's text (§4.2) says folds keep hidden gauges away from training gauges;
the first author's code hides one gauge per cluster per fold, which does not. The two
cannot both be reproduced, and too much else differs from the paper to reproduce it
exactly, so the split was chosen by what it measures. A fair test hides gauges about as
far from the nearest readable gauge as the places the method would really predict.
Distance from each hidden gauge to the nearest gauge a method can read, against distance
from any spot in Sydney's network to its nearest gauge:

| | Median | 90 % within | Furthest |
|---|---|---|---|
| Blocked split | 8.0 km | 14.2 km | 29.3 km |
| Round-robin split | 3.6 km | 7.4 km | 9.9 km |
| Any spot inside the outline of the 127 gauges, all gauges readable | 3.3 km | 8.1 km | 17.4 km |

Round-robin matches the real situation; blocked tests a harder job than Sydney's
network ever faces. Round-robin also matches the first author's code, and their result:
IDW reaches r = 0.846 on round-robin folds against the paper's 0.854, and 0.70 on the
blocked folds (commit `3ec5415`).

How this was measured (a one-off check; D08.6's tool is the formal version): station
positions from the BoM station list (`database/raw/gauge/stations.json`); both splits
rebuilt in memory from their configs and `config/seeds.yaml` (the blocked rebuild matched
the released `sydney_5fold_blocked.json` exactly); the spots are a 0.01° grid inside the
outline of the 127 gauges.

**Cost.** We knowingly depart from the paper's text on how folds are formed. Four of the
127 hidden gauges have a readable gauge within 0.5 km, probably a second gauge at the
same site, which every method predicts almost for free.

**Changes.** D08's Gap 1 and the Component 03 README now name round-robin as Sydney's
split. `sydney_5fold_round_robin.json` has to be built from the real `stations.csv` and
committed before results are reported (rules 5 and 6). D08.7 still scores both splits.

<a id="dec-04"></a>

### DEC-04 — Sydney 2022 is the baseline; 2021-2022 and raw radar are extensions
**Date:** 2026-10-19 · **Decided by:** project lead

**Decision.** The baseline Sydney experiments are G, G+R and G+R+S on 2022 alone:
`config/models/d09_2022_round_robin_{g,gr,grs}.yaml`, reading the tables built by
`config/canonical/standard_2022.yaml` (published on Gadi as `sydney-2022`) and the 2022
round-robin split `config/splits/sydney_5fold_round_robin_2022.json` (DEC-03). Every
other Sydney experiment is an extension of that baseline and is reported against it:

- **2021-2022**: the same sources over two years
  (`config/models/d09_round_robin_*.yaml`, tables published as `sydney-2021-2022`).
- **Raw radar**: G+R with radar computed from level-2 reflectivity instead of the
  gauge-corrected product (`feature/raw-radar-level2`, built on the 2022 tables), so
  the radar has not seen the gauges it is tested on.
- Ablations of model settings run on 2022 unless they say otherwise.

**Why.** It is the closest we can come to the paper's Sydney testbed (§4.1): 2022 at
hourly resolution, gauge-corrected Rainfields 3 radar at about 5 km and Himawari CRRPh
satellite at 2 km, with the paper's graph and model settings (§4.3).

**Cost.** It still differs from the paper: 128 gauges over about 100 by 100 km against
the paper's 75 over about 70 by 55 km, because the paper's station list is not available
(ISS-23), and folds follow DEC-03 rather than the paper's text. One year gives each fold
half the training hours of 2021-2022.

**Changes.** Results tables lead with the 2022 runs. Extensions name the baseline run
they are compared with.

