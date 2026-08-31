# Client briefing — data and implementation issues found during baseline reconciliation

> **Implementation-scope update:** The component handoff now fixes the baseline study
> period at **2021-01-01 through 2022-12-31**. References below to regenerating the
> inherited “same period” are superseded by that decision. The findings about the
> 2021–2025 hand export remain evidence for replacing it.

**Prepared:** 31 August 2026 · **Team:** COMP3888 M17-01
**Scope:** everything found while auditing and integrating the two inherited codebases
(the two inherited rainfall-reconstruction implementations
implementation) against the source paper (arXiv:2607.01621, "Spatial Support Matters").
Written to be read aloud; supporting evidence for every claim is in the repository under
`docs/archive/baseline-reconciliation/`.

---

## The one-paragraph summary

Both inherited codebases now run, and we can train the fusion framework gauge-only on
the real Sydney data end-to-end. Getting there surfaced three kinds of problems: **data
problems** (most seriously: the hand-exported rainfall values are distorted by a
time- and station-varying factor — verified against BOM's own records — and we
recommend regenerating the dataset from BOM's API, which we have confirmed works; also
station identity conflicts, now resolvable, and five dead gauges), **implementation
problems** (most importantly, both codebases implement the paper's edge-weighting
equation incorrectly, and on the original Sydney data this made the neural network
learn nothing at all — we have fixed and validated this), and **comparability
problems** (our dataset is not the dataset the paper's Sydney results were computed
on, so numbers cannot be compared directly). None of these block progress, and the
main thing we need from you is a go-ahead, listed at the end.

---

## Part 1 — Data issues

### 1.1 The rainfall values are distorted by the original export — verified against BOM's own records, and not repairable by a scale factor

The data column is named `rainfall_mm`, but the data was downloaded by hand from BOM's
WaterData portal and no record of the export settings survives. The magnitudes are
plainly wrong as millimetres: the median station's annual total is 8,494 units, which
would be 8.5 metres of rain a year (Sydney gets 0.7–1.6 m), and the average rainy hour
reads 14 units, an absurd hourly intensity in mm.

With the WaterData API back online we compared our values directly against BOM's
tip-by-tip gauge records (BOM's are in genuine millimetres). The result: for a verified
storm in January 2023, our hourly values are **exactly six times** the true rainfall —
the signature of an export that summed six 10-minute *intensity* readings per hour
instead of averaging them. But the factor is **not constant**: other periods at the
same station don't fit ×6 against BOM's quality-controlled totals, and the fine
structure of the values shows the distortion varies across stations and across time.
In plain terms: **the dataset's values are internally inconsistent — the same true
rainfall can appear as different numbers depending on station and period — and no
single correction factor can repair it.** This is more serious than a mislabelled
unit, because the model's *training targets* are inconsistently scaled: it affects not
just absolute error numbers but, to some degree, everything trained on this data.

The good news is that the fix is now cheap and fully within reach: the WaterData API
serves the underlying tip-level data in true millimetres with full provenance, and we
have verified access, the complete station list, and the method. **We recommend
regenerating the rainfall dataset directly from the API** — a scripted, rate-limited
download of the same stations and period — which eliminates the units question
permanently and gives the project a reproducible data pipeline instead of a hand
download. We have deliberately applied no correction to the current data in the
meantime.

### 1.2 The suspicious maximum value is explained — but five dead stations were found

Earlier we worried that the dataset's largest value (exactly 150.0, at five different
stations) indicated quality-control clipping of genuine extremes. The BOM comparison
resolved this: the 150.0 we checked is a **real** storm hour — 25 mm/h of verified
rainfall (50 gauge tips), multiplied by the ×6 export artefact above. No evidence of
removed extremes remains.

However, the same investigation found that **five stations are entirely
zero-valued** for the whole 4.5 years (e.g. Jilliby Creek, 43,824 hours, not one
nonzero reading) — dead series that passed the data-completeness filter because zeros
count as "present". BOM's records show these gauges were live and reporting. These
stations feed the model nothing but false "always dry" signal and will be excluded.

A related note on timestamps: comparison with BOM shows the timestamps follow the
**local Sydney clock** — one hour ahead of standard time in summer (daylight saving),
aligned in winter. Harmless for modelling as long as all sources use the same
convention, but it must be stated when aligning radar or satellite data later.

### 1.3 Four stations have identity problems; three are excluded, with cause

- **Thirteen stations are registered twice** (under both Sydney Water and Water NSW
  networks) with identical data and coordinates differing by up to ~540 m. We collapse
  each to one entry; harmless, but the duplication previously caused a subtle evaluation
  leak (see 2.2).
- **Two station numbers (`563083`, `567163`) are genuine BOM-side ID collisions**: each
  public number corresponds to two real, physically distant sites. The API check
  confirmed this and gave us the missing disambiguator — BOM internally assigns each
  site its own distinct identifier (`563083` = Blackheath Golf Course *and* North
  Richmond Water Filtration Plant; `567163` = Regentville *and* Regentville RFS, each
  with exact coordinates). Since the API serves each internal site's data separately,
  we can now determine which site our series belongs to by direct comparison — this is
  resolvable by us, no longer waiting on anyone.
- **One station (`563086`, Greaves Creek)** has a real data series but its listed
  coordinate was a copy-paste of another station's. The API check found the station in
  BOM's system with its **true coordinate (−33.6547, 150.3089, Blue Mountains)** — so
  this one is now simply fixable.

Net effect: **170 of 173 stations currently pass into the integrated pipeline; 3 are
excluded with documented cause** — and all three exclusions can now be lifted as part
of the recommended dataset regeneration (1.1), using BOM's internal identifiers.
Separately, the five dead all-zero stations from 1.2 must come out.

### 1.4 Provenance generally

There is no download script, no API usage, and no export metadata in either repository —
the data cannot be re-fetched or independently verified as delivered. The timestamps'
timezone is also undocumented. Neither blocks work, but both belong in any future data
handover.

---

## Part 2 — Implementation issues, including deviations from the paper

### 2.1 The paper's edge-weighting equation was implemented incorrectly in both codebases — found, fixed, validated

The paper (Section 3.2, Equation 1) weights graph connections by inverse distance,
normalised **per node**: each station's nearest neighbour gets weight 1. Both inherited
codebases instead divide every weight by the **single largest weight in the whole
graph**. On the original Sydney data this was catastrophic, because two stations sat
5 centimetres apart (the copy-pasted coordinate from 1.3), producing one enormous weight
that crushed 99.8% of all other connections to near zero. The network stopped using its
graph entirely: the inherited Sydney model scored RMSE 8.96 with correlation −0.01 —
statistically indistinguishable from predicting a constant. With the per-node fix, the
same pipeline scores RMSE 5.30 with correlation 0.80 — genuine learning. The fix is
implemented in both codebases as an explicit option, with the original behaviour
preserved as the default so before/after comparisons stay honest.

One important nuance discovered later: once the misplaced-coordinate station is excluded
(1.3), the global normalisation is no longer degenerate — the incorrect formula then
performs comparably to the correct one at this network density. So the original headline
failure was a data defect and a code defect compounding each other.

### 2.2 Evaluation-integrity issues (all documented, most still open by design)

These are defects we have **flagged but deliberately not silently fixed**, because
correcting them changes results and must be done as labelled, separate experiments:

- **Duplicate-station leakage:** because 13 stations had two metadata entries, the
  original spatial cross-validation could place the *same station* in both training and
  test sets — its own reference split provably assigns one station to the test set of
  two different folds. Our integration collapses duplicates and resolves conflicts
  conservatively.
- **The validity flag is destroyed:** the fusion codebase marks each reading as
  observed-vs-missing, but then z-score normalisation flattens that flag to a constant —
  the model cannot distinguish "genuinely dry" from "no observation". This deviates from
  the paper, which relies on that flag.
- **Positional encodings are computed inconsistently:** the fusion codebase computes
  graph positional features separately per data split, making them mathematically
  incomparable between training and test graphs. The Sydney codebase fixed this; the
  fusion codebase has not.
- **No random-seed control:** model initialisation and dropout are unseeded in the
  fusion codebase, so no run is exactly reproducible.
- **The baseline comparison is not apples-to-apples:** the inherited IDW baseline sees
  ~18% fewer stations than the neural model at test time, is scored against differently
  resampled ground truth, and over a different timestamp set. Any inherited "GNN vs IDW"
  numbers should be treated as indicative only.
- **Structural fragility:** the prediction target is physically the same array as the
  input feature, protected only by a runtime masking step with no safety assertion —
  currently correct everywhere, but one line away from total leakage. A test suite now
  encodes each known defect so regressions are caught.

### 2.3 Deviations from the paper's experimental setup that cannot be reconciled from our side

- **Different dataset:** the paper's Sydney testbed is **calendar year 2022 with 75
  gauges** (retrieved February 2026). Our data is **2021–2025 with 173 stations** from
  an earlier retrieval. Different station population, different period, and possibly a
  different export format — the paper's Sydney numbers cannot be directly compared to
  anything we produce on this data.
- **No satellite pathway exists in code:** the paper's Sydney experiment fuses gauges,
  radar and satellite (G+R+S). The inherited codebase has no satellite implementation at
  all, and its radar preprocessing depends on a step that exists only inside a notebook.
- **Hyperparameters:** the Sydney codebase's defaults differ from the paper's stated
  settings (fewer layers, higher learning rate, no weight decay or dropout); the fusion
  codebase matches the paper.
- **An open question that cuts both ways:** since the codebase implements the *wrong*
  normalisation but the paper describes the right one, either the paper's numbers came
  from code we haven't seen, or they were produced with the global normalisation
  described above. On the paper's 75-gauge dataset — which may not contain the
  degenerate station pair — the difference may have been small, but it is worth
  confirming with the authors.

### 2.4 Where we are now

The two codebases are integrated to the point that the fusion framework trains genuinely
gauge-only on the real Sydney data, end-to-end, on GPU, using the corrected station set
and the original spatial cross-validation folds. Five-fold results are healthy
in relative terms (correlations 0.73–0.80 on the best runs — correlation is far less
sensitive to the value distortion of 1.1 than error magnitudes are, though with a
varying factor it is not fully immune either). We do not present error magnitudes in mm/h at all
until the data is regenerated, and we do not present any of this as science yet: runs
were at reduced training length, on the distorted values, with the
evaluation-integrity items of 2.2 still in their inherited state. It is also worth
saying plainly: the paper's own Sydney finding is that **fusion adds little where the
gauge network already resolves the rainfall field** — Sydney was the paper's illustration
of the method's limits, not its strengths — and our denser 170-station network is even
further inside that regime. The interesting scientific territory for the Australia-wide
objective is where gauges are sparse, which is exactly where the paper predicts fusion
pays off.

---

## Part 3 — What we need from you

1. **Approval to regenerate the rainfall dataset from the WaterData API** (our
   recommendation, see 1.1): the approved Greater Sydney study population for
   **2021–2022**, downloaded scripted and
   rate-limited from BOM's tip-level records in true millimetres, with full provenance
   and the three identity problems resolved via BOM's internal identifiers. This
   replaces the distorted hand export and permanently closes the units, extremes,
   dead-station and identity questions. Estimated effort: roughly a day, mostly
   unattended downloading.
2. **How the original export was produced** (optional but useful): if the person who
   downloaded `all_stations_rainfall_hourly_combined.csv` can recall the portal
   settings used, it would confirm our ×6 intensity-sum diagnosis and tell us whether
   any prior results can be salvaged quantitatively rather than re-run.
3. **The paper's Sydney gauge list** (75 stations, Feb 2026 retrieval), if reproducing
   the paper's Table 2 matters to you — without it, only qualitative comparison is
   possible.
4. **A decision on the corrected-experiment policy:** we propose reporting inherited
   behaviour as the baseline and running corrections (per-node normalisation, validity
   flag, positional encodings, seeding) as separately labelled experiments — and, given
   1.1, re-running key results on the regenerated data. Please confirm this is the
   reporting you want.
