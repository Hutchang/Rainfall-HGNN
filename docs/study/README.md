# Component 03: study splits

Decides which gauges are hidden, which train the model and which stop training, for
each cross-validation fold. Every later component reads the one file this writes, so
every method predicts the same hidden gauges.
Sprint 1 contract: [03_STUDY_SPLIT_REQUIREMENTS.md](../archive/sprint1/03_STUDY_SPLIT_REQUIREMENTS.md)
(archived). The Sprint 2 change is D08.1 in
[D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md](../requirements/D08_SYDNEY_GAUGE_BASELINE_REQUIREMENTS.md).

## Command

```
uv run python -m src.study.split config/splits/sydney_5fold_round_robin.yaml
uv run python -m src.study.split config/splits/sydney_5fold_blocked.yaml
```

Sydney's split is round-robin ([DEC-03](../DECISIONS.md#dec-03)); the blocked split is
kept as the comparison D08.7 scores.

Reads `database/standard/stations.csv` (Component 02) and `config/seeds.yaml`. Tests run
offline against `tests/data/study/`:

```
uv run python -m pytest tests/test_study_clustering.py tests/test_study_schema.py -v
```

## What it writes

Beside the config, named after its `output`:

| File | Committed | Contents |
|---|---|---|
| `<name>.json` | yes | Per fold: `train`, `validation` and `test` (hidden) station ids. Also the seed, the eligible count, and every excluded station with Component 02's reason. |
| `<name>.md` | no | Stations per fold and the settings used. |
| `<name>.png` | no | Fold map: which fold hides each station, and fold 0's roles. |

The `.md` and `.png` are regenerated on every run. A released `.json` is never
overwritten: the command leaves an identical file alone and refuses a different one, so a
changed rule needs a new config and a new name. The `created` date is UTC and is part of
the file; pass `--created` with the original date to rerun a released split on a later day.

## How a split is made

1. Keep the stations Component 02 marks `use_in_study`.
2. Group them into `n_clusters` areas with k-means on coordinates projected to kilometres.
3. Choose each fold's hidden stations from those areas (`fold_assignment`, below).
4. In each fold, draw `validation_fraction` of the remaining stations at random as
   validation; the rest are train.

All randomness comes from `split_seed` in `config/seeds.yaml`.

## `fold_assignment`: the two rules

| Value | Each fold hides | Config |
|---|---|---|
| `whole_clusters` | `n_clusters / n_folds` whole areas. No hidden station has a station from its own area in train or validation. What the paper's text describes (§4.2); kept as the comparison. | `sydney_5fold_blocked.yaml` |
| `round_robin` (default) | About a fifth of every area, dealt one station at a time. Every hidden station keeps neighbours from its own area visible. What the first author's code does, and Sydney's split ([DEC-03](../DECISIONS.md#dec-03)). | `sydney_5fold_round_robin.yaml` (127 stations); `sydney_5fold.yaml` is the inherited one over 148 |

`whole_clusters` needs `n_clusters` to divide evenly by `n_folds` and refuses otherwise.

`sydney_5fold_blocked.json` covers the 127 eligible stations of the 2026-09-20 build:

| Fold | Train | Validation | Hidden |
|---|---|---|---|
| 0 | 93 | 16 | 18 |
| 1 | 84 | 15 | 28 |
| 2 | 88 | 15 | 24 |
| 3 | 87 | 15 | 25 |
| 4 | 81 | 14 | 32 |

Fold sizes are uneven because whole areas are dealt and areas differ in size.

## Known limits

- Areas touch, so a hidden station on the edge of its area can still have a visible
  station a few kilometres away. How far is measured by D08.6 and D08.7 (ISS-10).
- Validation stations are placed at random and can land beside a hidden area (ISS-34).
- How areas are drawn, how many, and which pair up in a fold are our choices; the paper
  does not say. Each is marked in `sydney_5fold_blocked.yaml`.
- The paper's own station list was never published (ISS-23).
