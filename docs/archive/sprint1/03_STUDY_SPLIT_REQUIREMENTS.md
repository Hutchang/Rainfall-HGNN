# 03 — Pick the train, validation and test stations

**Owner:** _unassigned_ · **Reviewer:** Components 04 and 06 owners

## What this component is for

Decide which gauges the model learns from and which gauges it is tested on, write that
decision to a file, and make it reproducible.

This is the smallest component. It is one function plus a config file. It matters
because everything downstream is only comparable if every method is scored on the same
held-out stations — if two people generate their own splits, their numbers cannot be
compared and the whole experiment is void.

The task is **spatial interpolation**: hide some gauges entirely and reconstruct their
rainfall from the others. So the split is over *stations*, not over time. Every fold
uses the whole 2021–2022 period.

## What you receive

From Component 02, `database/standard/stations.csv`:

| Column you use | Why |
|---|---|
| `station_id` | the thing you are assigning |
| `latitude`, `longitude` | you split spatially, so test gauges are spread across Sydney instead of clustered in one suburb — a test set that is all coastal tells you nothing about the west |
| `use_in_study` | only these are eligible. A station excluded by Component 02 for poor coverage or a dead sensor never enters a split |

You do not read rainfall values. Nothing in this component looks at the data it is
splitting.

### Your own config

`config/splits/sydney_5fold.yaml`:

```yaml
stations_file: database/standard/stations.csv
n_folds: 5
validation_fraction: 0.15    # of the non-test stations in each fold
n_clusters: 10               # spatial clusters used to spread the test stations
output: config/splits/sydney_5fold.json
```

And `config/seeds.yaml`, which you own on behalf of the whole project:

```yaml
split_seed: 123
model_seed: 123
```

Every other component imports the seed from here. No module contains a literal seed.

## What you produce

`config/splits/sydney_5fold.json`:

```json
{
  "name": "sydney_5fold",
  "created": "2026-09-02",
  "stations_file": "database/standard/stations.csv",
  "n_eligible": 148,
  "seed": 123,
  "folds": [
    {"fold_index": 0,
     "ml": {"train": ["7730155", "..."],
            "validation": ["49645132", "..."],
            "test": ["566018", "..."]}},
    {"fold_index": 1, "ml": {"train": [], "validation": [], "test": []}}
  ]
}
```

This exact shape is not a free choice — `train_fused.py:60` already reads
`payload["folds"][i]["ml"]["train"]`, so keeping it means the existing runner works
unchanged.

Required properties, and these are what your tests check:

1. Every eligible station appears in `test` **exactly once** across the five folds. Every
   station therefore gets reconstructed once, and there is no accidentally favoured
   subset.
2. Within a fold, `train`, `validation` and `test` do not overlap.
3. Test stations are spread across the spatial clusters, not drawn from one region.
4. Same seed and same station list gives byte-identical output.

Also write `config/splits/sydney_5fold.md`: how many stations per fold, and a scatter
plot of the fold-0 assignment. One picture stops a lot of arguments about whether the
split is sane.

## How your output is used

**Component 04** reads the fold's three station lists and builds three graphs from them:
the train graph contains train stations only; the validation graph contains train plus
validation; the test graph contains all three. It needs the lists as station ids because
that is the only stable identity — node positions change between graphs.

**Component 06** uses the split to define exactly which rows every method must predict.
The target rows are defined by a rule, not by a file you write:

```python
targets = gauge_hourly[gauge_hourly.station_id.isin(fold["ml"]["test"])]
```

Every row of `gauge_hourly` for a test station is a target. Because `gauge_hourly` only
contains observed hours, this needs no extra filtering, and two people running it get
identical target sets without exchanging a file.

## Rules

1. The split file is the only source of station roles. Component 04 must never re-derive
   them.
2. A released split is never edited. Change the method or the seed and write a new name.
3. Only `use_in_study` stations are eligible.
4. Randomness comes from `config/seeds.yaml`.

## Done when

- [ ] `python -m study.split config/splits/sydney_5fold.yaml` writes the JSON and the
      summary.
- [ ] Test: the union of the five test sets equals the eligible station set, with no
      station appearing twice.
- [ ] Test: no station is in two roles within a fold.
- [ ] Test: running twice with the same seed produces identical JSON; changing the seed
      changes the assignment.
- [ ] Test: a station with `use_in_study = false` never appears anywhere.
- [ ] Tests use a small synthetic station table, so they do not wait for Component 02.
- [ ] `train_fused.py` runs against your split file without modification.

## Open questions

- **k-means on raw latitude/longitude degrees** is what the inherited code did. At
  Sydney's latitude one degree of longitude is about 93 km against 111 km for latitude,
  so degree-space clusters are slightly stretched east–west. Projecting to km first is a
  few lines and is the better default. Say which one you used in the summary; if you keep
  degrees, say why.
