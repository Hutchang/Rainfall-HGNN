# Split summary: sydney_5fold_round_robin

- stations file: `database/standard/stations.csv`
- eligible stations: 127; excluded: 57
- folds: 5; clusters: 10; fold assignment: `round_robin`
- validation fraction: 0.15; seed: 123
- created: 2026-10-15

## Stations per fold

| Fold | Train | Validation | Hidden (test) |
|---|---|---|---|
| 0 | 86 | 15 | 26 |
| 1 | 86 | 15 | 26 |
| 2 | 87 | 15 | 25 |
| 3 | 87 | 15 | 25 |
| 4 | 87 | 15 | 25 |

## Fold map

![Fold map for sydney_5fold_round_robin](sydney_5fold_round_robin.png)

Left: every eligible station, coloured by the fold that hides it. Right: fold 0's roles.
