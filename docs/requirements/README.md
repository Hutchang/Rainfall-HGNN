# Component requirements

**Baseline period:** 2021-01-01 through 2022-12-31 (gauges).
**Radar and satellite currently cover 2022 only.**

Five components, one straight line. Each is owned end-to-end by one teammate.

```text
BoM API + local radar/satellite files
                |
            [01] get raw data onto disk
                |  raw files + index.csv
                v
            [02] build the standard tables
                |  hourly rainfall, station list, radar/satellite grids
                +-------------------------> [03] pick train/val/test stations
                |                                    |  split.json
                v                                    v
            [04] build graph samples <---------------+
                |  PyG HeteroData
                v
            [06] train and write predictions
                   predictions.csv
```

There is no Component 05. The golden-fixture component was cut on 2026-09-01; the small
committed test sample it was meant to produce is now one bullet in each component's
"Done when" list. The old document is in `../archive/`.

## The whole contract on one page

| # | Reads | Writes | Read by |
|---|---|---|---|
| [01](01_DATA_SOURCE_INGESTION_REQUIREMENTS.md) | BoM WaterData API; NCI THREDDS radar archive; satellite file already on disk | `database/raw/**` + `database/raw/index.csv` | 02 |
| [02](02_CANONICAL_DATA_REQUIREMENTS.md) | `database/raw/**` | `database/standard/*.csv` | 03, 04 |
| [03](03_STUDY_SPLIT_REQUIREMENTS.md) | `standard/stations.csv` | `config/splits/<name>.json`, `config/seeds.yaml` | 04, 06 |
| [04](04_GRAPH_SAMPLE_REQUIREMENTS.md) | `standard/*.csv` + split JSON | PyG `HeteroData` per fold and role | 06 |
| [06](06_MODEL_ADAPTER_REQUIREMENTS.md) | 04 samples + split JSON | `experiments/<run>/predictions.csv` | evaluation |

If you are stuck on what to build, read your own document plus the **What you produce**
section of the component above you. That is enough. You do not need the other three.

## Rules everyone follows

1. **Australian data only.** Singapore, NEA layouts, and commercial microwave links are
   permanently out of scope and are not restored behind a flag.
2. **Millimetres and UTC after 02.** Every timestamp downstream of 02 is timezone-aware
   UTC. Rainfall is millimetres.
3. **`station_id` is the key.** BoM's internal `station_id`. Public `station_no` is a
   display label and is not unique — two real Sydney stations share `563083`, and two
   share `567163`.
4. **Missing is not zero.** A missing observation is an absent row. `0.0` means the gauge
   was working and no rain fell. Never fill a gap with an estimate.
5. **One seed file.** All randomness comes from `config/seeds.yaml`. No literal seed in
   a module.
6. **One split.** Every method predicts the same rows, defined by the same split file.
   No component invents a private split.
7. **Say what you dropped.** If you exclude a station, an hour, or a file, it appears in
   your component's report with a reason. Not only in console output.
8. **Don't commit data.** `database/` and `experiments/` are gitignored. The only
   exception is the small test sample each component commits under `tests/data/`.

## Working agreement

- Branch from `integration/data-pipeline` as `feature/0X-<name>`.
- Change a shared column or file path only after telling the owner who reads it.
- Requirements change when implementation proves them wrong. Edit the document in the
  same pull request and say what changed.
- Mark real uncertainty `TO CONFIRM` in the document rather than guessing.

[GLOSSARY.md](GLOSSARY.md) defines BoM, KiWIS, tip, rain day, HeteroData, LPE, and the
rest of the vocabulary.
