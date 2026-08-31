# DATA INVENTORY — both implementations

Covers FR-003 (filesystem data dependencies), FR-004 (data dictionary) and the
FR-604 2D gap list. Companion to `AUDIT_NIRAMAY.md`, `AUDIT_JYLOW.md`, `SYSTEM_MAP.md`.

**Headline:** Niramay bundles everything it needs (after regenerating one CSV).
Jylow bundles **nothing** — `database/`, `local/`, `experiments/`, `results/`, `logs/`,
`idw_results/` and `testing/` all absent, with `*.csv` gitignored. There are **zero**
network fetches and **zero** environment variables in either tree; every dependency is a
local path that must be supplied by hand.

---

## 1. Asset classification (FR-003)

### Niramay — self-contained

| Asset | Path | Class |
|---|---|---|
| Per-station gauge CSVs | `database/australia/raw/*.csv` (173 files) | **bundled** |
| Station metadata | `database/australia/station_metadata.csv` (425 rows) | **bundled** |
| Combined gauge CSV | `all_stations_rainfall_hourly_combined.csv` | **generated-by-preprocessing** — removed in `8e90ef0`; regenerate by concatenating the 173 raw files. Verified: 7,581,552 rows, 235.2 MB, sha256 `93e692ca97c15c98f6c21abeada4f0eb7100d8c54d3d03aba0b3cece9941c097` |
| Optuna DB / experiments | `optuna_australia.db`, `experiments/` | generated, gitignored |

**No NCI dependency.** The gauge-only baseline is fully reproducible offline.

### Jylow — nothing present

| Asset | Path (hardcoded) | Support | Class |
|---|---|---|---|
| Gauge observations | `database/raingauge_nea_data/{year}/weather_station_data_{year}.csv` | 0D | **missing** — NEA Singapore dump, no download code in repo |
| Gauge metadata | `database/raingauge_nea_data/{year}/weather_stations_{year}.csv` | 0D | **missing** |
| CML | `database/combined_microwave_link_data_2024_2025.nc` | 1D | **missing** |
| Radar (processed) | `database/processed_radar_dataset.pkl` | 2D | **generated** — only writer is `src/radar/utils.py:74`, whose only caller is `eda.ipynb` cell 14 |
| Radar (raw) | `database/area_23/**/*.tif`, `database/sg_radar_data/**` | 2D | **missing** |
| Basemap | `database/NationalMapPolygon.geojson` | — | **downloadable-public** (data.gov.sg) |

**Satellite: none.** An exhaustive grep for `satellite|himawari|gsmap|imerg|gpm|persiann`
across every `.py` and `.ipynb` returns zero hits outside a paper citation. Jylow's "2D"
is **radar only** — so the paper's Sydney `G+R+S` row has no implementation here.

**Three different CML filenames** appear across the tree
(`combined_microwave_link_data_2024_2025.nc`, `CML_data_Feb2025-April2025.nc`,
`cml_data_processed_2025.nc`) with nothing to disambiguate them.

### Recoverable from git history
Niramay's Sydney data is reachable from this repo without crossing branches:
```bash
git ls-tree -r --name-only cc23a25 -- database/australia   # 175 files
```
Useful for the golden fixture (FR-401/402).

---

## 2. 🔑 Jylow's 0D gauge contract — the critical interface

This is what the adapter (FR-301) must produce. Two CSVs per year, **both required**.

### A. `weather_station_data_{year}.csv` — observations
Read with a bare `pd.read_csv()` — no `dtype=`, `parse_dates=`, `na_values=`, `index_col=`
(`src/raingauge/utils.py:24`).

| Column | Dtype | Contract |
|---|---|---|
| `timestamp` | str | `strptime(x, "%Y-%m-%dT%H:%M:00+08:00")` (`:30-32`). **The `00` seconds and `+08:00` offset are literal characters, not directives.** Verified against Python 3.12: `2024-06-17T00:05:00+08:00` parses; `…:05:30+08:00`, `…+10:00` and `2024-06-17 00:05:00` all raise `ValueError`. Result is **tz-naive SGT wall-clock**. |
| `stationId` | str | becomes column labels via `pivot`. NEA-style `S##`. |
| `value` | numeric | rainfall **depth in mm over a 5-minute interval**; `× 12` at `:33` → **mm/hr**. |

- **Missing values are encoded by row absence**, not a sentinel. The pivot yields `NaN`;
  `notna()` becomes the validity flag. An explicit `-9999` would be silently treated as a
  real reading.
- **Duplicate `(timestamp, stationId)` raises** `ValueError: Index contains duplicate entries`.
- Observed real shape: 209,691 five-minute timestamps × 70 stations → **51 stations**
  after the 0.9 uptime filter.

### B. `weather_stations_{year}.csv` — metadata

| Column | Contract |
|---|---|
| `id` | must match `stationId` exactly |
| `latitude`, `longitude` | decimal degrees, WGS84 |
| `deviceId` | **required** — `drop(columns='deviceId')` at `:97` raises `KeyError` if absent, **even though the value is never used** |
| 4th column (likely `name`) | inferred from a dummy analogue in `generate_grid.ipynb`; moderate confidence |

**Two latent loader bugs:** `load_raingauge_dataset` raises `NameError` when
`uptime_threshold` is `0`/`None` (`filtered_stations_index` bound only inside the `if` at
`:41-43`, referenced at `:49`); and `get_station_coordinate_mappings()` defaults to
`start=0, end=0`, reading `database/raingauge_nea_data/0/weather_stations_0.csv`.

---

## 3. Niramay → Jylow field mapping (seeds FR-301)

| Niramay | Jylow expects | Transformation required |
|---|---|---|
| `timestamp` = `2021-01-01 00:00:00`, tz-naive | `2021-01-01T00:00:00+08:00`, seconds forced to `00` | **reformat** — otherwise `ValueError`. Note this spoofs an SGT offset onto Sydney data; the offset is discarded on parse, so it is cosmetic, but it must be recorded. |
| `station_id` int64 `211010` | `stationId` str | cast to str; IDs are otherwise opaque to the code |
| `rainfall_mm`, **hourly**, units suspect | `value`, **5-min mm**, then `×12` | ⚠️ **must not reuse `×12`.** Two compounding scale risks — see §5 |
| long format, one combined CSV | long format, per-year files | reshape + split by year (2021–2025 ⇒ 5 files) |
| metadata: headerless `(name,id,network,lat,lon)`, **425 rows / 409 unique** | headed `id,latitude,longitude,deviceId,name` | add header, **de-duplicate**, synthesise `deviceId` |
| missing = NaN (0.04%) | missing = row absent | drop NaN rows rather than emitting them |
| 173 stations | 51 (Singapore) | `n_clusters=8` in the sampler is tuned for 51 — revisit |

---

## 4. Data dictionary — model tensors

| Tensor | Niramay | Jylow |
|---|---|---|
| gauge feature 0 | `log1p(rainfall)` | rainfall mm/hr, **z-scored** |
| gauge feature 1 | validity flag, preserved | validity flag, **z-scored → collapses** (see §5) |
| gauge features 2-5 | Laplacian PE k=4, **computed once on the full graph** | Laplacian PE k=4, **computed per split** |
| target `y` | raw rainfall, `[N,T,1]` | raw mm/hr, `[N,T,1]`, never normalised |
| radar `x` | — | `[1550, T, 1]`, **raw band-1, no dBZ→R conversion anywhere** |
| CML `x` | — | `[828, T, 13]`, mixed units (dB, GHz, km), one-hot polarization |

**Radar grid geometry** (verified numerically from `config.yaml` bounds): W=50 lons
(103.605…104.095), H=31 lats (1.495…1.195, **descending**), step 0.01° → **1550 nodes**,
matching the cached `(31, 50)` notebook output.

### Coordinate order — inconsistent across Jylow
| Site | Order |
|---|---|
| gauge KNN (`gaugegraphnew.py:103-107`) | **(lon, lat)** |
| gauge coords into `connect_graphs` (`:221-226`) | **(lon, lat)** |
| radar coords into `connect_graphs` (`:321`) | **(lat, lon)** ⚠️ |
| KMeans folds (`sampling/main.py:279`) | (lon, lat) |
| IDW baseline (`idw.py:85-92`) | (lat, lon) |

---

## 5. Silent-wrong-result risks, ranked for an Australian swap

1. **🛑 Unit ambiguity, compounding.** Niramay's `rainfall_mm` is **probably 0.1 mm**
   (`AUDIT_NIRAMAY.md` §8.2, ~10×), and Jylow's loader applies **`×12`** for 5-min→hourly.
   Reusing that path on hourly data could stack to a **~120× error**. Everything
   downstream is scale-coupled: `rain_threshold=0.5`, `weighted_mse` weights
   `1+α·log1p(target)`, and the config colour scales. **Blocks the port.**
2. **⚠️ Euclidean distance on raw degrees.** Used for gauge KNN, KMeans fold assignment,
   and every IDW weight. At Singapore (lat 1.3°) 1° lon ≈ 1° lat, so it is ~isotropic and
   harmless. **At Sydney (−34°) 1° lon ≈ 92 km vs 1° lat ≈ 111 km — ~20 % anisotropy**
   that silently distorts graph topology, fold assignment and IDW weights. No error, just
   different neighbours. *Niramay has the same issue.*
3. **⚠️ lat/lon order bug in radar↔gauge edges.** `gaugegraphnew.py:221-226` builds gauge
   coords as (lon,lat); `:321` builds radar coords as (lat,lon); both go into one
   `NearestNeighbors(metric='haversine')`. This is a **live bug in the shipped Singapore
   pipeline**, not merely a porting hazard. Irrelevant in 0D-only mode.
4. **Validity flag destroyed by normalisation.** Z-scoring a near-constant flag with
   `std.clamp(min=1e-8)` drives it to ≈0 for all nodes; masked and unmasked nodes become
   bit-identical on that channel.
5. **Mask token ≠ "missing".** Zeroing after z-scoring injects the **training mean**.
   Changing the missing-data fraction changes what the sentinel means.
6. **Fixed test-station count assumed** — `all_preds.reshape(-1, test_station_count)`
   (`logic_hetero.py:287`) silently reshapes into garbage if the test mask ever varies
   across timesteps.
7. **`_DATA_FEATURE_DIM = 2` hardcoded** — adding a gauge feature (elevation, say)
   silently mis-slices the mask and mis-sizes layer 1.
8. **`int(station[1:])`** (`src/utils.py:252,372`) raises `ValueError` on `GW040994`.
   Legacy paths only, not `train_fused.py` — but it will bite anyone reusing those helpers.
9. **IDW resamples `.first()` while the GNN resamples `.mean()`** — the "same" gauge
   series differs between model and baseline.
10. **Radar physical units undetermined from code** — no conversion anywhere; substituting
    a differently-scaled product changes nothing observable until the metrics move.

---

## 6. FR-604 — 2D acquisition gap list (prepare, do not implement)

What Jylow's 2D interface actually expects, for when NCI access arrives:

| Requirement | Jylow's expectation | Australian equivalent |
|---|---|---|
| Product | Singapore weather radar GeoTIFFs, `area_23` | BOM Rainfields 3 (the paper's Sydney run used gauge-corrected Rainfields 3, ~5 km, subsampled) |
| Variable | band 1, raw, **units undetermined** | must establish dBZ vs mm/hr before ingest |
| Grid | **0.01° hardcoded** (`radargraph.py:24-25`), derived from `bounds`, **ignoring the GeoTIFF transform** | Rainfields is ~5 km ≈ 0.05° — **the hardcoded step yields geographically wrong node coordinates with a correctly-shaped tensor** |
| Extent | 50 × 31 = 1550 cells | Greater Sydney ≈ 136 × 170 km; at 0.01° that is ~136 × 170 = 23,120 cells — **~15× more radar nodes**, with direct memory/runtime consequences given the O(N) leave-one-out loop |
| Ingest path | `.tif` → `process_radar_dataset` → `.pkl` | **notebook-only** (`eda.ipynb`); the `.py` preprocessor cannot import |
| Satellite | **not implemented at all** | Himawari 8/9 CRRPH (the paper's Sydney satellite source) would need a new node type from scratch |

**Do not port Singapore radar assumptions blindly.** Three are unsafe at Sydney: the
0.01° grid step, the fixed 50×31 extent, and the untyped band-1 units.

**Still requires human/NCI confirmation:** which Rainfields product and version; whether
the CRRPH satellite product is on NCI; the radar unit convention; and the study time range
(Niramay's gauges span 2021-01-01…2025-12-31, whereas the paper's Sydney run used 2022).

---

## 7. Open data questions

1. **Units of `rainfall_mm`** — blocks the port (§5.1). Needs provenance.
2. **Timezone** of Niramay's timestamps — undocumented.
3. **Why 173 stations** when the paper's Sydney testbed used 75?
4. **Which of the three CML `.nc` files** is canonical — moot for 0D, relevant later.
5. **4th column of `weather_stations_{year}.csv`** — inferred as `name`, unproven.
