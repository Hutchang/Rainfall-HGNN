# Component 03 test station tables

Two offline samples so the split tests run on a fresh clone with no network and no
dependency on Component 02 having produced `database/standard/stations.csv`.

| File | Rows | Purpose |
|---|---|---|
| `stations.csv` | 3 | Identifier and schema edge cases: a leading-zero `station_id`, a duplicated public `station_no`, and one `use_in_study = false` station carrying its `exclusion_reason`. |
| `stations_sydney.csv` | 162 (148 eligible, 14 excluded) | A full-size split. Large enough that five folds each hold a non-empty train, validation and test set, and that the ten spatial clusters are populated. |

Both files carry the 13 columns Component 02 writes
(`src/canonical/gauge/stations.py:16`), not just the 8 Component 03 consumes, so the
reader is exercised against the real input shape.

**`stations_sydney.csv` coordinates are synthetic.** Stations are scattered over six
Greater Sydney sub-regions (coastal east, inner west, Parramatta/west, south-west,
north shore, Blue Mountains) to give the spatial clustering a realistic shape. The
rainfall totals, coverage figures and station numbers are invented. It is fixture data
for testing the splitting logic and must never be quoted as BoM observation data or
used to produce a scientific result.

`exclusion_reason` strings are copied from the wording Component 02 actually emits, so
the exclusion record in the split artifact is tested against realistic text.
