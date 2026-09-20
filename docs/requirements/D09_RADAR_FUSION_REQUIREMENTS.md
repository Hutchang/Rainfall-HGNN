# D09 — Radar fusion

Template: study (see [AUTHORING_STANDARD.md](AUTHORING_STANDARD.md)). Owner: 
What counts as work, and the rules, are in [README.md](README.md). The data path — reading the radar and
satellite products on NCI and writing the grid tables — is D13; this deliverable is the
science that runs on those tables.

## Summary

Add radar, then satellite, to the gauge-only model and find out whether either helps on
Sydney under the paper's protocol. The system is built on the gauge-corrected radar
product we have — the same kind the paper used (§4.1); every result carries the caveat
below, and switching to an uncorrected
product later is a config change, not a rebuild.

## Question

Does HGNN(G+R) improve on HGNN(G) and on IDW for Sydney under the paper's protocol, and
does HGNN(G+R+S) improve further? The paper's Table 2 says no for their data (0.688 and
0.698 against 0.686 RMSE for G); their explanation is that Sydney's gauges already
resolve the field.

## Protocol

Same as D08 (split, which gauges each method may read, scaling, metrics), plus a radar node layer and then a
satellite node layer. Every result row states the product used (`prcp-m15`,
gauge-corrected) and carries the caveat below with D13's contributing-gauges count when
it is available.

## Inputs

| Input | From | State today |
|---|---|---|
| `radar_hourly.csv`, `radar_nodes.csv` | D13.2 | not yet produced |
| `satellite_hourly.csv`, `satellite_nodes.csv` | D13.3 | not yet produced |
| gauge tables, split, graphs | D08 | see D08 |

## Subtasks

| Subtask | What | Needs | Done when |
|---|---|---|---|
| **D09.1** | *Moved to D13.2* (radar reader, hourly sums, subsample rule, grid tables). The number is kept so references stay valid. | — | — |
| **D09.2** | Grid edges and edge direction in Component 04 — see below. | nothing; synthetic data suffices | Tests on the synthetic pipeline fixture pass; the graph report shows grid-to-grid edges and one-way cross-support edges. |
| **D09.3** | *Moved to D13.4* (the contributing-gauges count, now a line in the grid report rather than a decision input). | — | — |
| **D09.4** | Run HGNN(G+R) with D08's settled protocol on D13.2's tables. | D08.1, D08.2, D09.2, D13.2 | A G+R row beside G and IDW in the D08 table, with the product and the caveat stated. |
| **D09.5** | Run HGNN(G+R+S). | D09.4, D13.3 | A G+R+S row in the same table. |

### D09.2 — Grid edges and edge direction in Component 04

What the code does today: radar nodes are attached to gauges only. There are no edges
between neighbouring radar cells (`src/canonical/grids.py:38` leaves topology to the
builder; `src/graph/gaugegraphnew.py:294-336` builds only gauge-to-sensor edges). The
paper connects every grid cell to its eight surrounding cells so information moves
across the radar field. A builder for that exists (`src/graph/radargraphnew.py`) but is
used only by a test and gives coincident nodes a tiny weight where a large one is meant
(`:59`). Cross-support edges are built in both directions and the reverse direction is
compulsory (`src/models/modalities.py:64-79`); the paper directs them into the gauge
layer only. Grid-to-gauge neighbours are 4 (`config/graphs/gauge_baseline.yaml:14`); the
paper uses 9.

Must do: wire the eight-neighbour grid edges into `builder_new.py`; fix the weight at
`radargraphnew.py:59`; make the reverse cross-support edge optional and off by default;
set `layer_k_neighbours: 9`. Tests on the synthetic pipeline fixture. Needs nothing from
D08 or D13; it can be built and tested on synthetic data now.

## Deliverable

Added rows in `docs/results/B_sydney_gauge_baseline.md` (or a D09 section of it): G+R
and G+R+S beside G and IDW; the product used; the caveat with the count; the
subsample rule D13 recorded; run ids.

## Acceptance

- Same targets, same split, same visible gauges as D08; the only change is the added layer.
- Every radar or satellite result states the product, that it is gauge-corrected, and
  the contributing-gauges count if D13.4 has produced it; none of that is buried in an
  appendix.
- The coverage floors (README rule 10) hold for the graph changes in D09.2 at 80 %.

## Dependencies

D08.1 and D08.2 for the protocol. D13.2 for D09.4; D13.3 for D09.5.
Issues: ISS-02 and ISS-04 are D13's to close; D09 only inherits their caveats.

## This week and backlog

**This week:** D09.2.
**Backlog (week 7+):** D09.4, D09.5.

## Findings

*(filled in as the work runs)*
