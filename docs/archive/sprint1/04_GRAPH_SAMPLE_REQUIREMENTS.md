# 04 — Build the graph samples

**Owner:** _unassigned_ · **Reviewer:** Component 06 owner

## What this component is for

Turn tables into the graph objects the model consumes: nodes, edges, and feature
tensors, for each fold and each of the three roles.

A **graph** here is: one node per rain gauge, and an edge between gauges that are near
each other. The model predicts a hidden gauge's rainfall by passing messages along those
edges from its neighbours. Radar and satellite, when they are ready, become extra node
layers connected to the gauge nodes.

Most of this already exists in `src/graph/gaugegraphnew.py` and works. Your first job is
not to rewrite it — it is to feed it Component 02 data instead of the legacy CSV export,
and to make the choices it currently hard-codes explicit.

## What you receive

From Component 02, `database/standard/`, and from Component 03, the split JSON.

The existing builder's signature is the contract (`src/graph/gaugegraphnew.py:19`):

```python
GaugeGraphNew(data_df, mapping_df, split_info, knn, edge_normalisation)
```

| Argument | Exact shape it needs | Built from |
|---|---|---|
| `data_df` | wide DataFrame: index = timestamp, one column per `station_id`, values = `rainfall_mm`. **Missing hours must be `NaN`, not `0.0`** | `gauge_hourly.pivot(index="timestamp", columns="station_id", values="rainfall_mm")`. Load with `dtype={"station_id": str}` or the ids stop matching the split file |
| `mapping_df` | columns `id`, `longitude`, `latitude`, `order`, where `order` is `0..n-1` and fixes node ordering | `stations.csv` filtered to `use_in_study` |
| `split_info` | `{"ml": {"train": [...], "validation": [...], "test": [...]}}` for one fold | `split.json["folds"][i]` |
| `knn` | int, currently 5 | your config |
| `edge_normalisation` | `"global"` or `"per_node"` | your config |

The `NaN` requirement is why Component 02 omits missing hours instead of writing zeros:
`fill_heterodata` reads `notna()` as the validity channel
(`gaugegraphnew.py:159`). A zero-filled input silently tells the model that a broken
gauge measured no rain.

### Radar and satellite

`add_heterodata(heterodata_layer, coords, layer_name, knn)` attaches another node layer
and connects it to the gauge nodes by nearest neighbour. It needs:

- `heterodata_layer[layer_name].x` — a tensor shaped `[nodes, time, features]`
- `coords` — a DataFrame with `latitude` and `longitude`, one row per node, same order

From Component 02's tables that is:

| Layer | `x` shape | `coords` |
|---|---|---|
| radar | `[952, T, 1]` — the rain value | `radar_nodes.csv` |
| satellite | `[726, T, 2]` — rain value and the source's own 0/1 availability flag | `satellite_nodes.csv` — **does not exist yet** |

So the data we have already matches what the code wants, with one gap: satellite has no
coordinates, so satellite nodes cannot be placed or connected. Build gauge-only first and
radar second; satellite is blocked on Component 01.

The model takes per-layer input widths — `GNNInductiveHetero(in_channels_dict, ...)`
(`models/gnn.py:276`) — so adding a layer means adding a dict entry, not reshaping
everything.

### Your own config

`config/graphs/gauge_baseline.yaml`:

```yaml
k_neighbours: 5
distance: haversine_km        # or euclidean_degrees for the inherited comparison
edge_normalisation: per_node  # or global for the inherited comparison
include_lpe: true
lpe_dimensions: 4
```

Every field here changes the numbers the model sees, so none of them gets a silent
default.

## What you produce

Three `HeteroData` objects per fold — one each for train, validation and test — either
returned in memory or saved to `database/graphs/<name>/fold-<i>/<role>.pt`.

Contents of each:

| Field | Shape | Meaning |
|---|---|---|
| `['raingauge'].x` | `[nodes, time, 6]` | channel 0 rainfall, channel 1 validity (1 = observed), channels 2–5 the Laplacian positional encoding |
| `['raingauge'].y` | `[nodes, time, 1]` | the true rainfall, kept separate from `x` |
| `['raingauge'].mask` | `[nodes]` bool | which nodes are the prediction targets for this role |
| `['raingauge','connects','raingauge'].edge_index` | `[2, edges]` | source and target node positions |
| `…edge_attr` | `[edges]` | inverse-distance weight |
| `station_ids` | list of `n` ids | node position → `station_id`. **Required**, and not currently produced — see Rules |
| `timestamps` | list of `T` UTC timestamps | node position → hour |

Which stations appear in which role's graph:

- **train graph** — train stations only
- **validation graph** — train + validation stations, `mask` marks the validation ones
- **test graph** — all stations, `mask` marks the test ones

The graph gets denser at evaluation time on purpose: at test time the model is allowed to
see every gauge except the ones it is predicting.

The **positional encoding (LPE)** gives each node four numbers describing where it sits
in the network, so the model can tell two gauges apart when their rainfall looks alike.

Also produce `report.md`: node and edge counts per fold and role, and the edge-weight
range. It is how the Component 06 owner sees at a glance that a graph is sane.

## How your output is used

Component 06 wraps each object in `HeterogeneousWeatherGraphDatasetInductive`, which
yields **one timestep per item** — `x` becomes `[nodes, 6]` for that hour — and batches
128 hours of the same graph together. Then, in `training/logic_hetero.py:137`, it zeroes
channels 0–1 of the masked nodes and computes the loss only on those nodes.

So the target hiding happens in the training loop, not here. Your job is to make sure
`mask` is right and that `y` is a genuine copy, not a view into `x`.

## Rules

1. The split decides roles. Never re-derive station roles from the data.
2. `x` and `y` must not share storage. Zeroing `x` must not change `y`.
3. No test station may appear in a train graph.
4. Node ordering is `mapping_df["order"]`, and `station_ids` must be carried on every
   saved object. The inherited code identifies nodes by integer position only, which is
   how a prediction ends up attributed to the wrong gauge after batching. Fixing this is
   part of your work.
5. Gauge-only runs create no placeholder radar or satellite nodes. An absent modality is
   absent, not zero-filled.
6. Same inputs and config produce the same tensors.

## Done when

- [ ] Graphs build from `database/standard/`, with no path into
      `database/sydney_gauge_legacy/`.
- [ ] Test: every node's `station_id` survives a batch and unbatch.
- [ ] Test: changing a held-out station's rainfall in Component 02 data changes nothing
      in the training graph or its normalisation statistics.
- [ ] Test: `y` is unchanged after `x` is zeroed.
- [ ] Test: with `per_node` normalisation, every node's largest incoming edge weight is
      exactly 1.
- [ ] Test: no self-edges, and no node has more than `k_neighbours` outgoing edges.
- [ ] Regression test: with `distance: euclidean_degrees` and
      `edge_normalisation: global`, node counts, edge counts and weights match the
      current `train_fused.py` run. This is one test pinning the old behaviour, not a
      second supported mode.
- [ ] Radar attaches as a second node layer and the model trains with it.
- [ ] `docs/graphs/README.md` shows the shapes above with a real printed example.

## Decisions to make explicitly

These are known defects in the inherited code. Each is cheap to fix, but a silent fix
makes old and new results incomparable — so fix it, name it in `report.md`, and say what
changed.

- **Distance in degrees.** KNN currently runs on raw longitude/latitude. At Sydney's
  latitude one degree of longitude is about 93 km against 111 km for latitude, so
  neighbours are chosen slightly wrong. `haversine_km` fixes it.
- **Global edge normalisation** divides every weight by the single largest weight in the
  graph, so one unusually close pair of gauges flattens every other edge. `per_node`
  matches the paper's Equation 1.
- **Validity gets z-scored.** `train_fused.py:120` normalises all channels including the
  0/1 validity flag, so "observed" stops being 1. Exclude channel 1 from normalisation.
- **Masking happens after normalisation** (`logic_hetero.py:137`), so a hidden gauge is
  set to the training *mean* rather than to no rain. Either mask before normalising or
  substitute the normalised value of physical zero.
