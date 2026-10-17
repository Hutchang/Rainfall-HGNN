# Design discussion — spatial CV leakage and regional partitioning

**Exported:** 2026-09-12
**Branch:** `feature/04-graph-construction`
**Participants:** Rick (ricklfo) · Claude Opus 5

A two-part design conversation. Part 1 diagnoses spatial leakage in the current
k-fold split. Part 2 evaluates a proposal to partition Australia into regions with
one GNN per region. Nothing in here has been implemented — it is a record of the
reasoning and the recommendations that came out of it.

---

## Part 1 — Spatial leakage in the k-fold split

### Rick

> I just had a great idea for fixing the k fold, the stratified spatial k fold algorithm. Currently, we hold out exactly one station. The test fold thus is a singular point. But what happens when two points or two stations are right next to each other? While the model trains on one of them and then it's forced... and then perhaps it predicts the other. As a result, while the while the holdout is meant to reduce overfitting, the model can literally just predict on that gauge that it is right next to the held out one, effectively overfitting the model. I was thinking since we used k means clustering, we could form some sort of rule where if a candidate holdout point or some type of candidate system where inside each cluster, you have to be... a holdout point has to be a certain distance from the centroid. or something along those lines. Just some rules based system, um, that the clustering could invite based off of the geographical and genuinely clustered nature of of data, of the data points.

### Claude

I dug through both the splitter and the graph builder. Your instinct is right, and the situation is worse than you framed it — the current splitter doesn't just permit that leakage, it maximises it by construction.

#### Where the leakage actually lives

Two things to separate first. The active splitter is [src/study/split.py](../src/study/split.py), not [src/sampling/main.py](../src/sampling/main.py) (that's the legacy one-station-per-cluster version). The active one isn't leave-one-out at all — [create_test_folds](../src/study/split.py#L162-L185) round-robins stations *within* each cluster across the 5 folds:

```python
for cluster in sorted(cluster_assignments):
    station_ids = sorted(cluster_assignments[cluster]); rng.shuffle(station_ids)
    for station_id in station_ids:
        test_folds[next_fold].append(station_id)
        next_fold = (next_fold + 1) % n_folds
```

So each cluster is deliberately sprayed evenly across all folds. For any test station, ~80% of its own cluster-mates — i.e. its geographically nearest gauges — are sitting in training. That's the exact pathology you described, applied to every test station rather than occasionally.

And the graph path makes it bite. The test graph is built over **all** stations ([gaugegraphnew.py:122](../src/graph/gaugegraphnew.py#L122), `mask = np.ones(...)`), k=5 kNN. At eval the test node's own rainfall is zeroed ([error_analysis.py:927](../src/visualization/error_analysis.py#L927)) — good, that part is correct and inductive — so the prediction is a learned function of exactly the 5 nearest gauges' live observations. If one of those sits 300 m away, the network only has to learn "copy neighbour 1."

Worth being precise about the failure mode: this isn't overfitting. The model generalises fine to the task it's being scored on. The problem is that the task is "interpolate across 300 m," and you're reporting it as skill at rainfall field reconstruction.

#### Centroid distance is the wrong knob

The instinct to use the clustering is right, but distance-to-centroid doesn't measure the thing that causes the leak. What matters is **distance to the nearest training station**, and those two come apart badly: a station way out at the cluster edge can still have a twin right beside it, while a station sitting near the centroid of a sparse cluster can be genuinely isolated. The centroid is also an artifact of an arbitrary k=10 — it has no physical relationship to rainfall decorrelation distance.

There's a second problem with any rule of the form "only test stations that are isolated": it makes the test set unrepresentative. Isolated gauges in Sydney skew toward the urban fringe and higher terrain, so your metric would move for two reasons at once and you couldn't attribute the change.

#### The fix that fits your existing code

Invert the loop. Assign **whole clusters to folds** instead of stations to folds — with 10 clusters and 5 folds, two clusters per fold. This is standard spatial block CV, and it's close to a drop-in:

- Every station still appears in test exactly once, so `validate_fold_roles` and its `train | validation | test == eligible_ids` invariant ([split.py:208](../src/study/split.py#L208)) hold unchanged.
- Held-out stations are contiguous, so a test station's neighbours are held out *with* it. You get the buffer effect for free, without a radius parameter.
- Downstream (`sydney_5fold.json`, D06) sees the same schema.

The one wrinkle is that k-means gives unbalanced clusters, so greedy bin-pack them (sort by size descending, assign each to the currently-smallest fold) rather than pairing them arbitrarily.

The alternative from the literature is a **buffered holdout**: keep the current folds but delete every training station within radius *r* of a test station (Brenning 2012; Roberts et al. 2017; Ploton et al. 2020 on spatial autocorrelation inflating CV). It's more surgical, but it fights your k-fold design — with 20% of stations in test, a 5 km buffer around each one removes most of the map from training. That's why the literature pairs buffering with leave-*one*-out and pairs k-fold with blocking. Given you're on a deadline, blocking is the one that fits what you already have.

#### Measure before you build

The cheap diagnostic that decides whether this is worth your time: the nearest-neighbour distance distribution across the Sydney gauges. If the 10th percentile is 8 km, this is theoretical. If there's a knot of gauges under 1 km, it's inflating your headline number right now.

I couldn't run it — `database/standard/stations.csv` isn't in the repo, only a 3-row fixture at [tests/data/study/stations.csv](../tests/data/study/stations.csv). If you point me at the real file I'll compute it.

The stronger version, if you have the compute: evaluate at r = 0, 1, 2, 5 km buffers and plot test RMSE against r. That turns "our RMSE is X" into "our RMSE is X at zero separation and Y at 5 km," which is a much more defensible claim and is a methods result in its own right.

---

## Part 2 — Clustering for regional partitioning

### Rick

> I'm having even more thoughts about clustering. Geography's shit. Characteristics. Clustering is a great way to group data points with share characteristics. And if we're measuring rainfall in sparse geographies of Australia, then it could be a great idea to utilize clustering more carefully and potentially an even extension idea where we use clustering perhaps for determining what partitions or what subgraphs we train. Um, for example, this chunk of Sydney gets... is one cluster, but this chunk of Sydney, which is central coast, gets its own cluster, and then those two clusters are the data for which their associated graphs get trained on. Graph neural networks get trained on. Potentially even hierarchical clustering. Oh, I haven't exactly thought about how I'd use hierarchical clustering. Does that make sense? because currently, one of the issues we're facing is that we have Australia wide data and one graph neural network across the entire entirety of Australia just doesn't make sense. You're forcing a neural network to learn multiple geographies or at least it feels like it doesn't make sense. And on top of that, you face data scalability issues where you're feeding it a ridiculous amount of data across all of Australia and inference slash training could take very long. So our solution was: have multiple graphs trained on regions with their own geographies (so a network doesnt attempt to learn sydney and perth at the same time). The issue with that though, was: "how do we determining the partition? Which data goes into what graph? At inference time, given a queried geographical point, how does the model know which subgraph to inference on? I'm now thinking clustering could be a solution to explore for these problems

### Claude

Yes, it makes sense, and the "characteristics, not geography" instinct is the best part of it. But I think the idea is right about *partitioning* and wrong about *what to partition* — and one of your two motivations doesn't survive contact with your actual numbers.

#### The compute premise doesn't hold here

Your graph is tiny and your time axis is enormous. From [config/canonical/](../config/canonical/): 152 Sydney stations, k=5 neighbours → ~750 edges. From [config/ingestion/sydney.yaml](../config/ingestion/sydney.yaml): 2021-01-01 to 2022-12-31 hourly → ~17,520 timesteps. And [HeterogeneousWeatherGraphDatasetInductive](../src/graph/gaugegraphnew.py#L426-L465) yields **one graph per timestep**:

```python
def __len__(self):  return self.num_timesteps
def __getitem__(self, idx):
    x = self.heterodata["raingauge"].x[:, idx, :]   # [N, F] at one t
```

So an epoch is 17,520 forward/backward passes over a 150-node graph. The cost is `T`, not `N`. Scale to all of Australia and you might reach a few thousand usable sub-daily gauges — still a trivial graph by GNN standards, people train these on millions of nodes — with the *same* T.

Now price your proposal against that. Splitting into R regions leaves total node-work per epoch unchanged (the regions sum to N) but multiplies optimiser steps by R, each step R× smaller. At N=150 a step is dominated by Python and kernel-launch overhead, not FLOPs. **R separate models would run meaningfully slower in wall-clock, not faster.**

The actual win is sitting right there and has nothing to do with clustering: `edge_index` is identical at every timestep ([build_graph](../src/graph/gaugegraphnew.py#L111) reads only lat/lon), so you can batch over time instead of looping one timestep at a time. That's potentially two to three orders of magnitude, and it's methodologically free.

Worth measuring before building anything: profile one epoch and confirm the split between per-step overhead and actual convolution.

#### The modelling premise is the interesting one — and it's contested

"Forcing a network to learn Sydney and Perth at once" isn't quite what's happening. With k=5 and L layers the receptive field is an L-hop neighbourhood — tens of gauges. The model never sees Australia at once. What's shared is the *function*: "given neighbours' rainfall, distances, and local geometry, estimate here." A large part of that is interpolation geometry, which transfers fine across regions.

The closest evidence we have points against separate models. Kratzert et al. (2019, HESS, *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets*) found a single LSTM trained across 500+ catchments beat per-catchment calibrated models — including in ungauged basins — and that regional specialisation came from feeding **static catchment attributes as inputs**, not from separate weights. That's a direct analogue of what you're proposing, and the pooled model won.

There's also a backwards incentive in the partitioning plan: each regional model gets 1/R of the data, and sparse inland Australia — the case that motivated this — is exactly where you have the *fewest* gauges. **Partitioning starves precisely the regions that most need to borrow strength from elsewhere.**

#### Where clustering genuinely earns its place

These are four different jobs. They want different clusterings, and collapsing them into one `n_clusters=10` is a mistake:

| Job | Clustering on | Contiguity | Status |
|---|---|---|---|
| CV blocking (Part 1) | space | required | do this |
| Minibatch construction | graph structure | required | if compute ever bites |
| Conditioning features | rainfall characteristics | **not** wanted | do this |
| Separate weights | space | required | only if 3 fails |

Row 2 is your idea, executed correctly. Cluster-GCN (Chiang et al., KDD 2019) partitions the graph with METIS and trains on subgraph minibatches — **one model, shared weights**, partitioning purely as a batching trick. GraphSAINT and PyG's `ClusterLoader`/`NeighborLoader` do the same. You get the scalability story without fragmenting the training set.

Row 3 is where "geography's shit, characteristics" pays off. Cluster gauges on rainfall *regime* — annual total, seasonality, wet-day frequency, intensity distribution, elevation, distance to coast, Köppen zone — and feed the cluster id or a learned embedding as a node feature. Two coastal NSW sites 400 km apart may behave more alike than a coastal and an inland site 80 km apart, and here non-contiguous clusters are a *feature*, not a bug. This gets you regional specialisation while keeping one model and all the data.

#### Hierarchical clustering — the use you haven't found yet

It resolves a contradiction between this idea and Part 1. If each region has its own GNN, **leave-one-cluster-out CV becomes impossible** — hold out a region and its model has no training data. You need nested blocking, and a dendrogram gives you exactly that: cut coarse for region identity (which model, or which embedding), cut fine *within* each region for CV folds.

Three other concrete advantages over the current k-means:

- **Picking k.** Merge heights show you the natural separation scale instead of guessing 8 or 10. Cut at a height tied to something physical — rainfall decorrelation length.
- **Determinism.** No seed sensitivity. [assign_spatial_clusters](../src/study/split.py#L126-L160) currently has to sort station IDs specifically because k-means++ samples by index; that whole class of fragility disappears.
- **Contiguity on demand.** `AgglomerativeClustering(connectivity=<kNN graph>)` guarantees spatially contiguous clusters, which k-means does not. That's what blocking and partitioning both need. Use Ward linkage — single linkage will chain along the coastline and hand you one 3,000 km "cluster."

#### If you do go regional, the seam problem is real

Nearest-centroid routing is trivial to implement, but it makes predictions discontinuous across boundaries. Since you're producing a gridded field ([grid_prediction.py](../src/visualization/grid_prediction.py)), those seams show up as visible artifacts in the output raster and anyone looking at the map will spot them. The fix is halo overlap — each region's graph includes stations r km past its boundary — plus distance-weighted blending near boundaries. Standard domain decomposition, but real complexity, and it eats the compute saving you were chasing. A single global model has no seams by construction.

#### What I'd actually do

1. **Batch over time.** Biggest win available, zero methodological risk, doesn't touch the split contract.
2. **Add static/climate features and a regime-cluster embedding to node features.** Cheap, likely moves your numbers, and directly tests whether regional specialisation is needed at all.
3. **Run the decisive experiment when you get a second region.** Global model scored on region A, versus region-A-only model scored on region A. That single comparison gates the entire architecture decision — and either outcome is a genuine result. [config/ingestion/](../config/ingestion/) is already one-file-per-region, so a second region is a config file, not a refactor. That's the cheapest way to make this question answerable.
4. **Separate weights only if 3 says so.**

One scope flag, given where your deadline is: partitioning-by-region touches components 03, 04, and 06 — every contract in the [requirements table](requirements/README.md#L32-L40) — and the committed pipeline is Sydney-only, 152 stations, 2021–2022. As a written-up extension with the reasoning above, this is strong. As something to build now, the compute motivation doesn't hold at your current N and the data to justify the modelling motivation doesn't exist yet.

---

## Open actions

- [ ] Compute nearest-neighbour distance distribution over the real `database/standard/stations.csv` (not in repo — needs to be pointed at).
- [ ] Profile one epoch: per-step overhead vs convolution FLOPs.
- [ ] Decide on connectivity-constrained agglomerative clustering to replace k-means in `assign_spatial_clusters`.
- [ ] Decide whether cluster-to-fold blocking replaces, or sits behind a flag alongside, the current round-robin in `create_test_folds`.

## References cited

- Brenning (2012) — spatial cross-validation with buffering.
- Roberts et al. (2017) — *Cross-validation strategies for data with spatial, temporal, hierarchical, or phylogenetic structure.*
- Ploton et al. (2020) — spatial autocorrelation inflating CV scores.
- Kratzert et al. (2019, HESS) — *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets.*
- Chiang et al. (KDD 2019) — *Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.*
- Zeng et al. (ICLR 2020) — *GraphSAINT: Graph Sampling Based Inductive Learning Method.*
