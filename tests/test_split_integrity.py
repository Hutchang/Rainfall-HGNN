"""T-002 — Split integrity and graph-construction invariants.

Verifies that every target belongs to its intended split, that no target value
is reachable by a permitted predictor, and that the edge-weight scheme actually
lets messages flow.

Findings encoded here are documented in
docs/archive/baseline-reconciliation/AUDIT_NIRAMAY.md sections 8.0 and 8.1.

Tests marked xfail record defects in the inherited Niramay data/path. New component
contracts fix them with internal station ids, audited split artifacts, and explicit
graph modes; these tests remain as legacy regression evidence.
"""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

KNN = 5  # inherited Niramay config/CLI default
SPLIT_SEED = 123  # inherited Niramay config.yaml split_seed
N_SPLITS = 5


@pytest.fixture(scope="module")
def splits(station_meta):
    """Niramay's own sampler, imported from the worktree if importable;
    otherwise reimplemented is NOT attempted — we skip rather than diverge."""
    import sys
    from pathlib import Path

    from tests.conftest import DATA_ROOT

    repo = Path(DATA_ROOT).parent.parent  # .../COMP3888-niramay
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from data.sampling import stratified_spatial_kfold
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot import niramay's sampler: {e}")
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return stratified_spatial_kfold(
            station_meta, n_splits=N_SPLITS, seed=SPLIT_SEED
        )


# ── Split partitioning ───────────────────────────────────────────────────────


def test_splits_are_produced(splits):
    assert len(splits) == N_SPLITS
    for f in splits:
        for key in ("train", "validation", "test"):
            assert len(f["ml"][key]) > 0


@pytest.mark.xfail(
    reason="LEGACY DATA REGRESSION — AUDIT_NIRAMAY.md section 8.1. The "
    "component 03 split contract retires this behaviour. Inherited "
    "_make_masks maps a duplicated public id "
    "to BOTH node orders, train/validation overlap occurs in ALL 5 folds "
    "(up to 8 stations in fold 4) and train/test in 4 of 5. Validation "
    "leakage corrupts early stopping and model selection, not just the "
    "reported test metric.",
    strict=False,
)
def test_every_target_belongs_to_exactly_one_role_per_fold(splits):
    """A station must not hold two roles in the same fold."""
    offenders = []
    for f in splits:
        tr = set(f["ml"]["train"])
        va = set(f["ml"]["validation"])
        te = set(f["ml"]["test"])
        if tr & va or tr & te or va & te:
            offenders.append(
                (f["fold_index"], sorted(tr & te),
                 sorted(tr & va), sorted(va & te))
            )
    assert not offenders, f"stations holding multiple roles: {offenders}"


@pytest.mark.xfail(
    reason="LEGACY DATA REGRESSION — AUDIT_NIRAMAY.md section 8.1. Duplicated metadata rows put "
    "the same station in train and test. 4 of 5 folds are affected; the "
    "sampler prints its own fold-overlap warning at runtime. Component 03 "
    "requires unique internal ids and an audited target index.",
    strict=False,
)
def test_no_station_is_both_trained_on_and_tested(splits):
    """The core spatial-CV premise: a held-out station was never trained on."""
    leaks = {}
    for f in splits:
        overlap = set(f["ml"]["test"]) & set(f["ml"]["train"])
        if overlap:
            leaks[f["fold_index"]] = sorted(overlap)
    assert not leaks, f"test stations also present in train: {leaks}"


@pytest.mark.xfail(
    reason="LEGACY DATA REGRESSION — AUDIT_NIRAMAY.md section 8.1. stratified_spatial_kfold's "
    "docstring guarantees disjoint test sets across folds; duplicated "
    "ids break it (id 567102 appears in folds 0 and 2). Component 03's "
    "corrected protocol requires full, disjoint test coverage.",
    strict=False,
)
def test_test_sets_are_disjoint_across_folds(splits):
    seen, dupes = set(), {}
    for f in splits:
        te = set(f["ml"]["test"])
        if seen & te:
            dupes[f["fold_index"]] = sorted(seen & te)
        seen |= te
    assert not dupes, f"test sets overlap across folds: {dupes}"


def test_test_coverage_is_recorded(splits, station_meta):
    """Only a fixed subset is ever evaluated (AUDIT_NIRAMAY.md section 8.3).
    Asserted as a documented property, not a defect to fix here."""
    tested = set().union(*(set(f["ml"]["test"]) for f in splits))
    assert len(tested) < len(station_meta), "expected partial test coverage"
    assert len(tested) == 39, (
        f"test coverage changed: {len(tested)} of {len(station_meta)} nodes "
        f"(was 39 of 188 at audit time)"
    )


# ── Edge weights: can messages actually flow? ────────────────────────────────


def _edge_weights(meta_subset):
    """Replicates GaugeGraph._build_knn_edges (graph/gauge_graph.py:121-135)."""
    coords = meta_subset[["longitude", "latitude"]].values
    nbrs = NearestNeighbors(n_neighbors=KNN + 1,
                            algorithm="ball_tree").fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    return np.array([1.0 / d if d > 0 else 1e6 for row in dists for d in row[1:]])


@pytest.mark.xfail(
    reason="LEGACY GLOBAL-MODE REGRESSION — AUDIT_NIRAMAY.md section 8.0. "
    "The opt-in per-node correction is implemented and has separate passing "
    "tests; global remains reproducible by design. In the inherited metadata, "
    "stations 212285 and 563086 sit 0.1 m apart, "
    "giving 1/d = 2,219,128 against a median of 13.7. Because "
    "graph/gauge_graph.py:172 divides by the GLOBAL max, 99.8% of edges "
    "fall below 1e-3 and GraphConv's neighbour term sits ~5 orders of "
    "magnitude under its self term. The paper's per-node normalisation "
    "(arXiv:2607.01621 Eq. 1) is immune.",
    strict=False,
)
def test_normalised_edge_weights_permit_message_passing(station_meta):
    """After normalisation a typical edge must retain non-negligible weight,
    otherwise the GNN degenerates into a per-node MLP."""
    w = _edge_weights(station_meta.drop_duplicates(subset=["id"]))
    normalised = w / w.max()
    median = float(np.median(normalised))
    assert median > 1e-3, (
        f"median normalised edge weight is {median:.2e}; "
        f"{(normalised < 1e-3).mean():.1%} of edges are below 1e-3, so "
        f"neighbour messages are effectively zero"
    )


def test_per_node_normalisation_would_fix_it(station_meta):
    """Positive control: the paper's Eq. 1 keeps every node's strongest
    incoming edge at 1.0 regardless of pathological pairs elsewhere.
    This should pass now and is the target behaviour for the fix."""
    meta = station_meta.drop_duplicates(subset=["id"]).reset_index(drop=True)
    coords = meta[["longitude", "latitude"]].values
    nbrs = NearestNeighbors(n_neighbors=KNN + 1,
                            algorithm="ball_tree").fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    per_node_medians = []
    for row in dists:
        w = np.array([1.0 / d if d > 0 else 1e6 for d in row[1:]])
        per_node_medians.append(np.median(w / w.max()))  # normalise per node
    median = float(np.median(per_node_medians))
    assert median > 1e-3, f"per-node normalisation still collapses: {median:.2e}"


# ── Tests that build a real graph ─────────────────────────────────────────
# The two tests above work out the edge-weight formula by hand. They never
# build a graph, so they cannot tell you whether the graph code itself does
# the right thing. The tests below build a real graph and check that.
#
# They use GaugeGraphNew, in src/graph/gaugegraphnew.py. That is the graph
# builder this project keeps and extends, and Component 04's requirements
# name it as the one to use. An older copy of the same class also exists in
# the inherited niramay repository. Commit 65237ba fixed both copies, but only
# the fix in this repository was committed, and we keep the niramay repository
# exactly as we inherited it. So these tests use our copy.


@pytest.fixture(scope="module")
def gauge_graph_cls():
    """The graph builder these tests run against.

    Takes the same arguments as the inherited niramay version, plus
    edge_normalisation, which chooses how the edge weights get rescaled.
    """
    from src.graph.gaugegraphnew import GaugeGraphNew

    return GaugeGraphNew


def test_per_node_normalisation_fixes_the_real_graph(
    gauge_graph_cls, gauge_wide, station_meta, splits
):
    """Checks that asking for the corrected rescaling actually gives you it.

    With edge_normalisation="per_node" the weights should no longer be
    squashed to almost zero, so a typical weight should sit well above it.
    """
    graph = gauge_graph_cls(
        gauge_wide,
        station_meta,
        split_info=splits[0],
        knn=KNN,
        edge_normalisation="per_node",
    )
    ea = graph.get_train_heterodata()[
        "raingauge", "connects", "raingauge"
    ].edge_attr.numpy()
    median = float(np.median(ea))
    assert median > 1e-3, f"per-node normalisation still collapses: median={median:.2e}"


def test_per_node_normalisation_scales_every_node_by_its_own_neighbourhood(
    gauge_graph_cls, gauge_wide, station_meta, splits
):
    """Checks each station's edge weights are rescaled using its own
    neighbours, and not somebody else's.

    Every station is joined to its 5 nearest neighbours, and each connection
    gets a weight of 1 divided by the distance. Those raw weights vary
    enormously, so they are rescaled to sit between 0 and 1 before the model
    sees them.

    The paper rescales each station based on the local max, and since this
    is on the denominator, every station should end up
    with its closest neighbour weighted exactly 1.0. This test checks that,
    one station at a time. If a station's closest neighbour comes out below
    1.0, its weights were rescaled using some other station's neighbours.

    Two things in src/graph/gaugegraphnew.py currently break it:

    1. Three pairs of stations in the inherited data sit at identical
       coordinates, so the distance between them is zero. The code divides by
       distance without checking for zero, which gives infinity, which turns
       into "not a number" once rescaled. The inherited version avoids this
       by using a million whenever the distance is zero
       (graph/gauge_graph.py:133).

    2. Connections are stored in an undirected graph, which holds only one
       weight per pair of stations. When A and B are each other's neighbours,
       the second weight written overwrites the first, and the survivor is
       then copied to both directions. So one of the two stations is left
       with a weight rescaled using the other station's neighbours. Storing
       the graph as directed, with its own weight in each direction, fixes
       this.
    """
    graph = gauge_graph_cls(
        gauge_wide,
        station_meta,
        split_info=splits[0],
        knn=KNN,
        edge_normalisation="per_node",
    )
    store = graph.get_train_heterodata()["raingauge", "connects", "raingauge"]
    edge_index = store.edge_index.numpy()
    edge_attr = store.edge_attr.numpy()

    assert not np.isnan(edge_attr).any(), (
        f"{int(np.isnan(edge_attr).sum())} of {edge_attr.size} edge weights are "
        "'not a number'. Stations at identical coordinates are zero distance "
        "apart, and the code divides by distance without checking for zero. "
        "See reason 1 in this test's description."
    )

    max_incoming: dict[int, float] = {}
    for destination, weight in zip(edge_index[1], edge_attr, strict=True):
        node = int(destination)
        max_incoming[node] = max(max_incoming.get(node, 0.0), float(weight))

    maxima = np.array(list(max_incoming.values()))
    off = ~np.isclose(maxima, 1.0, atol=1e-6)
    assert not off.any(), (
        f"{int(off.sum())} of {len(maxima)} stations do not have their closest "
        f"neighbour weighted 1.0 (lowest is {maxima.min():.3f}). Their closest "
        "neighbour counts for less than it should, so the weights were not "
        "rescaled per station. See reason 2 in this test's description."
    )
