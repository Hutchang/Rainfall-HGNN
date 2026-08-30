"""T-002 — Split integrity and graph-construction invariants.

Verifies that every target belongs to its intended split, that no target value
is reachable by a permitted predictor, and that the edge-weight scheme actually
lets messages flow.

Findings encoded here are documented in
docs/baseline-reconciliation/AUDIT_NIRAMAY.md sections 8.0 and 8.1.

Tests marked xfail record KNOWN, UNFIXED defects (non-strict, so they begin
passing once fixed). These are the executable form of the two blockers.
"""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

KNN = 5          # config default, train.py --knn
SPLIT_SEED = 123 # config.yaml data.split_seed
N_SPLITS = 5


@pytest.fixture(scope="module")
def splits(station_meta):
    """Niramay's own sampler, imported from the worktree if importable;
    otherwise reimplemented is NOT attempted — we skip rather than diverge."""
    import sys
    from pathlib import Path
    from tests.conftest import DATA_ROOT
    repo = Path(DATA_ROOT).parent.parent          # .../COMP3888-niramay
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from data.sampling import stratified_spatial_kfold
    except Exception as e:                        # pragma: no cover
        pytest.skip(f"cannot import niramay's sampler: {e}")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return stratified_spatial_kfold(station_meta, n_splits=N_SPLITS,
                                        seed=SPLIT_SEED)


# ── Split partitioning ───────────────────────────────────────────────────────

def test_splits_are_produced(splits):
    assert len(splits) == N_SPLITS
    for f in splits:
        for key in ("train", "validation", "test"):
            assert len(f["ml"][key]) > 0


@pytest.mark.xfail(
    reason="AUDIT_NIRAMAY.md section 8.1 — UNFIXED, and broader than the "
           "train/test overlap alone. Because _make_masks maps a duplicated id "
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
        tr = set(f["ml"]["train"]); va = set(f["ml"]["validation"]); te = set(f["ml"]["test"])
        if tr & va or tr & te or va & te:
            offenders.append((f["fold_index"],
                              sorted(tr & te), sorted(tr & va), sorted(va & te)))
    assert not offenders, f"stations holding multiple roles: {offenders}"


@pytest.mark.xfail(
    reason="AUDIT_NIRAMAY.md section 8.1 — UNFIXED. Duplicated metadata rows put "
           "the same station in train and test. 4 of 5 folds are affected; the "
           "sampler prints its own fold-overlap warning at runtime.",
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
    reason="AUDIT_NIRAMAY.md section 8.1 — UNFIXED. stratified_spatial_kfold's "
           "docstring guarantees disjoint test sets across folds; duplicated "
           "ids break it (id 567102 appears in folds 0 and 2).",
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
        f"(was 39 of 188 at audit time)")


# ── Edge weights: can messages actually flow? ────────────────────────────────

def _edge_weights(meta_subset):
    """Replicates GaugeGraph._build_knn_edges (graph/gauge_graph.py:121-135)."""
    coords = meta_subset[["longitude", "latitude"]].values
    nbrs = NearestNeighbors(n_neighbors=KNN + 1, algorithm="ball_tree").fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    return np.array([1.0 / d if d > 0 else 1e6
                     for row in dists for d in row[1:]])


@pytest.mark.xfail(
    reason="AUDIT_NIRAMAY.md section 8.0 — UNFIXED, and the root cause of the "
           "baseline not learning. Stations 212285 and 563086 sit 0.1 m apart, "
           "giving 1/d = 2,219,128 against a median of 13.7. Because "
           "graph/gauge_graph.py:172 divides by the GLOBAL max, 99.8% of edges "
           "fall below 1e-3 and GraphConv's neighbour term sits ~5 orders of "
           "magnitude under its self term. The paper's per-node normalisation "
           "(arXiv:2607.01621 Eq. 1) is immune. Jylow shares the same formula "
           "at src/graph/gaugegraphnew.py:172-173.",
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
        f"neighbour messages are effectively zero")


def test_per_node_normalisation_would_fix_it(station_meta):
    """Positive control: the paper's Eq. 1 keeps every node's strongest
    incoming edge at 1.0 regardless of pathological pairs elsewhere.
    This should pass now and is the target behaviour for the fix."""
    meta = station_meta.drop_duplicates(subset=["id"]).reset_index(drop=True)
    coords = meta[["longitude", "latitude"]].values
    nbrs = NearestNeighbors(n_neighbors=KNN + 1, algorithm="ball_tree").fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    per_node_medians = []
    for row in dists:
        w = np.array([1.0 / d if d > 0 else 1e6 for d in row[1:]])
        per_node_medians.append(np.median(w / w.max()))   # normalise per node
    median = float(np.median(per_node_medians))
    assert median > 1e-3, f"per-node normalisation still collapses: {median:.2e}"


# ── Real GaugeGraph, edge_normalisation="global" vs "per_node" ──────────────
# The two tests above are standalone reproductions of the maths; they never
# call GaugeGraph itself, so they can't detect whether the real fix (added to
# graph/gauge_graph.py, arXiv:2607.01621 sec 3.2 Eq. 1) is actually wired up.
# These exercise the real class.

@pytest.fixture(scope="module")
def gauge_graph_cls():
    """Niramay's GaugeGraph, imported from the worktree if importable;
    otherwise skip rather than diverge (same policy as `splits`)."""
    import sys
    from pathlib import Path
    from tests.conftest import DATA_ROOT
    repo = Path(DATA_ROOT).parent.parent          # .../COMP3888-niramay
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from graph.gauge_graph import GaugeGraph
    except Exception as e:                        # pragma: no cover
        pytest.skip(f"cannot import niramay's GaugeGraph: {e}")
    return GaugeGraph


def test_global_normalisation_default_matches_baseline_defect(
    gauge_graph_cls, gauge_wide, station_meta, splits
):
    """Regression guard: edge_normalisation defaults to 'global' and must
    still reproduce the exact collapse in AUDIT_NIRAMAY.md sec 8.0 — the
    unmodified baseline must stay unmodified by the fix below."""
    graph = gauge_graph_cls(gauge_wide, station_meta, split_info=splits[0], knn=KNN)
    ea = graph.get_train_heterodata()["raingauge", "connects", "raingauge"].edge_attr.numpy()
    median = float(np.median(ea))
    assert median < 1e-3, f"expected the known global-max collapse, got median={median:.2e}"


def test_per_node_normalisation_fixes_the_real_graph(
    gauge_graph_cls, gauge_wide, station_meta, splits
):
    """edge_normalisation='per_node' (arXiv:2607.01621 sec 3.2 Eq. 1),
    exercised through the real GaugeGraph class rather than a standalone
    reproduction of the formula."""
    graph = gauge_graph_cls(
        gauge_wide, station_meta, split_info=splits[0], knn=KNN,
        edge_normalisation="per_node",
    )
    ea = graph.get_train_heterodata()["raingauge", "connects", "raingauge"].edge_attr.numpy()
    median = float(np.median(ea))
    assert median > 1e-3, f"per-node normalisation still collapses: median={median:.2e}"
