import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import is_undirected
from src.graph.gaugegraphnew import GaugeGraphNew
from src.graph.gaugegraphnew import HeterogeneousWeatherGraphDatasetInductive
from src.graph.radargraphnew import RadarGraphNew

@pytest.fixture
def sample_graph_builder():
    """
    Fixture to load the committed test data and return a configured GaugeGraphNew instance.
    """
    # 1. Load the hourly data first and force IDs to strings
    long_gauge_df = pd.read_csv("tests/data/graphs/gauge_hourly.csv", dtype={"station_id": str})
    data_df = long_gauge_df.pivot(index="timestamp", columns="station_id", values="rainfall_mm")

    # 2. Load the mapping data and force IDs to strings
    mapping_df = pd.read_csv("tests/data/graphs/stations.csv", dtype={"station_id": str})
    mapping_df = mapping_df.rename(columns={"station_id": "id"})
    
    # 3. FIX: Filter the stations to ONLY those present in our mock hourly data
    mapping_df = mapping_df[mapping_df["id"].isin(data_df.columns)].reset_index(drop=True)
    
    if "use_in_study" in mapping_df.columns:
        mapping_df = mapping_df[mapping_df["use_in_study"] == True].reset_index(drop=True)
    mapping_df["order"] = range(len(mapping_df))
    
    # Mock a simple split for testing purposes
    all_ids = mapping_df["id"].tolist()
    split_info = {
        "ml": {
            "train": all_ids[:2], # First 2 for train
            "validation": all_ids[2:3], # 3rd for val
            "test": all_ids[3:] # 4th for test
        }
    }
    
    return GaugeGraphNew(
        data_df=data_df,
        mapping_df=mapping_df,
        split_info=split_info,
        knn=1,
        distance="haversine_km",
        edge_normalisation="per_node",
        include_lpe=False,
        lpe_dimensions=8,
        layer_is_directed=True
    )

def test_y_unchanged_after_x_zeroed(sample_graph_builder):
    """Test 1: y is unchanged after x is zeroed (Rule 2 - catches view vs copy)."""
    train_data = sample_graph_builder.get_train_heterodata()
    
    # Capture original y
    original_y = train_data["raingauge"].y.clone()
    
    # Zero out x
    train_data["raingauge"].x.zero_()
    
    # Assert y is unchanged (meaning it was properly cloned, not a view)
    assert torch.equal(train_data["raingauge"].y, original_y), "y tensor was mutated when x was changed. Missing .clone()!"

def test_no_self_edges_and_max_knn(sample_graph_builder):
    """Test 2: No self-edges, no node with more than k_neighbours outgoing edges."""
    train_data = sample_graph_builder.get_train_heterodata()
    edge_index = train_data["raingauge", "connects", "raingauge"].edge_index
    
    # Check for self edges (source == destination)
    self_edges = (edge_index[0] == edge_index[1]).sum().item()
    assert self_edges == 0, f"Found {self_edges} self-edges in the graph."
    
    # Check max outgoing edges per node
    # bincount counts the occurrences of each source node index
    outgoing_counts = torch.bincount(edge_index[0])
    max_outgoing = outgoing_counts.max().item()
    assert max_outgoing <= sample_graph_builder.knn, f"Node has {max_outgoing} edges, exceeding k_neighbours ({sample_graph_builder.knn})"

def test_per_node_normalisation(sample_graph_builder):
    """Test 3: with per_node, every node's largest incoming edge weight is exactly 1."""
    train_data = sample_graph_builder.get_train_heterodata()
    edge_index = train_data["raingauge", "connects", "raingauge"].edge_index
    edge_attr = train_data["raingauge", "connects", "raingauge"].edge_attr
    
    # Group weights by destination node (index 1)
    dest_nodes = edge_index[1]
    num_nodes = train_data["raingauge"].x.shape[0]
    
    max_weights = torch.zeros(num_nodes)
    for i in range(len(dest_nodes)):
        node = dest_nodes[i].item()
        weight = edge_attr[i].item()
        if weight > max_weights[node]:
            max_weights[node] = weight
            
    # Check that for nodes that have incoming edges, their max weight is 1.0
    for i, max_w in enumerate(max_weights):
        if max_w > 0: # Only test nodes that actually received edges
            assert np.isclose(max_w.item(), 1.0, atol=1e-5), f"Node {i} max incoming weight is {max_w}, expected 1.0"

def test_no_test_station_in_train_graph(sample_graph_builder):
    """Test 4: No test station appears in a train graph (Rule 3)."""
    train_data = sample_graph_builder.get_train_heterodata()
    test_gauges = set(sample_graph_builder.test_gauges)
    
    train_station_ids = set(train_data.station_ids)
    
    overlap = train_station_ids.intersection(test_gauges)
    assert len(overlap) == 0, f"Data leakage: Test stations found in train graph: {overlap}"

def test_station_id_survives_batching(sample_graph_builder):
    """Test 5: station_id survives a batch and unbatch (Rule 4)."""
    train_data = sample_graph_builder.get_train_heterodata()
    
    # Create two dummy graphs to batch
    data1 = train_data.clone()
    data2 = train_data.clone()
    
    # Batch them
    batched = Batch.from_data_list([data1, data2])
    
    # Ensure station_ids attribute exists on the batched object
    assert hasattr(batched, "station_ids"), "station_ids attribute lost during batching"
    
    # Unbatch them
    unbatched_list = batched.to_data_list()
    
    # Verify the first unbatched graph retained its original station_ids
    assert hasattr(unbatched_list[0], "station_ids"), "station_ids lost during unbatching"
    assert unbatched_list[0].station_ids == train_data.station_ids, "station_ids corrupted or misaligned after batch/unbatch"



def test_nan_validity_mask_handling(sample_graph_builder):
    """Test 6: Verifies that missing data generates a 0 in the validity mask and 0.0 in the value tensor."""
    train_data = sample_graph_builder.get_train_heterodata()
    # Check the feature dimension (Channel 0 is value, Channel 1 is mask)
    x = train_data["raingauge"].x
    
    # If a value is exactly 0.0, check if it was actually missing data (mask == 0)
    # The fixture data contains NaNs, so we should find at least one 0 in the mask
    assert torch.any(x[:, :, 1] == 0), "Validity mask did not register any missing (NaN) values."
    
    # Ensure that wherever the mask is 0, the value was safely filled with 0.0
    missing_indices = (x[:, :, 1] == 0)
    assert torch.all(x[missing_indices][:, 0] == 0.0), "NaN values were not replaced with 0.0 in the value channel."

def test_global_edge_normalisation(sample_graph_builder):
    """Test 7: Verifies 'global' normalisation scales the entire graph's max edge to 1.0."""
    # Temporarily change the normalisation method
    sample_graph_builder.edge_normalisation = "global"
    
    # Rebuild the train heterodata
    heterodata = sample_graph_builder.fill_heterodata("train")
    edge_attr = heterodata["raingauge", "connects", "raingauge"].edge_attr
    
    # The absolute maximum weight in the entire graph should be 1.0
    assert np.isclose(torch.max(edge_attr).item(), 1.0, atol=1e-5), "Global normalisation failed to set max edge weight to 1.0"

def test_lpe_feature_concatenation(sample_graph_builder):
    """Test 8: Verifies Laplacian Positional Encodings (LPE) correctly expand node features."""
    sample_graph_builder.include_lpe = True
    sample_graph_builder.lpe_dimensions = 1
    
    heterodata = sample_graph_builder.fill_heterodata("train")
    x = heterodata["raingauge"].x
    
    # Base features (2) + LPE dimensions (4) = 6 features
    assert x.shape[2] == 3, f"Expected 3 features after LPE, but got {x.shape[2]}"

def test_dataset_iterator_shapes(sample_graph_builder):
    """Test 9: Verifies the Inductive Dataset correctly slices time-series into single steps."""
    train_data = sample_graph_builder.get_train_heterodata()
    dataset = HeterogeneousWeatherGraphDatasetInductive(train_data)
    
    # Dataset length should match the number of timesteps (T)
    T = train_data["raingauge"].x.shape[1]
    assert len(dataset) == T, f"Dataset length {len(dataset)} does not match timesteps {T}"
    
    # Fetch the first timestep
    step_data = dataset[0]
    
    # Shape should drop the T dimension: [N, F] instead of [N, T, F]
    N = train_data["raingauge"].x.shape[0]
    F = train_data["raingauge"].x.shape[2]
    assert step_data["raingauge"].x.shape == (N, F), "Dataset iterator returned incorrect feature shape."

def test_radar_fusion_addition(sample_graph_builder):
    """Test 10: Verifies add_heterodata successfully injects the radar modality."""
    T = sample_graph_builder.train_heterodata["raingauge"].x.shape[1]
    
    # FIX: Provide a mock Pandas DataFrame instead of a tensor
    mock_radar_df = pd.DataFrame(
        np.zeros((T, 3)), 
        columns=[0, 1, 2] # 3 radar nodes
    )
    mock_coords = pd.DataFrame({"node_id": [0,1,2], "latitude": [-33.8, -33.9, -34.0], "longitude": [151.1, 151.2, 151.3]})
    
    radar_builder = RadarGraphNew(
        data_df=mock_radar_df, # Pass the DataFrame here
        coords_df=mock_coords,
        knn=2,
        distance_metric="haversine_km"
    )
    radar_builder.heterodata.timestamps = (
        sample_graph_builder.train_heterodata.timestamps
    )
    
    # Fuse them
    train_fused, _, _ = sample_graph_builder.add_heterodata(
        heterodata_layer=radar_builder.heterodata,
        coords=mock_coords,
        layer_name="radar",
        knn=2
    )
    
    assert "radar" in train_fused.node_types, "Radar node type missing after fusion."
    assert ("radar", "connects", "raingauge") in train_fused.edge_types, "Bipartite fusion edges missing."


@pytest.fixture
def gauge_frames():
    """The committed sample as (wide hourly table, station mapping, station ids)."""
    long_gauge_df = pd.read_csv(
        "tests/data/graphs/gauge_hourly.csv", dtype={"station_id": str}
    )
    data_df = long_gauge_df.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    mapping_df = pd.read_csv(
        "tests/data/graphs/stations.csv", dtype={"station_id": str}
    ).rename(columns={"station_id": "id"})
    mapping_df = mapping_df[mapping_df["id"].isin(data_df.columns)]
    if "use_in_study" in mapping_df.columns:
        mapping_df = mapping_df.loc[mapping_df["use_in_study"]]
    mapping_df = mapping_df.reset_index(drop=True)
    mapping_df["order"] = range(len(mapping_df))
    return data_df, mapping_df, mapping_df["id"].tolist()


def build_gauge_graph(frames, mapping_df=None, **overrides):
    """A one-neighbour builder over the sample, with DEC-4 on unless overridden."""
    data_df, default_mapping, ids = frames
    settings = dict(
        knn=1,
        distance="haversine_km",
        edge_normalisation="per_node",
        both_directions=True,
        include_lpe=False,
        lpe_dimensions=1,
    )
    settings.update(overrides)
    return GaugeGraphNew(
        data_df=data_df,
        mapping_df=default_mapping if mapping_df is None else mapping_df,
        split_info={
            "ml": {"train": ids[:2], "validation": ids[2:3], "test": ids[3:]}
        },
        **settings,
    )


def gauge_edges(builder):
    return builder.get_test_heterodata()["raingauge", "connects", "raingauge"]


def test_one_direction_per_pick_leaves_a_gauge_receiving_nothing(gauge_frames):
    """Test 11: the ISS-25 case DEC-4 exists to fix. An edge i -> j means j
    receives i's value, so with one edge per pick a gauge that is nobody's
    nearest receives no gauge value at all, and a held-out gauge there would be
    predicted from nothing."""
    edges = gauge_edges(build_gauge_graph(gauge_frames, both_directions=False))
    receiving = set(edges.edge_index[1].tolist())
    every_gauge = set(range(len(gauge_frames[2])))
    assert receiving != every_gauge, (
        "the committed sample is expected to contain a gauge that no other gauge "
        "picks; if the sample changes this test needs a different fixture"
    )


def test_both_directions_leaves_no_gauge_without_an_incoming_edge(gauge_frames):
    """Test 12: DEC-4 - every gauge receives from at least one other gauge."""
    edges = gauge_edges(build_gauge_graph(gauge_frames))
    receiving = set(edges.edge_index[1].tolist())
    every_gauge = set(range(len(gauge_frames[2])))
    assert receiving == every_gauge, (
        f"gauges receiving nothing: {sorted(every_gauge - receiving)}"
    )


def test_both_directions_makes_the_edge_set_symmetric(gauge_frames):
    """Test 13: DEC-4 - for every pick i -> j the reverse j -> i is present, which
    is what lets the positional encoding use the symmetric eigenvalue solver."""
    edges = gauge_edges(build_gauge_graph(gauge_frames))
    pairs = set(zip(*edges.edge_index.tolist(), strict=True))
    missing = {(i, j) for i, j in pairs if (j, i) not in pairs}
    assert not missing, f"edges with no reverse: {sorted(missing)}"


def test_both_directions_keeps_receiving_node_normalisation(gauge_frames):
    """Test 14: DEC-4 - the two directions of one pair start from the same raw
    weight but are rescaled at their own destination, so every gauge that
    receives anything still has a strongest incoming edge of exactly 1."""
    edges = gauge_edges(build_gauge_graph(gauge_frames))
    destinations, weights = edges.edge_index[1], edges.edge_attr
    for node in destinations.unique().tolist():
        strongest = weights[destinations == node].max().item()
        assert np.isclose(strongest, 1.0, atol=1e-6), (
            f"gauge {node} strongest incoming weight is {strongest}, expected 1.0"
        )


def test_both_directions_lets_the_symmetric_eigensolver_be_used(gauge_frames):
    """Test 15: DEC-4 - `get_laplacian` uses the edge list as given, so the
    positional encoding's matrix is symmetric only when both directions are
    present. That is what makes the symmetric solver the correct one; with one
    direction per pick the general solver is used and its eigenvectors can be
    complex."""
    both = gauge_edges(build_gauge_graph(gauge_frames)).edge_index
    one_way = gauge_edges(
        build_gauge_graph(gauge_frames, both_directions=False)
    ).edge_index
    assert is_undirected(both, num_nodes=4)
    assert not is_undirected(one_way, num_nodes=4)
