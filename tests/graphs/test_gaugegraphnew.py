import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch
from src.graph.gaugegraphnew import GaugeGraphNew

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