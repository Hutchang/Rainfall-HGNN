"""D09.2: direction, neighbour count and weights at the 04→06 boundary."""

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from src.graph.gaugegraphnew import GaugeGraphNew


@pytest.fixture
def attachment():
    ids = ["a", "b", "c", "d"]
    hours = pd.date_range("2022-01-01", periods=2, freq="h", tz="UTC")
    mapping = pd.DataFrame(
        {
            "id": ids,
            "latitude": [-33.0] * 4,
            "longitude": [151.0, 151.001, 151.002, 151.003],
            "order": range(4),
        }
    )
    builder = GaugeGraphNew(
        pd.DataFrame(1.0, index=hours, columns=ids),
        mapping,
        {"ml": {"train": ids[:2], "validation": ids[2:3], "test": ids[3:]}},
        knn=1,
        distance="haversine_km",
        edge_normalisation="per_node",
        both_directions=True,
        include_lpe=False,
    )
    coords = pd.DataFrame(
        {
            "latitude": [-33.0] * 12,
            "longitude": [151.0 + 0.01 * i for i in range(12)],
        }
    )
    sensors = HeteroData()
    sensors.timestamps = builder.train_heterodata.timestamps
    sensors["radar"].x = torch.ones(12, 2, 1)
    sensors["satellite"].x = torch.ones(12, 2, 2)
    return builder, sensors, coords


@pytest.mark.parametrize("kind", ["radar", "satellite"])
@pytest.mark.parametrize("normalisation", ["global", "per_node"])
def test_default_nine_neighbours_and_optional_reverse(attachment, kind, normalisation):
    builder, sensors, coords = attachment
    builder.edge_normalisation = normalisation
    forward = (kind, "connects", "raingauge")
    reverse = ("raingauge", "rev_connects", kind)
    graphs = builder.add_heterodata(sensors, coords, kind)
    original = [g.clone() for g in graphs]
    for graph in graphs:
        assert reverse not in graph.edge_types
        edge = graph[forward]
        # The first nine sensors are the nearest to every gauge in this fixture.
        for gauge in range(len(graph.station_ids)):
            selected = edge.edge_index[1] == gauge
            assert set(edge.edge_index[0, selected].tolist()) == set(range(9))
        assert torch.isfinite(edge.edge_attr).all()
        assert (edge.edge_attr > 0).all()
        assert edge.edge_attr.max() == 1
        if normalisation == "per_node":
            for gauge in edge.edge_index[1].unique():
                assert edge.edge_attr[edge.edge_index[1] == gauge].max() == 1
    enabled = builder.add_heterodata(sensors, coords, kind, reverse_cross_support=True)
    for before, graph in zip(original, enabled, strict=True):
        torch.testing.assert_close(
            graph[forward].edge_index, before[forward].edge_index
        )
        torch.testing.assert_close(graph[forward].edge_attr, before[forward].edge_attr)
        assert torch.equal(graph[reverse].edge_index, graph[forward].edge_index.flip(0))
        weights = graph[reverse].edge_attr
        assert torch.isfinite(weights).all() and (weights > 0).all()
        if normalisation == "per_node":
            for sensor in graph[reverse].edge_index[1].unique():
                assert weights[graph[reverse].edge_index[1] == sensor].max() == 1
        else:
            assert weights.max() == 1
    # Reattaching with the default must remove an earlier reverse relation too.
    for graph in builder.add_heterodata(sensors, coords, kind):
        assert reverse not in graph.edge_types


@pytest.mark.parametrize("value", ["false", 0, None])
def test_direct_attachment_rejects_nonboolean_reverse(attachment, value):
    builder, sensors, coords = attachment
    with pytest.raises(ValueError, match="reverse_cross_support must be a boolean"):
        builder.add_heterodata(sensors, coords, "radar", reverse_cross_support=value)
    assert builder.fused_train_heterodata is None
