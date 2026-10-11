"""D09.2: links between neighbouring grid cells, and their weights (paper sec 3.2)."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from src.graph.gaugegraphnew import GaugeGraphNew
from src.graph.radargraphnew import RadarGraphNew, grid_neighbour_edges

KM_PER_DEGREE = 6371.0 * np.pi / 180


def lattice(rows, cols, step_x_km, step_y_km, skew_km=0.0):
    """Cell centres of a rows x cols grid near Sydney, row by row.

    ``skew_km`` shifts each row sideways, as the satellite grid is shifted on the map.
    """
    lat0, lon0 = -33.9, 151.0
    points = []
    for row in range(rows):
        for col in range(cols):
            east = col * step_x_km + row * skew_km
            north = row * step_y_km
            points.append(
                (
                    lat0 + north / KM_PER_DEGREE,
                    lon0 + east / (KM_PER_DEGREE * np.cos(np.radians(lat0))),
                )
            )
    return pd.DataFrame(points, columns=["latitude", "longitude"])


def immediate_neighbours(rows, cols):
    return {
        row * cols + col: {
            (row + dr) * cols + col + dc
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and 0 <= row + dr < rows and 0 <= col + dc < cols
        }
        for row in range(rows)
        for col in range(cols)
    }


@pytest.mark.parametrize(
    ("step_x", "step_y", "skew"),
    # Radar's square 5 km grid; a stretched, skewed grid like satellite's 2 km one,
    # whose longest diagonal (1.82 times its closest spacing) is near the reach limit.
    [(5.0, 5.0, 0.0), (2.1, 2.7, 0.6)],
    ids=["radar-like", "satellite-like"],
)
def test_each_cell_links_to_its_immediate_neighbours_only(step_x, step_y, skew):
    rows, cols = 4, 5
    edge_index, weights = grid_neighbour_edges(
        lattice(rows, cols, step_x, step_y, skew)
    )
    received = {}
    for source, destination in edge_index.T.tolist():
        received.setdefault(destination, set()).add(source)
    expected = immediate_neighbours(rows, cols)
    # Corners get 3 links, edges 5, inside cells 8; nothing two steps away.
    assert received == expected
    pairs = set(map(tuple, edge_index.T.tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert torch.isfinite(weights).all() and (weights > 0).all()


def test_cells_at_the_same_position_get_the_strongest_link():
    coords = lattice(2, 2, 5.0, 5.0)
    coords = pd.concat([coords, coords.iloc[[0]]], ignore_index=True)
    edge_index, weights = grid_neighbour_edges(coords)
    links = dict(zip(map(tuple, edge_index.T.tolist()), weights.tolist(), strict=True))
    # Cell 4 sits on cell 0: their link outranks every real neighbour.
    assert links[4, 0] == links[0, 4] == 1e6
    assert max(w for (_, to), w in links.items() if to == 0) == 1e6
    # Sharing a position does not cost cell 4 its real neighbours.
    assert {source for source, to in links if to == 4} == {0, 1, 2, 3}
    assert all(source != to for source, to in links)


def test_one_cell_has_no_grid_links():
    edge_index, weights = grid_neighbour_edges(lattice(1, 1, 5.0, 5.0))
    assert edge_index.shape == (2, 0) and weights.numel() == 0


def test_old_grid_builder_gives_same_position_cells_the_strongest_link():
    coords = pd.DataFrame(
        {"latitude": [-33.9, -33.9, -33.95], "longitude": [151.0, 151.0, 151.0]}
    )
    graph = RadarGraphNew(pd.DataFrame(np.zeros((2, 3))), coords, knn=2).graph
    candidates = ((0, 1), (0, 0), (1, 1))
    same_position = [(u, v) for u, v in candidates if graph.has_edge(u, v)]
    assert same_position
    assert all(graph[u][v]["weight"] == 1.0 for u, v in same_position)


@pytest.fixture
def builder():
    ids = ["a", "b", "c", "d"]
    hours = pd.date_range("2022-01-01", periods=2, freq="h", tz="UTC")
    mapping = pd.DataFrame(
        {
            "id": ids,
            "latitude": [-33.9] * 4,
            "longitude": [151.0, 151.01, 151.02, 151.03],
            "order": range(4),
        }
    )
    return GaugeGraphNew(
        pd.DataFrame(1.0, index=hours, columns=ids),
        mapping,
        {"ml": {"train": ids[:2], "validation": ids[2:3], "test": ids[3:]}},
        knn=1,
        distance="haversine_km",
        edge_normalisation="per_node",
        both_directions=True,
        include_lpe=False,
    )


def sensors(builder, kind, count):
    layer = HeteroData()
    layer.timestamps = builder.train_heterodata.timestamps
    layer[kind].x = torch.ones(count, 2, 1 if kind == "radar" else 2)
    return layer


@pytest.mark.parametrize("kind", ["radar", "satellite"])
@pytest.mark.parametrize("normalisation", ["per_node", "global"])
def test_every_role_graph_gets_the_same_weighted_grid_links(
    builder, kind, normalisation
):
    builder.edge_normalisation = normalisation
    layer, coords = sensors(builder, kind, 12), lattice(3, 4, 5, 5)
    graphs = builder.add_heterodata(layer, coords, kind)
    relation = (kind, "connects", kind)
    first = graphs[0][relation]
    for graph in graphs:
        edge = graph[relation]
        torch.testing.assert_close(edge.edge_index, first.edge_index)
        torch.testing.assert_close(edge.edge_attr, first.edge_attr)
        # 4 corners x 3 + 6 edge cells x 5 + 2 inside cells x 8.
        assert edge.edge_index.shape[1] == 58
        assert edge.edge_attr.max() == 1
        if normalisation == "per_node":
            for cell in range(12):
                assert edge.edge_attr[edge.edge_index[1] == cell].max() == 1
            # Eq. (1): on a square grid a diagonal neighbour is sqrt(2) times as far.
            diagonal = (edge.edge_index[0] == 0) & (edge.edge_index[1] == 5)
            torch.testing.assert_close(
                edge.edge_attr[diagonal],
                torch.tensor([2**-0.5]),
                rtol=1e-3,
                atol=0,
            )


def test_grid_links_can_be_switched_off(builder):
    layer, coords = sensors(builder, "radar", 12), lattice(3, 4, 5, 5)
    builder.add_heterodata(layer, coords, "radar")
    for graph in builder.add_heterodata(layer, coords, "radar", grid_edges=False):
        assert ("radar", "connects", "radar") not in graph.edge_types
        assert ("radar", "connects", "raingauge") in graph.edge_types


@pytest.mark.parametrize("value", ["true", 1, None])
def test_grid_edges_must_be_a_boolean(builder, value):
    layer, coords = sensors(builder, "radar", 12), lattice(3, 4, 5, 5)
    with pytest.raises(ValueError, match="grid_edges must be a boolean"):
        builder.add_heterodata(layer, coords, "radar", grid_edges=value)
    assert builder.fused_train_heterodata is None
