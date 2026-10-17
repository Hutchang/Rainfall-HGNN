import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import AddLaplacianEigenvectorPE, ToUndirected

# A grid cell's immediate neighbours (up to eight: four along the grid's two directions
# and four diagonals) all lie within this multiple of the cell's own closest spacing,
# and the next ring out starts at twice it. Measured on the D13 sample grids: radar's
# square 5 km grid has diagonals at 1.42, satellite's stretched 2 km grid at up to 1.76.
GRID_REACH = 1.9


def grid_neighbour_edges(coords):
    """Link each grid cell to its immediate neighbours on the grid (paper sec 3.2).

    ``coords`` holds latitude and longitude, one row per node in tensor order. A cell
    at the edge of the grid gets fewer than eight links rather than links to cells two
    steps away. Returns ``(edge_index, weights)``: ``edge_index`` is ``[2, E]``, source
    then destination, with both directions of every link; ``weights`` are inverse
    kilometres (1e6 for cells at the same position), left for the caller to normalise
    at the receiving node as every other relation is.
    """
    points = np.radians(coords[["latitude", "longitude"]].to_numpy(dtype=float))
    if not np.isfinite(points).all():
        raise ValueError("grid coordinates must be finite")
    if len(points) < 2:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    neighbours = NearestNeighbors(
        n_neighbors=min(9, len(points)), metric="haversine", algorithm="ball_tree"
    )
    distances, indices = neighbours.fit(points).kneighbors(points)
    distances = distances * 6371.0
    # Each cell's closest spacing to a cell at a different position.
    spacing = np.array(
        [row[row > 0].min() if (row > 0).any() else np.inf for row in distances]
    )
    links = {}
    for i, (row_distances, row_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        for distance, j in zip(row_distances, row_indices, strict=True):
            j = int(j)
            if j != i and distance <= GRID_REACH * min(spacing[i], spacing[j]):
                links[i, j] = links[j, i] = distance
    pairs = sorted(links)
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    weights = torch.tensor(
        [1 / links[pair] if links[pair] > 0 else 1e6 for pair in pairs],
        dtype=torch.float32,
    )
    return edge_index, weights


class RadarGraphNew:
    def __init__(
            self,
            data_df: pd.DataFrame,
            coords_df: pd.DataFrame,
            knn: int = 8, # hardcoded to 8 because it just connects to the 8 surrounding nodes
            distance_metric: str = "haversine_km", # hardcoded to haversine_km for now, can be changed later if needed
    ):

        self.data_df = data_df
        self.coords_df = coords_df
        self.knn = knn
        self.distance_metric = distance_metric

        self.heterodata = HeteroData()

        self.graph = self.build_graph()
        self.heterodata = self.fill_heterodata()

    def build_graph(self):
        #create a graph (doesn't have to be directed because nodes are 
        # placed in a grid so normalisation is not needed)
        G = nx.Graph()

        #extract the coordinates from the coords_df and convert to a numpy array
        coords = self.coords_df[['latitude', 'longitude']].values
        # convert coords to haversine (not going to make this variable for the moment)
        coords = np.radians(coords)

        # do a nearest node search using nearest neighbor function
        model = NearestNeighbors(n_neighbors=self.knn + 1, metric="haversine")
        model.fit(coords) # fit model with the coords for radar

        # create two arrays that hold this models "neighbor" assignments and the distances
        distances, indices = model.kneighbors(coords)
        distances = distances*6371.0 # convert to km

        # fill in the graph
       
        for i in range(len(coords)):
            row_neighbors = []
            row_weights = []
            
            for j in range(1, self.knn + 1):
                neighbor_idx = indices[i, j]
                dist_km = distances[i, j]
                
                # Inverse weighting (1/d). Cells at the same position are the
                # strongest link, not the weakest (D09.2).
                weight = 1.0 / dist_km if dist_km > 0 else 1e6
                
                row_neighbors.append(neighbor_idx)
                row_weights.append(weight)
                # Find the maximum weight for this specific node (the closest orthogonal edges)

            max_weight = max(row_weights)
            
            # Normalize and attach the edges to the NetworkX graph
            for neighbor_idx, weight in zip(row_neighbors, row_weights):
                normalized_weight = weight / max_weight
                G.add_edge(i, neighbor_idx, weight=normalized_weight)

        return G

    def fill_heterodata(self):
        # this functio  turns the graph into PyG heterodata object
        G = self.graph

        

        # need to make G directed for message passing
        directed_G = G.to_directed()

        edge_list = list(directed_G.edges())
        weight_list = [data["weight"] for _, _, data in directed_G.edges(data=True)]

        # convert to pytorch tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(weight_list, dtype=torch.float32)

        values_tensor = torch.tensor(self.data_df.fillna(0.0).values.T, dtype=torch.float32)
        mask_tensor = torch.tensor((~self.data_df.isna()).values.T, dtype=torch.float32)
        x_tensor = torch.stack([values_tensor, mask_tensor], dim=-1)

        # fill heterodata object
        self.heterodata["radar"].x = x_tensor
        self.heterodata["radar","connects","radar"].edge_index = edge_index
        self.heterodata["radar","connects","radar"].edge_attr = edge_attr

        return self.heterodata









        

