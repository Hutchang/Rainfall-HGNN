import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import AddLaplacianEigenvectorPE, ToUndirected

class RadarGraphNew:
    def __init__(
            self,
            radar_tensor: torch.Tensor,
            coords_df: pd.DataFrame,
            knn: int = 8, # hardcoded to 8 because it just connects to the 8 surrounding nodes
            distance_metric: str = "haversine_km", # hardcoded to haversine_km for now, can be changed later if needed
    ):

        self.radar_tensor = radar_tensor
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
                
                # Inverse weighting (1/d)
                weight = 1.0 / dist_km if dist_km > 0 else 1e-6
                
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

        # fill heterodata object
        self.heterodata["radar"].x = self.radar_tensor
        self.heterodata["radar","connects","radar"].edge_index = edge_index
        self.heterodata["radar","connects","radar"].edge_attr = edge_attr

        return self.heterodata









        

