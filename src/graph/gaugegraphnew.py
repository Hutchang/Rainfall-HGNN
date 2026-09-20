import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import is_undirected

# config = read_config("config.yaml")
# removed the config file reading from this file as it is not needed here


class GaugeGraphNew:
    def __init__(
        self,
        data_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        split_info: dict,
        knn: int,
        edge_normalisation: str = "global",
        distance: str = "euclidean_degrees",
        both_directions: bool = False,
        include_lpe: bool = True,
        lpe_dimensions: int = 4,
        layer_is_directed: bool = False,
    ):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings

        edge_normalisation: "global" (unmodified baseline, default) divides every
            edge weight by the single maximum weight in the graph — a lone
            pathological close pair can crush every other edge (see
            docs/archive/baseline-reconciliation/AUDIT_NIRAMAY.md sec 8.0). "per_node"
            implements the paper's Eq. 1 (Low, Kachhadiya, Herath, Rasnayaka &
            Marshall, "Spatial Support Matters: Geometry-Aware Graph Fusion for
            Rainfall Field Reconstruction", arXiv:2607.01621, sec 3.2):
            w_ij = (1/d_ij) / max_{k in N(j)} (1/d_kj) — each node normalises by
            its own neighbourhood's maximum instead of the global one.

        both_directions: False (unmodified baseline, default) adds one edge per
            pick, i -> j only. An edge i -> j means j receives i's value, so a
            gauge receives from the gauges that picked it rather than from its
            own nearest, and a gauge nobody picked receives nothing at all
            (ISS-25). True matches the paper's symmetric connectivity (sec 3.2,
            Eq. 1): for every pick add j -> i as well.
            The paper's connectivity is symmetric — sec 3.2 connects each point
            node to its k nearest neighbours, and Eq. 1's N(j) is the set j is
            connected to — while only the weights are direction-dependent. The
            two gauges are the same distance apart either way, so both
            directions start from the same raw weight and the per-receiving-node
            normalisation above then scales each at its own destination. A gauge
            may then receive from more than k gauges, which is the intent.
        """
        if edge_normalisation not in ("global", "per_node"):
            raise ValueError(
                "edge_normalisation must be 'global' or 'per_node', "
                f"got {edge_normalisation!r}"
            )
        self.dtype = torch.float32
        self.raingauge_df = data_df[mapping_df["id"].values.tolist()]
        self.mapping_df = mapping_df
        self.split_info = split_info
        self.knn = knn
        self.edge_normalisation = edge_normalisation
        self.distance_metric = distance
        self.both_directions = both_directions
        # added parameters
        self.include_lpe = include_lpe
        self.lpe_dimensions = lpe_dimensions
        self.layer_is_directed = layer_is_directed

        self.train_gauges = split_info["ml"]["train"]
        self.validation_gauges = split_info["ml"]["validation"]
        self.test_gauges = split_info["ml"]["test"]
        self.fused_test_heterodata = None
        self.fused_train_heterodata = None
        self.fused_validation_heterodata = None
        self.inverse_weighted = True

        self.train_mask, self.val_mask, self.test_mask = self.initialise_masks()

        self.train_graph = self.build_graph("train")
        self.validation_graph = self.build_graph("validation")
        self.test_graph = self.build_graph("test")

        self.train_heterodata = self.fill_heterodata("train")
        self.validation_heterodata = self.fill_heterodata("validation")
        self.test_heterodata = self.fill_heterodata("test")

        # Directed weights are already final; ToUndirected would merge opposite
        # directions and destroy normalization by the receiving node.

    def get_train_graph(self):
        return self.train_graph

    def get_validation_graph(self):
        self.validation_graph.validation_mask = self.get_validation_graph_mask()
        return self.validation_graph

    def get_validation_graph_mask(self):
        return np.logical_or(self.train_mask, self.val_mask)

    def get_test_graph(self):
        return self.test_graph

    def get_train_heterodata(self, normalize=True):
        if self.fused_train_heterodata:
            return self.fused_train_heterodata
        else:
            return self.train_heterodata

    def get_validation_heterodata(self, normalize=True):
        if self.fused_validation_heterodata:
            return self.fused_validation_heterodata
        else:
            return self.validation_heterodata

    def get_test_heterodata(self, normalize=True):
        if self.fused_test_heterodata:
            return self.fused_test_heterodata
        else:
            return self.test_heterodata

    def build_graph(self, split: str):
        """
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        returns an nx digraph
        """
        match split:
            case "train":
                mask = self.train_mask
            case "validation":
                mask = np.logical_or(self.train_mask, self.val_mask)
            case "test":
                mask = np.ones(self.mapping_df.shape[0]).astype(bool)
            case _:
                raise ValueError(f"unknown graph role: {split!r}")

        filtered = self.mapping_df[mask].reset_index(drop=True)
        count = len(filtered)
        if type(self.knn) is not int or not 1 <= self.knn < count:
            raise ValueError(f"{split}: k_neighbours must be between 1 and {count - 1}")
        if self.distance_metric == "haversine_km":
            coords = np.radians(filtered[["latitude", "longitude"]].to_numpy())
            metric, scale = "haversine", 6371.0
        elif self.distance_metric == "euclidean_degrees":
            coords = filtered[["longitude", "latitude"]].to_numpy()
            metric, scale = "euclidean", 1.0
        else:
            raise ValueError("distance must be haversine_km or euclidean_degrees")
        if not np.isfinite(coords).all():
            raise ValueError("gauge coordinates must be finite")
        model = NearestNeighbors(
            n_neighbors=self.knn + 1, algorithm="ball_tree", metric=metric
        )
        distances, indices = model.fit(coords).kneighbors(coords)
        graph = nx.DiGraph()
        for i, row in filtered.iterrows():
            graph.add_node(i, lat=row["latitude"], lon=row["longitude"])
            # Coincident nodes can precede self in the returned list. Remove by
            # identity, not by discarding the first neighbour unconditionally.
            neighbours = [
                (j, d * scale)
                for j, d in zip(indices[i], distances[i], strict=True)
                if j != i
            ][: self.knn]
            for j, distance in neighbours:
                weight = (
                    (1 / distance if distance > 0 else 1e6)
                    if self.inverse_weighted
                    else distance
                )
                graph.add_edge(i, j, weight=weight)
                if self.both_directions:
                    # both_directions: i picked j, so j is also one of the gauges i may
                    # receive from. Same pair, same distance, same raw weight;
                    # the per-receiving-node loop below rescales each direction
                    # against its own destination's strongest incoming edge.
                    graph.add_edge(j, i, weight=weight)
        if self.edge_normalisation == "per_node" and self.inverse_weighted:
            for destination in graph.nodes:
                incoming = list(graph.in_edges(destination, data=True))
                if incoming:
                    maximum = max(data["weight"] for _, _, data in incoming)
                    for _, _, data in incoming:
                        data["weight"] /= maximum
        return graph

    def initialise_masks(self):
        """
        Returns mask tensors

        :param self: Description
        """
        train_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        validation_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        test_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)

        train_mask[
            self.mapping_df["order"][
                self.mapping_df["id"].isin(self.train_gauges)
            ].index.to_list()
        ] = True
        validation_mask[
            self.mapping_df["order"][
                self.mapping_df["id"].isin(self.validation_gauges)
            ].index.to_list()
        ] = True
        test_mask[
            self.mapping_df["order"][
                self.mapping_df["id"].isin(self.test_gauges)
            ].index.to_list()
        ] = True
        return train_mask, validation_mask, test_mask

    def fill_heterodata(self, graph: str) -> HeteroData:
        heterodata = HeteroData()
        # Missing observations use a computational zero placeholder only; the
        # independent validity channel below preserves missing-versus-dry identity.
        rainfall_values = torch.tensor(
            self.raingauge_df.fillna(0.0).values.T, dtype=torch.float32
        )
        rainfall_validity = torch.tensor(
            self.raingauge_df.notna().astype(int).values.T, dtype=torch.int16
        )
        rainfall_features = torch.stack([rainfall_values, rainfall_validity], dim=2)
        heterodata["raingauge"].x = rainfall_features
        heterodata["raingauge"].y = rainfall_values.unsqueeze(
            -1
        ).clone()  # added clone() so that y is a copy of x and not a view

        match graph:
            case "train":
                mask = torch.tensor(self.train_mask, dtype=bool)
                heterodata["raingauge"].mask = torch.ones(
                    int(mask.sum()), dtype=torch.bool
                )
                edges = self.train_graph.edges(data=True)
            case "validation":
                mask = torch.tensor(
                    np.logical_or(self.train_mask, self.val_mask), dtype=bool
                )
                val = self.mapping_df[
                    self.mapping_df["id"].isin(self.validation_gauges)
                    | self.mapping_df["id"].isin(self.train_gauges)
                ]
                heterodata["raingauge"].mask = torch.tensor(
                    val["id"].isin(self.validation_gauges).to_numpy(), dtype=torch.bool
                )
                edges = self.validation_graph.edges(data=True)
            case "test":
                mask = torch.tensor(np.ones(len(self.test_mask)), dtype=bool)
                heterodata["raingauge"].mask = torch.tensor(
                    self.test_mask, dtype=torch.bool
                )
                edges = self.test_graph.edges(data=True)

        heterodata["raingauge"].x = heterodata["raingauge"].x[mask]
        heterodata["raingauge"].y = heterodata["raingauge"].y[mask]

        # Convert mask tensor back to numpy to safely filter the dataframe
        mask_np = mask.numpy()
        heterodata.station_ids = self.mapping_df[mask_np]["id"].tolist()

        # Extract the T timestamps from the index of the original data_df
        heterodata.timestamps = [
            pd.Timestamp(t).isoformat() for t in self.raingauge_df.index
        ]

        edge_index = []
        edge_attr = []
        for source, destination, data in edges:
            edge_index.append([source, destination])
            weight = data["weight"]
            edge_attr.append(weight)
        if self.edge_normalisation == "global":
            max_attr = max(edge_attr)
            edge_attr = [x / max_attr for x in edge_attr]
        # else "per_node": already normalised per node in build_graph
        heterodata["raingauge", "connects", "raingauge"].edge_index = torch.tensor(
            edge_index, dtype=int
        ).T
        heterodata["raingauge", "connects", "raingauge"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float32
        )
        heterodata["raingauge"].num_nodes = torch.tensor(
            heterodata["raingauge"].x.shape[0], dtype=torch.int32
        )

        # Fill laplacian
        if self.include_lpe:
            lpe = self.positional_encoding(
                heterodata["raingauge", "connects", "raingauge"].edge_index,
                heterodata["raingauge"].x.shape[0],
            )
            timestamps = heterodata["raingauge"].x.shape[1]
            lpe_expanded = lpe.unsqueeze(1).expand(-1, timestamps, -1)
            heterodata["raingauge"].x = torch.cat(
                [heterodata["raingauge"].x, lpe_expanded], dim=2
            )

        return heterodata

    def positional_encoding(
        self, edge_index: torch.Tensor, node_count: int
    ) -> torch.Tensor:
        """Return the numbers describing where each gauge sits in the graph.

        These are the eigenvectors of the graph Laplacian with the smallest
        non-zero eigenvalues — a gauge's structural address, kept visible when
        its rainfall is hidden so the model knows which gauge it is predicting
        (paper secs 3.1 and 3.4).

        `get_laplacian` uses the edge list exactly as it is given, so with one
        direction per pick the matrix is not symmetric, its eigenvectors can be
        complex, and PyTorch Geometric keeps only their real part. With both
        directions present the matrix is symmetric, so we check and say so,
        and the symmetric solver is used instead of the general one. The
        two agree on a symmetric matrix; saying so stops us relying on that.

        The sign of each eigenvector is still drawn at random inside the
        transform, which is a separate question from both_directions and is being
        measured before anything is changed about it (ISS-27).
        """
        if self.lpe_dimensions >= node_count:
            raise ValueError("LPE dimensions must be smaller than the node count")
        transform = AddLaplacianEigenvectorPE(
            k=self.lpe_dimensions,
            attr_name="laplacian_pe",
            is_undirected=bool(is_undirected(edge_index, num_nodes=node_count)),
        )
        return transform(
            Data(
                x=torch.zeros(node_count, 1),
                edge_index=edge_index,
                num_nodes=node_count,
            )
        ).laplacian_pe

    def add_heterodata(self, heterodata_layer, coords, layer_name, knn=4):
        """Attach an aligned point layer using WGS84 latitude/longitude order.

        Numerical correction: finite zero-distance weights, receiving-node
        normalization on each directed relation, and no coordinate-axis swap.
        """
        if layer_name not in {"radar", "satellite"}:
            raise ValueError("only Australian radar/satellite layers are supported")
        if heterodata_layer.timestamps != self.train_heterodata.timestamps:
            raise ValueError("sensor timestamps must exactly match the gauge graph")
        if len(coords) != heterodata_layer[layer_name].x.shape[0]:
            raise ValueError("sensor coordinates and feature node count differ")
        if self.fused_train_heterodata is None:
            self.fused_train_heterodata = self.train_heterodata.clone()
            self.fused_validation_heterodata = self.validation_heterodata.clone()
            self.fused_test_heterodata = self.test_heterodata.clone()
        masks = (
            self.train_mask,
            self.train_mask | self.val_mask,
            np.ones(len(self.test_mask), dtype=bool),
        )
        graphs = self.get_fused_heterodata()
        for graph, mask in zip(graphs, masks, strict=True):
            graph[layer_name].x = heterodata_layer[layer_name].x.clone()
            gauge_coords = self.mapping_df.loc[
                mask, ["latitude", "longitude"]
            ].to_numpy()
            pairs, raw_weights = self.connect_graphs(gauge_coords, coords, knn)
            gauge_to_grid = torch.tensor(pairs, dtype=torch.long).T.contiguous()
            for relation, index in (
                ((layer_name, "connects", "raingauge"), gauge_to_grid.flip(0)),
                (("raingauge", "rev_connects", layer_name), gauge_to_grid),
            ):
                weights = torch.tensor(raw_weights, dtype=torch.float32)
                if self.edge_normalisation == "global":
                    weights /= weights.max()
                else:
                    for destination in index[1].unique():
                        selected = index[1] == destination
                        weights[selected] /= weights[selected].max()
                graph[relation].edge_index = index
                graph[relation].edge_attr = weights
        return graphs

    def connect_graphs(self, raingauge_coords, other_coords, knn=16):
        """Return gauge-to-grid neighbours and raw inverse-kilometre weights.

        Both coordinate arrays are latitude then longitude. Sensor order is retained.
        """
        if type(knn) is not int or not 1 <= knn <= len(other_coords):
            raise ValueError("layer k_neighbours exceeds the available sensor nodes")
        gauge = np.radians(np.asarray(raingauge_coords))
        grid = np.radians(other_coords[["latitude", "longitude"]].to_numpy())
        if not np.isfinite(gauge).all() or not np.isfinite(grid).all():
            raise ValueError("sensor attachment coordinates must be finite")
        neighbours = NearestNeighbors(
            n_neighbors=knn, metric="haversine", algorithm="ball_tree"
        )
        distances, indices = neighbours.fit(grid).kneighbors(gauge)
        pairs, weights = [], []
        for i, row in enumerate(indices):
            for j, grid_node in enumerate(row):
                distance = distances[i, j] * 6371.0
                pairs.append((i, int(grid_node)))
                weights.append(1 / distance if distance > 0 else 1e6)
        return pairs, weights

    def get_fused_heterodata(self):
        return (
            self.fused_train_heterodata,
            self.fused_validation_heterodata,
            self.fused_test_heterodata,
        )

    def visualise_graph_split(self):

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        # Build DataFrames from masks with reset_index so positions match
        # graph node labels (which are 0-based integers from enumerate in build_graph)
        train_df = self.mapping_df[self.train_mask].reset_index(drop=True)
        val_df = self.mapping_df[
            np.logical_or(self.train_mask, self.val_mask)
        ].reset_index(drop=True)
        test_df = self.mapping_df.reset_index(drop=True)

        train_pos = {
            i: (row["longitude"], row["latitude"]) for i, row in train_df.iterrows()
        }
        validation_pos = {
            i: (row["longitude"], row["latitude"]) for i, row in val_df.iterrows()
        }
        test_pos = {
            i: (row["longitude"], row["latitude"]) for i, row in test_df.iterrows()
        }

        train_ids = set(self.train_gauges)
        val_ids = set(self.validation_gauges)

        val_colors = [
            "blue" if row["id"] in train_ids else "green"
            for _, row in val_df.iterrows()
        ]
        test_colors = []
        for _, row in test_df.iterrows():
            if row["id"] in train_ids:
                test_colors.append("blue")
            elif row["id"] in val_ids:
                test_colors.append("green")
            else:
                test_colors.append("red")

        # 4. Plotting
        nx.draw(self.train_graph, pos=train_pos, with_labels=True, ax=ax[0])
        nx.draw(
            self.validation_graph,
            pos=validation_pos,
            node_color=val_colors,
            with_labels=True,
            ax=ax[1],
        )
        nx.draw(
            self.test_graph,
            pos=test_pos,
            node_color=test_colors,
            with_labels=True,
            ax=ax[2],
        )

        fig.show()


class HeterogeneousWeatherGraphDatasetInductive(Dataset):
    def __init__(self, heterodata, device="cpu"):

        self.heterodata = heterodata
        self.device = device

        # Graph has shape [N, T, F]
        self.num_timesteps = heterodata["raingauge"].x.shape[1]

        self.mask = heterodata["raingauge"].mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        """Return a PyG Data object for timestep idx"""
        # Node features at this timestep: shape [N, F]
        x = self.heterodata["raingauge"].x[:, idx, :]
        # Labels at this timestep: shape [N, ...]
        y = self.heterodata["raingauge"].y[:, idx, :]

        # Create a PyG Data object
        data = HeteroData()
        data["raingauge"].x = x
        data["raingauge"].y = y

        # added the station ids and timestamps to the data object
        data.station_ids = self.heterodata.station_ids
        data.timestamps = self.heterodata.timestamps[idx]

        for edge_type in self.heterodata.edge_types:
            data[edge_type].edge_index = self.heterodata[edge_type].edge_index
            data[edge_type].edge_attr = self.heterodata[edge_type].edge_attr
        data["raingauge"].mask = torch.tensor(self.heterodata["raingauge"].mask)
        data["raingauge"].num_nodes = self.heterodata["raingauge"].x.shape[0]
        for node_type in self.heterodata.node_types:
            if node_type == "raingauge":
                continue
            data[node_type].x = self.heterodata[node_type].x[:, idx, :]
        return data
