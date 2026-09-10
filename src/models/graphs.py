"""Validate and slice Component 04 graphs without rebuilding their topology."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage

from src.models.modalities import prepare_modalities, validate_layer_selection


def load_graph(path: str | Path) -> HeteroData:
    """Restrict deserialization to tensors and the prescribed PyG container."""
    with torch.serialization.safe_globals(
        [
            HeteroData,
            BaseStorage,
            EdgeStorage,
            NodeStorage,
            datetime,
            timezone,
            timedelta,
        ]
    ):
        graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, HeteroData):
        raise TypeError(f"{path}: expected Component 04 HeteroData")
    return graph


def validate_graph(graph, fold, role, gauge, input_layers=("raingauge",)):
    """Check identities, physical units and observed labels before constructing a model.

    Missing labels are selected from the canonical table, never from a zero-filled y.
    Graph rainfall must be physical millimetres: no undocumented inverse transform is
    guessed by Component 06. Channels after validity are preserved as supplied by 04.
    """
    prefix = f"fold {fold['fold_index']} {role}"
    if "raingauge" not in graph.node_types:
        raise ValueError(f"{prefix}: missing raingauge node type")
    expected_layers = validate_layer_selection(input_layers)
    if set(graph.node_types) != expected_layers:
        raise ValueError(
            f"{prefix}: graph node types {sorted(graph.node_types)} do not match "
            f"input_layers {sorted(expected_layers)}; supply the selected Component "
            "04 graphs, without placeholder or silently omitted layers"
        )
    for name in ("station_ids", "timestamps"):
        if name not in graph:
            raise ValueError(f"{prefix}: missing required graph.{name}")
    ids = list(graph.station_ids)
    if not ids or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError(f"{prefix}: station_ids must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{prefix}: duplicate station_ids")
    stamps = [pd.Timestamp(t) for t in graph.timestamps]
    if not stamps or any(
        t.tzinfo is None or t.utcoffset() != timedelta(0) for t in stamps
    ):
        raise ValueError(f"{prefix}: timestamps must be timezone-aware UTC")
    timestamps = pd.DatetimeIndex(stamps).tz_convert("UTC")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{prefix}: timestamps must be unique and sorted")
    ml = fold["ml"]
    expected_ids = set(ml["train"])
    if role in ("validation", "test"):
        expected_ids.update(ml["validation"])
    if role == "test":
        expected_ids.update(ml["test"])
    if set(ids) != expected_ids:
        raise ValueError(
            f"{prefix}: station_ids differ from the prescribed split roles"
        )
    gauge_node = graph["raingauge"]
    for key in ("x", "y", "mask"):
        if key not in gauge_node:
            raise ValueError(f"{prefix}: missing raingauge.{key}")
    x, y, mask = gauge_node.x, gauge_node.y, gauge_node.mask
    if x.shape != (len(ids), len(stamps), 6) or not x.is_floating_point():
        raise ValueError(f"{prefix}: raingauge.x must be floating [nodes,time,6]")
    if y.shape != (len(ids), len(stamps), 1) or not y.is_floating_point():
        raise ValueError(f"{prefix}: raingauge.y must be floating [nodes,time,1]")
    if x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr():
        raise ValueError(f"{prefix}: x and y share storage; 04 must make y a copy")
    expected_mask = torch.tensor([sid in ml[role] for sid in ids], dtype=torch.bool)
    if (
        mask.dtype != torch.bool
        or mask.shape != (len(ids),)
        or not torch.equal(mask, expected_mask)
    ):
        raise ValueError(f"{prefix}: mask does not match {role} station identities")
    canonical = gauge[gauge.station_id.isin(ids)]
    absent_hours = set(canonical.timestamp) - set(timestamps)
    if absent_hours:
        raise ValueError(
            f"{prefix}: graph omits {len(absent_hours)} observed canonical hours"
        )
    wide = canonical.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    values = wide.reindex(index=timestamps, columns=ids).to_numpy(dtype=float).T.copy()
    observed = torch.from_numpy(np.isfinite(values))
    valid = x[:, :, 1]
    if not torch.all((valid == 0) | (valid == 1)) or not torch.equal(
        valid.bool(), observed
    ):
        raise ValueError(
            f"{prefix}: validity must be unchanged 0/1 and match canonical observations"
        )
    true_values = torch.from_numpy(values).to(dtype=y.dtype)
    if not torch.allclose(
        y[:, :, 0][observed], true_values[observed], atol=1e-5, rtol=1e-5
    ):
        raise ValueError(
            f"{prefix}: observed y values differ from canonical rainfall_mm"
        )
    if not torch.allclose(
        x[:, :, 0][observed], true_values[observed], atol=1e-5, rtol=1e-5
    ):
        raise ValueError(
            f"{prefix}: x must contain physical rainfall_mm; normalized graphs need an agreed 04 transform contract"
        )
    if not torch.isfinite(x[:, :, 2:]).all():
        raise ValueError(f"{prefix}: nonfinite positional features")
    if not torch.allclose(
        x[:, :, 2:], x[:, :1, 2:].expand_as(x[:, :, 2:]), rtol=1e-5, atol=1e-6
    ):
        raise ValueError(
            f"{prefix}: positional features must be constant across hours for "
            "this fixed graph; time-varying LPE could expose held-out rainfall"
        )
    for node_type in graph.node_types:
        if "x" not in graph[node_type]:
            raise ValueError(f"{prefix}: missing {node_type}.x")
        features = graph[node_type].x
        if (
            features.ndim != 3
            or features.shape[1] != len(stamps)
            or not features.is_floating_point()
        ):
            raise ValueError(
                f"{prefix}: {node_type}.x must be floating [nodes,time,features]"
            )
    relation = ("raingauge", "connects", "raingauge")
    if relation not in graph.edge_types:
        raise ValueError(f"{prefix}: missing gauge edge relation")
    for edge_type in graph.edge_types:
        if edge_type[0] not in graph.node_types or edge_type[2] not in graph.node_types:
            raise ValueError(f"{prefix}: edge relation refers to an absent node type")
        edge = graph[edge_type]
        idx, weight = edge.edge_index, edge.edge_attr
        if idx.ndim != 2 or idx.shape[0] != 2 or idx.dtype != torch.long:
            raise ValueError(f"{prefix}: invalid edge_index for {edge_type}")
        if (
            weight.shape != (idx.shape[1],)
            or not torch.isfinite(weight).all()
            or (weight < 0).any()
        ):
            raise ValueError(f"{prefix}: invalid edge_attr for {edge_type}")
        for axis, node_type in ((0, edge_type[0]), (1, edge_type[2])):
            if idx.numel() and (
                (idx[axis] < 0).any()
                or (idx[axis] >= graph[node_type].x.shape[0]).any()
            ):
                raise ValueError(
                    f"{prefix}: edge_index outside {node_type} node positions"
                )
    clean = graph.clone()
    prepare_modalities(clean, prefix)
    # This is a masked computational placeholder, not an observed dry-hour label.
    clean["raingauge"].x[:, :, 0][~observed] = 0.0
    clean["raingauge"].observed = observed
    clean.timestamps = [t.isoformat() for t in timestamps]
    return clean


class HourlyGraphDataset(Dataset):
    """The inherited one-hour slicing interface, extended with stable identities."""

    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.graph.timestamps)

    def __getitem__(self, timestep):
        graph = self.graph
        data = HeteroData()
        for node_type in graph.node_types:
            data[node_type].x = graph[node_type].x[:, timestep, :]
        node = data["raingauge"]
        size = graph["raingauge"].x.shape[0]
        node.y = graph["raingauge"].y[:, timestep, :]
        node.mask = graph["raingauge"].mask
        node.observed = graph["raingauge"].observed[:, timestep]
        node.node_position = torch.arange(size)
        node.hour_position = torch.full((size,), timestep, dtype=torch.long)
        for edge_type in graph.edge_types:
            data[edge_type].edge_index = graph[edge_type].edge_index
            data[edge_type].edge_attr = graph[edge_type].edge_attr
        return data
