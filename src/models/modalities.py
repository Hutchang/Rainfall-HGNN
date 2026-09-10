"""Consume Component 04's radar/satellite feature contract without building graphs."""

import torch


def validate_layer_selection(layers):
    """Require an explicit experiment choice; never silently discard a graph layer."""
    if (
        not isinstance(layers, (list, tuple))
        or any(not isinstance(layer, str) for layer in layers)
        or len(layers) != len(set(layers))
        or "raingauge" not in layers
        or set(layers) - {"raingauge", "radar", "satellite"}
    ):
        raise ValueError(
            "input_layers must contain raingauge and optionally radar/satellite, "
            "with no duplicates or unsupported layers"
        )
    return set(layers)


def prepare_modalities(graph, prefix):
    """Validate supplied layers and mask unavailable satellite payloads on a clone.

    Component 04 owns alignment, coordinates and topology. These checks establish
    tensor compatibility, not source provenance or independence from held-out gauges.
    """
    hours = len(graph.timestamps)
    gauge_dtype = graph["raingauge"].x.dtype
    for kind, width in (("radar", 1), ("satellite", 2)):
        if kind not in graph.node_types:
            continue
        node = graph[kind]
        if "x" not in node:
            raise ValueError(f"{prefix}: missing {kind}.x")
        x = node.x
        if (
            x.ndim != 3
            or x.shape[0] == 0
            or x.shape[1:] != (hours, width)
            or not x.is_floating_point()
            or x.dtype != gauge_dtype
        ):
            raise ValueError(
                f"{prefix}: {kind}.x must be floating [nodes,time,{width}], "
                "with the same time count and dtype as raingauge.x"
            )
        rain = x[:, :, 0]
        if kind == "satellite":
            valid = x[:, :, 1]
            if not torch.all((valid == 0) | (valid == 1)):
                raise ValueError(f"{prefix}: satellite validity must be unchanged 0/1")
            observed = valid.bool()
        else:
            observed = torch.ones_like(rain, dtype=torch.bool)
        if not torch.isfinite(rain[observed]).all() or (rain[observed] < 0).any():
            raise ValueError(
                f"{prefix}: {kind} observed rainfall must be finite and nonnegative; "
                "radar's one-channel contract cannot represent missing pixels"
            )
        # A computational placeholder with validity=0, never an observed dry value.
        rain[~observed] = 0.0

        reaches_gauges = any(
            edge[0] == kind
            and edge[2] == "raingauge"
            and graph[edge].edge_index.shape[1] > 0
            for edge in graph.edge_types
        )
        receives_messages = any(
            edge[2] == kind and graph[edge].edge_index.shape[1] > 0
            for edge in graph.edge_types
        )
        if not reaches_gauges or not receives_messages:
            raise ValueError(
                f"{prefix}: {kind} needs a nonempty relation to raingauge and an "
                "incoming relation for the stacked HGNN; Component 04 must supply "
                "the topology"
            )
