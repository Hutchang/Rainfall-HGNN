"""Z-score the rainfall input per node type from the training graph only (D08.2).

Paper §4.3: "Features are z-score normalized per node type using training-split
statistics only." Graph files stay in physical millimetres; this runs after
Component 06 has validated them. Only the rainfall input channel is scaled. Validity
flags and positional features are left alone, and targets (``y``) and predictions stay
in millimetres.
"""

import math

import torch

RAINFALL_CHANNEL = 0


def _observed(graph, node_type):
    """Entries of the rainfall channel that hold a real observation."""
    node = graph[node_type]
    if node_type == "raingauge":
        return node.observed
    if node_type == "satellite":
        return node.x[:, :, 1].bool()
    # Radar's one-channel contract has no validity flag: every pixel-hour is observed.
    return torch.ones_like(node.x[:, :, RAINFALL_CHANNEL], dtype=torch.bool)


def rainfall_scaling(train_graph, prefix):
    """Mean and spread of observed rainfall per node type in a training graph."""
    statistics = {}
    for node_type in sorted(train_graph.node_types):
        rain = train_graph[node_type].x[:, :, RAINFALL_CHANNEL]
        values = rain[_observed(train_graph, node_type)].double()
        if values.numel() == 0:
            raise ValueError(
                f"{prefix}: no observed {node_type} rainfall in the training graph "
                "to compute input scaling from"
            )
        mean = values.mean().item()
        # Population spread: the training graph is the whole population being scaled.
        std = values.std(correction=0).item()
        if not math.isfinite(std) or std <= 0:
            raise ValueError(
                f"{prefix}: observed {node_type} rainfall in the training graph has "
                f"no spread (std={std}); a z-score is undefined"
            )
        statistics[node_type] = {
            "mean_mm": mean,
            "std_mm": std,
            "observed_values": int(values.numel()),
        }
    return statistics


def apply_rainfall_scaling(graph, statistics):
    """Scale observed rainfall inputs in place; unobserved entries stay at zero.

    Zero is the same placeholder a masked gauge receives, and after scaling it sits at
    the training mean. The validity flag is what tells the model the value is absent.
    """
    for node_type, values in statistics.items():
        rain = graph[node_type].x[:, :, RAINFALL_CHANNEL]
        observed = _observed(graph, node_type)
        rain[observed] = (rain[observed] - values["mean_mm"]) / values["std_mm"]
