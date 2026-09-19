"""Leak-safe wrapper around the retained GNNInductiveHetero architecture."""

import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.models.graphs import HourlyGraphDataset
from src.models.models import GNNInductiveHetero


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def model_spec(graph, settings):
    return {
        "in_channels_dict": {
            kind: graph[kind].x.shape[-1] for kind in graph.node_types
        },
        "hidden_channels": settings["hidden_channels"],
        "out_channels": 1,
        "num_layers": settings["num_layers"],
        "edge_types": list(graph.edge_types),
        "dropout": settings["dropout"],
    }


def masked_forward(model, batch, mask):
    x_dict = dict(batch.x_dict)
    x_dict["raingauge"] = x_dict["raingauge"].clone()
    x_dict["raingauge"][mask, :2] = 0.0
    weights = {kind: batch[kind].edge_attr for kind in batch.edge_types}
    return model(x_dict, batch.edge_index_dict, weights)["raingauge"].flatten()


def loss_sum(predictions, targets, settings):
    # Same physical-mm weighted-MSE formula as training.logic_hetero.weighted_mse.
    alpha = (
        settings["weighted_loss_alpha"] if settings["loss"] == "weighted_mse" else 0.0
    )
    return (
        (1.0 + alpha * torch.log1p(targets.clamp(min=0)))
        * (predictions - targets).square()
    ).sum()


def loader(graph, settings):
    return DataLoader(
        HourlyGraphDataset(graph), batch_size=settings["batch_size"], shuffle=False
    )


def train_epoch(model, graph, optimizer, settings):
    model.train()
    total_loss, total_count = 0.0, 0
    for batch in loader(graph, settings):
        batch = batch.to(settings["device"])
        node = batch["raingauge"]
        count = int((node.mask & node.observed).sum())
        if count == 0:
            continue
        optimizer.zero_grad()
        # Retain inherited leave-one-gauge-out training, without counting missing y.
        for position in range(graph["raingauge"].x.shape[0]):
            hidden = node.node_position == position
            selected = hidden & node.mask & node.observed
            if not selected.any():
                continue
            output = masked_forward(model, batch, hidden)
            loss = loss_sum(output[selected], node.y.flatten()[selected], settings)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / count).backward()
            total_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_count += count
    if total_count == 0:
        raise ValueError("training graph contains no observed target rows")
    return total_loss / total_count


@torch.no_grad()
def validation_loss(model, graph, settings):
    model.eval()
    total, count = 0.0, 0
    for batch in loader(graph, settings):
        batch = batch.to(settings["device"])
        node = batch["raingauge"]
        selected = node.mask & node.observed
        output = masked_forward(model, batch, node.mask).clamp(min=0)
        if selected.any():
            loss = loss_sum(output[selected], node.y.flatten()[selected], settings)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            total += loss.item()
            count += int(selected.sum())
    if not count:
        raise ValueError("validation graph contains no observed target rows")
    return total / count


def train_fold(graphs, fold_index, settings, on_epoch=None):
    seed_everything(settings["model_seed"], settings["deterministic"])
    spec = model_spec(graphs["train"], settings)
    model = GNNInductiveHetero(**spec).to(settings["device"])
    # Materialize lazy layers deterministically before optimizer construction.
    first = next(iter(loader(graphs["train"], settings))).to(settings["device"])
    with torch.no_grad():
        masked_forward(model, first, first["raingauge"].mask)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    best, stale, history = None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        train = train_epoch(model, graphs["train"], optimizer, settings)
        validation = validation_loss(model, graphs["validation"], settings)
        row = {
            "fold": fold_index,
            "epoch": epoch,
            "train_loss": train,
            "val_loss": validation,
        }
        history.append(row)
        if best is None or validation < best["val_loss"]:
            best = {
                "fold": fold_index,
                "epoch": epoch,
                "val_loss": validation,
                "model_spec": spec,
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            stale = 0
        else:
            stale += 1
        if on_epoch:
            on_epoch(row, best)
        if stale >= settings["early_stop_patience"]:
            break
    model.load_state_dict(best["state_dict"])
    return model, best, history


def restore_model(checkpoint, settings):
    seed_everything(settings["model_seed"], settings["deterministic"])
    model = GNNInductiveHetero(**checkpoint["model_spec"]).to(settings["device"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_fold(model, graph, fold_index, settings):
    model.eval()
    rows = []
    for batch in loader(graph, settings):
        batch = batch.to(settings["device"])
        node = batch["raingauge"]
        selected = node.mask & node.observed
        output = masked_forward(model, batch, node.mask).clamp(min=0)
        for position, hour, prediction in zip(
            node.node_position[selected].cpu().tolist(),
            node.hour_position[selected].cpu().tolist(),
            output[selected].cpu().tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "method": "hgnn",
                    "fold": fold_index,
                    "station_id": graph.station_ids[position],
                    "timestamp": pd.Timestamp(graph.timestamps[hour]),
                    "predicted_mm": prediction,
                }
            )
    return pd.DataFrame(
        rows, columns=["method", "fold", "station_id", "timestamp", "predicted_mm"]
    )
