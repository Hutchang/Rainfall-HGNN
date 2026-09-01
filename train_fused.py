import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.graph.gaugegraphnew import (
    GaugeGraphNew,
    HeterogeneousWeatherGraphDatasetInductive,
)
from src.performance_logger import PerformanceLogger
from src.raingauge.utils import load_raingauge_dataset
from src.utils import read_config
from training.logic_hetero import test_model, train_epoch, validate


# 1. Load the config file
def train_fused(config):
    """Run the transitional Greater Sydney gauge-only training path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["training_params"]["batch_size"]
    fold_count = config["training_params"]["fold_count"]

    # 2. Set up experiment folder
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_new"
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    shutil.copy("config.yaml", f"experiments/{experiment_name}/config.yaml")
    perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

    # 3. Load data
    uptime_threshold = config["filters"]["uptime_threshold"]
    start_year = config["dataset_parameters"]["start_year"]
    end_year = config["dataset_parameters"]["end_year"]
    gauge_folder = config["dataset_parameters"]["gauge_folder"]
    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year,
        end=end_year,
        uptime_threshold=uptime_threshold,
        folder_path=gauge_folder,
    )

    # The transitional Sydney export is already hourly. Do not rescale or resample it.
    raingauge_df = raingauge_df.reset_index().drop_duplicates().reset_index(drop=True)

    print(raingauge_df.shape)

    # 5. Stratified sampling
    # A reviewed Sydney split is mandatory. The runner has no alternate regional
    # sampler or silent fallback.
    split_file = config["training_params"]["split_file"]
    import json

    with open(split_file) as f:
        split_payload = json.load(f)
    split_info = split_payload["folds"]
    if len(split_info) < fold_count:
        raise ValueError(f"{split_file} has {len(split_info)} folds, need {fold_count}")
    available_ids = set(raingauge_station_mappings_df["id"].astype(str))
    for fold in split_info[:fold_count]:
        for key in ("train", "validation", "test"):
            ids = [s for s in fold["ml"][key] if str(s) in available_ids]
            dropped = len(fold["ml"][key]) - len(ids)
            if dropped:
                print(
                    f"split_file fold {fold['fold_index']} {key}: {dropped} station(s) not in loaded data, dropped"
                )
            if not ids:
                raise ValueError(
                    f"split_file fold {fold['fold_index']} {key} is empty after intersecting with loaded stations"
                )
            fold["ml"][key] = (
                pd.Series(ids)
                .astype(raingauge_station_mappings_df["id"].dtype)
                .tolist()
            )
    print(f"Using imported split assignment from {split_file}")

    # 6. Build graphs
    edge_normalisation = config["layer_connect"]["edge_normalisation"]
    print(f"Edge normalisation mode: {edge_normalisation}")
    gauge_graph_arr = []
    for i in range(fold_count):
        gauge_graph = GaugeGraphNew(
            raingauge_df,
            raingauge_station_mappings_df,
            split_info=split_info[i],
            knn=config["layer_connect"]["gauge_gauge"],
            edge_normalisation=edge_normalisation,
        )
        gauge_graph_arr.append(gauge_graph)

    # 7. Initialise HGNN model
    hidden_channels = config["model"]["hidden_channels"]
    out_channels = 1
    num_layers = config["model"]["num_layers"]
    dropout = config["model"].get("dropout", 0.0)
    model_arr = []
    # Derive the gauge feature count from the graph rather than duplicating it here.
    in_channels_dict = {
        node_type: gauge_graph_arr[0].get_train_heterodata()[node_type].x.shape[2]
        for node_type in gauge_graph_arr[0].get_train_heterodata().node_types
    }
    for i in range(fold_count):
        model_arr.append(
            GNNInductiveHetero(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                edge_types=gauge_graph_arr[i].get_train_heterodata().edge_types,
                dropout=dropout,
            ).to(device=device)
        )

    # 8. Batch the data
    def compute_norm_stats(heterodata):
        """Compute per-feature mean and std from one split's node features."""
        stats = {}
        for node_type in heterodata.node_types:
            x = heterodata[node_type].x  # [N, T, F]
            mean = x.mean(dim=(0, 1))
            std = x.std(dim=(0, 1)).clamp(min=1e-8)
            stats[node_type] = (mean, std)
        return stats

    def apply_norm(heterodata, stats):
        """Apply precomputed stats to a heterodata object. Only touches .x, never .y."""
        normed = heterodata.clone()
        for node_type in heterodata.node_types:
            if node_type in stats:
                mean, std = stats[node_type]
                normed[node_type].x = (heterodata[node_type].x - mean) / std
        return normed

    train_loader_arr = []
    val_loader_arr = []
    test_loader_arr = []
    for i in range(fold_count):
        train_data = gauge_graph_arr[i].get_train_heterodata()
        val_data = gauge_graph_arr[i].get_validation_heterodata()
        test_data = gauge_graph_arr[i].get_test_heterodata()

        # Compute stats from training split only, apply same stats to val and test
        stats = compute_norm_stats(train_data)

        train_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(train_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(val_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(test_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )

        train_loader_arr.append(train_loader)
        val_loader_arr.append(val_loader)
        test_loader_arr.append(test_loader)

    def train_fold(model, train_loader, val_loader, fold, device="cpu"):
        # CHECK 1: Print initial weights
        # torch.autograd.set_detect_anomaly(True)
        print("Training")
        print(f"Device type: {device}")
        first_param = next(model.parameters())
        print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        training_loss_arr = []
        validation_loss_arr = []
        early = 0
        mini = float("inf")
        stopping_condition = config["training_params"]["early_stop"]
        epochs = 0
        total_epochs = config["training_params"]["epochs"]
        print(f"-----FOLD: {fold}-----")
        training_start = time.time()
        for i in range(total_epochs):
            epoch_start = time.time()
            print(f"-----EPOCH: {i + 1}-----")

            # CHECK 2: Print weight before training
            # weight_before = first_param.data.clone()

            weighted_alpha = config["training_params"].get("weighted_loss_alpha", 0.0)
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weighted_loss_alpha=weighted_alpha,
            )
            print(train_loss)

            validation_loss = validate(
                model, val_loader, device, weighted_loss_alpha=weighted_alpha
            )
            training_loss_arr.append(train_loss)
            validation_loss_arr.append(validation_loss)
            perf.log_epoch(i, train_loss, validation_loss)
            if mini >= validation_loss:
                mini = validation_loss
                early = 0
                torch.save(
                    model.state_dict(),
                    f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth",
                )
                print("✅ model weights saved to weather_gnn_best.pth")
            else:
                early += 1
            epochs += 1
            if early >= stopping_condition:
                print("Early stop loss")
                break

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")

            # CHECK 4: Print gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            print(f"Gradient norm: {total_norm:.6f}")
            epoch_end = time.time()
            print(f"epoch {i} took {epoch_end - epoch_start}")
        training_end = time.time()
        total_time = training_end - training_start
        perf.finalise(total_time)

        print(f"Training took {total_time} seconds over {epochs} epochs")
        plt.plot(training_loss_arr, label="training_loss", color="blue")
        plt.plot(validation_loss_arr, label="validation_loss", color="red")
        plt.legend()
        plt.savefig(
            f"experiments/{experiment_name}/train_loss_plot_{fold}.png", dpi=300
        )
        plt.close()

    fold_metrics = []
    for i in range(fold_count):
        train_fold(
            model_arr[i],
            train_loader=train_loader_arr[i],
            val_loader=val_loader_arr[i],
            fold=i,
            device=device,
        )

        # load the best model (not the last model found)
        model_arr[i].load_state_dict(
            torch.load(
                f"experiments/{experiment_name}/weather_gnn_best_{i}.pth",
                map_location=device,
            )
        )
        metrics_dict = test_model(
            model_arr[i],
            raingauge_station_mappings_df,
            test_loader_arr[i],
            device,
            fold=i,
            experiment_name=experiment_name,
        )
        fold_metrics.append(metrics_dict)

    averaged_metrics_dict = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in [
            "rmse",
            "mae",
            "pearson_r",
            "timestep_rmse",
            "precision",
            "recall",
            "f1",
        ]
    }

    return averaged_metrics_dict


if __name__ == "__main__":
    config = read_config("config.yaml")
    train_fused(config=config)
