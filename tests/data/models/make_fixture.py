"""Write a small SYNTHETIC Component 02/03/04 sample for software tests only.

Usage from the repository root:
    uv run --locked python tests/data/models/make_fixture.py experiments/models-fixture
    uv run --locked python -m src.models.run experiments/models-fixture/config.yaml

This is not a production graph or split builder. Its explicit fake station roles and
features exercise the published contracts while the upstream owners finish real data.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData


def make_fixture(directory, modalities=()):
    if set(modalities) - {"radar", "satellite"} or len(modalities) != len(
        set(modalities)
    ):
        raise ValueError("fixture modalities must be unique radar/satellite names")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    ids = ["00001", "00002", "00003", "00004", "00005"]
    hours = pd.date_range("2022-01-01", periods=6, freq="h", tz="UTC")
    gauge = pd.DataFrame(
        [
            {
                "station_id": sid,
                "timestamp": hour.isoformat(),
                "rainfall_mm": float((n + 2 * h) % 7) / 2,
            }
            for n, sid in enumerate(ids)
            for h, hour in enumerate(hours)
            if not (n == 0 and h == 3)
        ]
    )
    gauge.to_csv(directory / "gauge_hourly.csv", index=False)
    pd.DataFrame(
        {
            "station_id": ids,
            "station_no": ["1000"] * 5,
            "latitude": [-33.9, -33.8, -33.85, -33.95, -33.75],
            "longitude": [151.0, 151.1, 151.2, 151.15, 151.05],
            "use_in_study": [True] * 5,
        }
    ).to_csv(directory / "stations.csv", index=False)
    folds = []
    for index in range(5):
        test, validation = ids[index], ids[(index + 1) % 5]
        folds.append(
            {
                "fold_index": index,
                "ml": {
                    "train": [sid for sid in ids if sid not in (test, validation)],
                    "validation": [validation],
                    "test": [test],
                },
            }
        )
    split = {
        "name": "synthetic-component06-only",
        "created": "2026-09-06",
        "n_eligible": 5,
        "seed": 123,
        "folds": folds,
    }
    (directory / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    (directory / "seeds.yaml").write_text("split_seed: 123\nmodel_seed: 123\n")
    indexed = gauge.assign(timestamp=pd.to_datetime(gauge.timestamp)).pivot(
        index="station_id", columns="timestamp", values="rainfall_mm"
    )
    for fold in folds:
        graph_dir = directory / "graphs" / f"fold-{fold['fold_index']}"
        graph_dir.mkdir(parents=True)
        for role in ("train", "validation", "test"):
            included = set(fold["ml"]["train"])
            if role in ("validation", "test"):
                included.update(fold["ml"]["validation"])
            if role == "test":
                included.update(fold["ml"]["test"])
            # Deliberately reverse node identity order to catch positional assumptions.
            node_ids = [sid for sid in reversed(ids) if sid in included]
            values = torch.tensor(
                indexed.reindex(index=node_ids, columns=hours).to_numpy(),
                dtype=torch.float32,
            )
            observed = torch.isfinite(values)
            graph = HeteroData()
            node = graph["raingauge"]
            node.x = torch.zeros((len(node_ids), len(hours), 6), dtype=torch.float32)
            node.x[:, :, 0] = torch.nan_to_num(values)
            node.x[:, :, 1] = observed.float()
            # Fixed fake positional channels: not a scientific LPE computation.
            for position, sid in enumerate(node_ids):
                node.x[position, :, 2:] = torch.tensor([int(sid) / 10, 0.2, 0.3, 0.4])
            node.y = values.unsqueeze(-1).clone()
            node.mask = torch.tensor(
                [sid in fold["ml"][role] for sid in node_ids], dtype=torch.bool
            )
            graph.station_ids = node_ids
            graph.timestamps = [hour.isoformat() for hour in hours]
            edge = graph["raingauge", "connects", "raingauge"]
            pairs = [
                (a, b)
                for a in range(len(node_ids))
                for b in range(len(node_ids))
                if a != b
            ]
            edge.edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
            edge.edge_attr = torch.ones(len(pairs), dtype=torch.float32)
            for kind in modalities:
                # Invented sensor values and edges; no geographic inference here.
                rain = (
                    torch.arange(2 * len(hours), dtype=torch.float32).reshape(
                        2, len(hours), 1
                    )
                    / 10
                )
                graph[kind].x = rain
                if kind == "satellite":
                    graph[kind].x = torch.cat([rain, torch.ones_like(rain)], dim=2)
                    graph[kind].x[0, 1] = torch.tensor([999.0, 0.0])
                grid_to_gauge = torch.tensor(
                    [
                        (sensor, gauge)
                        for sensor in range(2)
                        for gauge in range(len(node_ids))
                    ],
                    dtype=torch.long,
                ).T.contiguous()
                forward = graph[kind, "connects", "raingauge"]
                forward.edge_index = grid_to_gauge
                forward.edge_attr = torch.ones(grid_to_gauge.shape[1])
                reverse = graph["raingauge", "rev_connects", kind]
                reverse.edge_index = grid_to_gauge.flip(0).contiguous()
                reverse.edge_attr = forward.edge_attr.clone()
            torch.save(graph, graph_dir / f"{role}.pt")
    with Path("config/models/hgnn.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config.update(
        {
            "gauge_file": str(directory / "gauge_hourly.csv"),
            "stations_file": str(directory / "stations.csv"),
            "split_file": str(directory / "split.json"),
            "seeds_file": str(directory / "seeds.yaml"),
            "graphs_dir": str(directory / "graphs"),
            "input_layers": ["raingauge", *modalities],
            "epochs": 2,
            "hidden_channels": 4,
            "num_layers": 1,
            "batch_size": 4,
            "learning_rate": 0.05,
            "fixture_label": "five invented stations, six hours, one missing observation; any enabled radar/satellite values and edges are invented; software-only learning rate 0.05; no scientific claims",
        }
    )
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--modalities", nargs="+", choices=["radar", "satellite"], default=[]
    )
    args = parser.parse_args()
    print(make_fixture(args.directory, args.modalities))
