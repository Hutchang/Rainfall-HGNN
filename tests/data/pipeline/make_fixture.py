"""Create invented raw-format inputs and configs for a complete offline 02→06 run.

Nothing is downloaded; coordinates and measurements are synthetic test values.
"""

import argparse
import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def make_fixture(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / "raw"
    standard = directory / "standard"
    (raw / "gauge").mkdir(parents=True)
    ids = [f"{i:05}" for i in range(1, 13)]
    rows, index = [], []
    for i, sid in enumerate(ids):
        # Non-collinear points, with one coincident pair to exercise zero distance.
        location = i if i != 1 else 0
        rows.append(
            [
                sid,
                "shared-display-number",
                f"SYNTHETIC {sid}",
                str(-33.95 + 0.025 * (location // 4)),
                str(151.0 + 0.02 * (location % 4)),
            ]
        )
        tips = [[f"2022-01-05T10:{j:02}:00+10:00", 0.5, 10] for j in range(1 + i % 4)]
        daily = [["2022-01-06T09:00:00+10:00", 0.5 * len(tips), 10]]
        for source, relative, values in (
            ("gauge_tips", f"gauge/tips/{sid}/2022-01.json.gz", tips),
            ("gauge_daily", f"gauge/daily/{sid}/2022.json.gz", daily),
        ):
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as stream:
                json.dump([{"data": values}], stream)
            index.append(
                {
                    "source": source,
                    "station_id": sid,
                    "path": relative,
                    "start": "2022-01-01",
                    "end": "2022-01-31",
                    "status": "ok",
                }
            )
    (raw / "gauge/stations.json").write_text(
        json.dumps(
            [
                [
                    "station_id",
                    "station_no",
                    "station_name",
                    "station_latitude",
                    "station_longitude",
                ],
                *rows,
            ]
        )
    )
    hours = pd.date_range("2022-01-04T23:00:00Z", periods=24, freq="h")
    layers = {}
    for layer in ("radar", "satellite"):
        folder = raw / layer
        folder.mkdir()
        rainfall = np.arange(24 * 12, dtype=np.float32).reshape(24, 12) / 10
        values = (
            rainfall
            if layer == "radar"
            else np.stack([rainfall, np.ones_like(rainfall)], axis=-1)
        )
        if layer == "satellite":
            values[0, 0] = [999, 0]
        data_path = f"{layer}/{layer}.pkl"
        coord_path = f"{layer}/coords.csv"
        pd.DataFrame({"timestamp": hours, "data": list(values)}).to_pickle(
            raw / data_path
        )
        pd.DataFrame(
            {
                "latitude": [-33.95 + 0.05 * (i // 4) for i in range(12)],
                "longitude": [151.0 + 0.05 * (i % 4) for i in range(12)],
            }
        ).to_csv(raw / coord_path, index=False)
        for relative in (data_path, coord_path):
            index.append(
                {
                    "source": layer,
                    "station_id": "",
                    "path": relative,
                    "start": "2022-01-05",
                    "end": "2022-01-06",
                    "status": "ok",
                }
            )
        layers[layer] = {
            "data_file": data_path,
            "coordinates_file": coord_path,
            "product": "SYNTHETIC",
            "units": "invented mm",
            "timestamp_convention": "hour start UTC",
            "gauge_correction": "invented independent inputs",
            "evidence": "tests/data/pipeline/make_fixture.py",
        }
    pd.DataFrame(index).to_csv(raw / "index.csv", index=False)
    configs = {}
    canonical = yaml.safe_load(Path("config/canonical/standard.yaml").read_text())
    canonical.update(
        raw_dir=str(raw),
        output_dir=str(standard),
        # The synthetic fixture observes one complete day in its one-month index.
        # Keep production coverage unchanged while making all fixture stations eligible.
        min_coverage=0.03,
        # Its raw folder holds only the synthetic stations, so there is no wider fetch
        # to cut down to the Sydney box.
        restrict_to_bounds=False,
    )
    configs["canonical"] = canonical
    configs["grids"] = {
        "raw_dir": str(raw),
        "output_dir": str(standard),
        # D13.4: build() reports days/hours against the configured period;
        # matches the fixture's own `hours` range above.
        "start": date(2022, 1, 4),
        "end": date(2022, 1, 5),
        "layers": layers,
    }
    split_file = directory / "split.json"
    configs["split"] = {
        "stations_file": str(standard / "stations.csv"),
        "n_folds": 5,
        "validation_fraction": 0.2,
        "n_clusters": 3,
        "output": str(split_file),
    }
    graph = yaml.safe_load(Path("config/graphs/gauge_baseline.yaml").read_text())
    graph.update(
        k_neighbours=2,
        data_dir=str(standard),
        output_dir=str(directory / "graphs"),
        split_file=str(split_file),
        input_layers=["raingauge", "radar", "satellite"],
    )
    configs["graph"] = graph
    model = yaml.safe_load(Path("config/models/hgnn.yaml").read_text())
    model.update(
        gauge_file=str(standard / "gauge_hourly.csv"),
        stations_file=str(standard / "stations.csv"),
        split_file=str(split_file),
        graphs_dir=str(directory / "graphs"),
        input_layers=graph["input_layers"],
        hidden_channels=4,
        num_layers=2,
        epochs=2,
        batch_size=24,
        fixture_label=(
            "12 invented gauges and raw-format observations, invented radar/satellite; "
            "complete component integration check only"
        ),
    )
    configs["model"] = model
    for name, config in configs.items():
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(make_fixture(parser.parse_args().directory))
