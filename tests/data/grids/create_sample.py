"""Create trusted synthetic pickle inputs matching the team's grid export shapes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def create_sample(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    raw = destination / "raw"
    raw.mkdir()
    sample = json.loads(Path(__file__).with_name("sample.json").read_text())
    layers, entries = {}, []
    for layer in ("radar", "satellite"):
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(sample["timestamps"]),
                "data": [
                    np.asarray(value, dtype=np.float32) for value in sample[layer]
                ],
            }
        ).to_pickle(raw / f"{layer}.pkl")
        pd.DataFrame(sample["coordinates"], columns=["latitude", "longitude"]).to_csv(
            raw / f"{layer}_coords.csv", index=False
        )
        layers[layer] = {
            "data_file": f"{layer}.pkl",
            "coordinates_file": f"{layer}_coords.csv",
            "product": "SYNTHETIC TEST FIXTURE",
            "units": "synthetic",
            "timestamp_convention": "invented UTC hour starts",
            "gauge_correction": "not applicable to invented data",
            "evidence": "tests/data/grids/sample.json; no scientific scores",
        }
        for key in ("data_file", "coordinates_file"):
            entries.append({"path": layers[layer][key], "status": "ok"})
    pd.DataFrame(entries).to_csv(raw / "index.csv", index=False)
    config = destination / "grids.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(raw),
                "output_dir": str(destination / "standard"),
                "layers": layers,
            }
        )
    )
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    create_sample(parser.parse_args().destination)
