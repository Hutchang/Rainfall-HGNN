"""load_standard_grid reads the standard grid tables Component 04 consumes.

A table's own columns decide whether a layer carries a validity feature,
rather than the layer's name -- satellite has always had one; radar gained
one with the archive reader (D13.2), so a legacy one-feature radar table must
keep working exactly as before.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.canonical.grid_tables import load_standard_grid


def _write_table(directory: Path, layer: str, rows: list[dict], node_count: int):
    pd.DataFrame(rows).to_csv(directory / f"{layer}_hourly.csv", index=False)
    pd.DataFrame(
        {
            "node_id": range(node_count),
            "latitude": [-33.0] * node_count,
            "longitude": [151.0] * node_count,
        }
    ).to_csv(directory / f"{layer}_nodes.csv", index=False)


def test_radar_without_a_valid_column_loads_as_one_feature(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "radar",
        [
            {"node_id": 0, "timestamp": hour, "rain_rate": 1.0},
            {"node_id": 1, "timestamp": hour, "rain_rate": 2.0},
        ],
        node_count=2,
    )
    grid = load_standard_grid(tmp_path, "radar")
    assert grid.values.shape == (1, 2, 1)
    np.testing.assert_allclose(grid.values[0, :, 0], [1.0, 2.0])


def test_radar_with_a_valid_column_loads_as_two_features_and_masks_observed(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "radar",
        [
            {"node_id": 0, "timestamp": hour, "rain_rate": 0.0, "valid": 0},
            {"node_id": 1, "timestamp": hour, "rain_rate": 2.0, "valid": 1},
        ],
        node_count=2,
    )
    grid = load_standard_grid(tmp_path, "radar")
    assert grid.values.shape == (1, 2, 2)
    np.testing.assert_allclose(grid.values[0, :, 0], [0.0, 2.0])
    np.testing.assert_allclose(grid.values[0, :, 1], [0, 1])


def test_satellite_still_requires_a_valid_column(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "satellite",
        [
            {"node_id": 0, "timestamp": hour, "rain_rate": 999.0, "valid": 0},
            {"node_id": 1, "timestamp": hour, "rain_rate": 1.5, "valid": 1},
        ],
        node_count=2,
    )
    grid = load_standard_grid(tmp_path, "satellite")
    assert grid.values.shape == (1, 2, 2)
    np.testing.assert_allclose(grid.values[0, :, 0], [999.0, 1.5])


def test_a_valid_column_must_be_exactly_zero_or_one(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "radar",
        [{"node_id": 0, "timestamp": hour, "rain_rate": 1.0, "valid": 0.5}],
        node_count=1,
    )
    with pytest.raises(ValueError, match="radar: availability must be exactly 0 or 1"):
        load_standard_grid(tmp_path, "radar")
