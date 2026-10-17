"""load_standard_grid reads the standard grid tables Component 04 consumes.

The layer's name decides its columns. Radar is one value per cell-hour, as
the paper's gridded nodes are (ISS-36 records adding a validity flag);
satellite always carries `valid`.
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


def test_radar_with_a_valid_column_is_refused(tmp_path):
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
    with pytest.raises(ValueError, match="radar: unexpected standard grid columns"):
        load_standard_grid(tmp_path, "radar")


def test_satellite_with_a_valid_column_loads_as_two_features(tmp_path):
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


def test_satellite_without_a_valid_column_is_refused(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "satellite",
        [{"node_id": 0, "timestamp": hour, "rain_rate": 1.5}],
        node_count=1,
    )
    with pytest.raises(ValueError, match="satellite: unexpected standard grid columns"):
        load_standard_grid(tmp_path, "satellite")


def test_a_valid_column_must_be_exactly_zero_or_one(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "satellite",
        [{"node_id": 0, "timestamp": hour, "rain_rate": 1.0, "valid": 0.5}],
        node_count=1,
    )
    with pytest.raises(
        ValueError, match="satellite: availability must be exactly 0 or 1"
    ):
        load_standard_grid(tmp_path, "satellite")


def test_a_radar_hour_with_a_cell_without_a_value_is_set_aside(tmp_path):
    """DEC-01: radar has no validity flag, so the whole hour is set aside for
    Component 04 to drop and list, rather than the table being refused."""
    first, second = "2022-01-05T00:00:00+00:00", "2022-01-05T01:00:00+00:00"
    _write_table(
        tmp_path,
        "radar",
        [
            {"node_id": 0, "timestamp": first, "rain_rate": 1.0},
            {"node_id": 1, "timestamp": first, "rain_rate": None},
            {"node_id": 0, "timestamp": second, "rain_rate": 0.0},
            {"node_id": 1, "timestamp": second, "rain_rate": 2.0},
        ],
        node_count=2,
    )
    grid = load_standard_grid(tmp_path, "radar")
    assert list(grid.timestamps) == [pd.Timestamp(second)]
    assert list(grid.hours_without_values) == [pd.Timestamp(first)]
    np.testing.assert_allclose(grid.values[0, :, 0], [0.0, 2.0])


def test_an_infinite_radar_value_is_still_refused(tmp_path):
    hour = "2022-01-05T00:00:00+00:00"
    _write_table(
        tmp_path,
        "radar",
        [{"node_id": 0, "timestamp": hour, "rain_rate": float("inf")}],
        node_count=1,
    )
    with pytest.raises(ValueError, match="radar: observed rainfall must be finite"):
        load_standard_grid(tmp_path, "radar")
