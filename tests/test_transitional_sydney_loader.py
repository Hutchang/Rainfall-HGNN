"""Focused checks for the Sydney-only transitional gauge loader."""

import pandas as pd
import pytest

from src.raingauge.utils import load_raingauge_dataset


def _write_year(root, timestamp):
    year_dir = root / "2022"
    year_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": [timestamp, "2022-01-01 01:00:00"],
            "stationId": ["station-a", "station-a"],
            "value": [0.5, 1.0],
        }
    ).to_csv(year_dir / "weather_station_data_2022.csv", index=False)
    pd.DataFrame(
        {
            "id": ["station-a"],
            "latitude": [-33.86],
            "longitude": [151.21],
            "deviceId": ["station-a"],
            "name": ["Sydney test station"],
        }
    ).to_csv(year_dir / "weather_stations_2022.csv", index=False)


def test_loader_preserves_hourly_sydney_values(tmp_path):
    _write_year(tmp_path, "2022-01-01 00:00:00")

    observations, stations = load_raingauge_dataset(
        start=2022,
        end=2022,
        uptime_threshold=0.0,
        folder_path=tmp_path,
    )

    assert observations["station-a"].tolist() == [0.5, 1.0]
    assert observations.index.tolist() == [
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
    ]
    assert stations["id"].tolist() == ["station-a"]


def test_loader_rejects_a_fabricated_timezone_suffix(tmp_path):
    _write_year(tmp_path, "2022-01-01T00:00:00+08:00")

    with pytest.raises(ValueError):
        load_raingauge_dataset(
            start=2022,
            end=2022,
            uptime_threshold=0.0,
            folder_path=tmp_path,
        )
