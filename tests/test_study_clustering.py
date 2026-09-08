"""Tests for Component 03's spatial grouping helper."""

import json
import subprocess
import sys
from itertools import chain
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from src.study.schema import StationRecord
from src.study.split import (
    assign_fold_roles,
    assign_spatial_clusters,
    build_study_split,
    convert_coordinates_to_km,
    create_test_folds,
    load_seed_config,
    load_split_config,
    validate_station_records,
)

REPOSITORY_ROOT = Path(__file__).parent.parent
STUDY_DATA = REPOSITORY_ROOT / "tests" / "data" / "study"


def station(
    station_id: str,
    latitude: float,
    longitude: float,
    *,
    use_in_study: bool = True,
) -> StationRecord:
    return StationRecord(
        station_id=station_id,
        station_no=f"display-{station_id}",
        name=f"Station {station_id}",
        latitude=latitude,
        longitude=longitude,
        hours_observed=1,
        use_in_study=use_in_study,
        exclusion_reason="" if use_in_study else "not eligible",
    )


def test_only_eligible_stations_receive_clusters():
    stations = [
        station("west-1", -33.80, 150.70),
        station("west-2", -33.81, 150.71),
        station("east-1", -33.80, 151.25),
        station("east-2", -33.81, 151.26),
        station("excluded", -33.80, 151.00, use_in_study=False),
    ]

    clusters = assign_spatial_clusters(stations, n_clusters=2, seed=123)
    clustered_station_ids = {
        station_id for station_ids in clusters.values() for station_id in station_ids
    }

    assert clustered_station_ids == {"west-1", "west-2", "east-1", "east-2"}


def test_duplicate_internal_station_ids_are_rejected():
    stations = [
        station("duplicate", -33.9, 151.0),
        station("duplicate", -33.8, 151.1),
    ]

    with pytest.raises(ValueError, match="station_id values must be unique"):
        _ = validate_station_records(stations)


def test_empty_station_collection_is_rejected():
    with pytest.raises(ValueError, match="at least one station record is required"):
        _ = validate_station_records([])


def test_projects_sydney_coordinates_to_kilometres():
    coordinates_degrees = np.asarray(
        [(-34.0, 151.0), (-33.0, 151.0), (-34.0, 152.0)],
        dtype=np.float64,
    )

    coordinates = convert_coordinates_to_km(coordinates_degrees)
    latitude_distance = cast(float, coordinates[1, 1] - coordinates[0, 1])
    longitude_distance = cast(float, coordinates[2, 0] - coordinates[0, 0])

    assert latitude_distance == pytest.approx(111.2, rel=0.01)
    assert longitude_distance == pytest.approx(92.5, rel=0.01)


def test_clustered_stations_are_spread_across_test_folds_exactly_once():
    cluster_assignments = {
        0: [f"west-{index}" for index in range(5)],
        1: [f"east-{index}" for index in range(5)],
    }

    folds = create_test_folds(cluster_assignments, n_folds=5, seed=123)
    repeated = create_test_folds(cluster_assignments, n_folds=5, seed=123)

    assigned_station_ids = [station_id for fold in folds for station_id in fold]
    expected_station_ids = {
        station_id
        for station_ids in cluster_assignments.values()
        for station_id in station_ids
    }
    assert set(assigned_station_ids) == expected_station_ids
    assert len(assigned_station_ids) == len(set(assigned_station_ids))
    assert folds == repeated
    for fold in folds:
        assert sum(station_id.startswith("west-") for station_id in fold) == 1
        assert sum(station_id.startswith("east-") for station_id in fold) == 1


def test_non_test_stations_are_partitioned_into_train_and_validation():
    clusters = {
        0: [f"west-{index}" for index in range(5)],
        1: [f"east-{index}" for index in range(5)],
    }
    test_folds = create_test_folds(clusters, n_folds=5, seed=123)

    folds = assign_fold_roles(test_folds, validation_fraction=0.25, seed=123)
    repeated = assign_fold_roles(test_folds, validation_fraction=0.25, seed=123)
    changed_seed = assign_fold_roles(test_folds, validation_fraction=0.25, seed=456)
    eligible_ids = set(chain.from_iterable(clusters.values()))

    assert folds == repeated
    assert folds != changed_seed
    for fold in folds:
        train = set(fold.ml.train)
        validation = set(fold.ml.validation)
        test = set(fold.ml.test)

        assert len(train) == 6
        assert len(validation) == 2
        assert len(test) == 2
        assert train | validation | test == eligible_ids
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)


def test_config_wires_the_complete_in_memory_split_pipeline():
    stations = [
        station(f"station-{index}", -34.0 + index * 0.02, 150.7 + index * 0.03)
        for index in range(20)
    ]
    stations.append(station("excluded", -33.8, 151.2, use_in_study=False))
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold.yaml"
    )
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")

    output = build_study_split(stations, config, seeds, created="2026-09-06")

    assert output.name == "sydney_5fold"
    assert output.stations_file == "database/standard/stations.csv"
    assert output.n_eligible == 20
    assert output.seed == 123
    assert len(output.folds) == 5


def test_module_command_writes_reproducible_split_json(tmp_path: Path):
    output_path = tmp_path / "split.json"
    config_path = tmp_path / "split.yaml"
    _ = config_path.write_text(
        "\n".join(
            (
                f"stations_file: {STUDY_DATA / 'stations.csv'}",
                "n_folds: 2",
                "validation_fraction: 0.15",
                "n_clusters: 2",
                f"output: {output_path}",
            )
        ),
        encoding="utf-8",
    )

    command = (
        sys.executable,
        "-m",
        "src.study.split",
        str(config_path),
        "--seeds",
        str(REPOSITORY_ROOT / "config" / "seeds.yaml"),
        "--created",
        "2026-09-08",
    )
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    first_bytes = output_path.read_bytes()
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert output_path.read_bytes() == first_bytes
    assert json.loads(first_bytes)["n_eligible"] == 2
