"""Tests for Component 03's spatial grouping helper."""

from itertools import chain
from typing import cast

import numpy as np
import pytest

from src.study.schema import StationRecord
from src.study.split import (
    assign_fold_roles,
    assign_spatial_clusters,
    convert_coordinates_to_km,
    create_test_folds,
    validate_station_records,
)


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
