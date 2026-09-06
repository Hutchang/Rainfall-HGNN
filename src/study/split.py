"""Spatial grouping helpers for Component 03 study splits."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from math import cos, radians
from random import Random
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from src.study.schema import StationRecord

EARTH_RADIUS_KM = 6371.0088


def validate_station_records(
    stations: Iterable[StationRecord],
) -> list[StationRecord]:
    """Validate and materialise Component 02's station collection."""
    records = list(stations)
    if not records:
        raise ValueError("at least one station record is required")

    station_ids = [station.station_id for station in records]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("station_id values must be unique")

    return records


def convert_coordinates_to_km(
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert ``[latitude, longitude]`` degrees to local ``[x, y]`` kilometres."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("coordinates must be a non-empty array shaped [stations, 2]")

    latitudes = coordinates[:, 0]
    longitudes = coordinates[:, 1]
    reference_latitude = np.mean(latitudes)
    reference_longitude = np.mean(longitudes)

    north_south_km = EARTH_RADIUS_KM * np.deg2rad(
        latitudes - reference_latitude
    )
    longitude_scale = EARTH_RADIUS_KM * cos(radians(reference_latitude))
    east_west_km = longitude_scale * np.deg2rad(
        longitudes - reference_longitude
    )

    return np.column_stack((east_west_km, north_south_km))


def project_station_coordinates_km(
    stations: Iterable[StationRecord],
) -> NDArray[np.float64]:
    """Project WGS84 coordinates to local kilometre offsets."""
    records = list(stations)
    if not records:
        raise ValueError("at least one station record is required")

    coordinates = np.asarray(
        [(station.latitude, station.longitude) for station in records],
        dtype=np.float64,
    )
    return convert_coordinates_to_km(coordinates)


def assign_spatial_clusters(
    stations: Iterable[StationRecord], n_clusters: int, seed: int
) -> dict[int, list[str]]:
    """Group eligible station IDs into reproducible spatial clusters."""
    records = validate_station_records(stations)
    eligible = [station for station in records if station.use_in_study]

    coordinates = project_station_coordinates_km(eligible)
    labels = cast(
        "NDArray[np.int64]",
        KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init="auto",
        ).fit_predict(coordinates),
    )

    assignments: defaultdict[int, list[str]] = defaultdict(list)
    for index, station in enumerate(eligible):
        label = cast("np.int64", labels[index])
        assignments[int(label)].append(station.station_id)

    return {
        cluster: sorted(station_ids)
        for cluster, station_ids in sorted(assignments.items())
    }


def create_test_folds(
    cluster_assignments: Mapping[int, Sequence[str]], n_folds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Spread clustered stations across test folds exactly once."""
    all_station_ids = list(chain.from_iterable(cluster_assignments.values()))
    if len(all_station_ids) != len(set(all_station_ids)):
        raise ValueError("a station_id cannot appear in more than one cluster")

    rng = Random(seed)
    test_folds: list[list[str]] = [[] for _ in range(n_folds)]
    next_fold = 0
    for cluster in sorted(cluster_assignments):
        station_ids = sorted(cluster_assignments[cluster])
        rng.shuffle(station_ids)
        for station_id in station_ids:
            test_folds[next_fold].append(station_id)
            next_fold = (next_fold + 1) % n_folds

    folds = tuple(tuple(sorted(fold)) for fold in test_folds)
    assigned_counts = Counter(station_id for fold in folds for station_id in fold)
    if assigned_counts != Counter(all_station_ids):
        raise RuntimeError("each station must appear in exactly one test fold")

    return folds
