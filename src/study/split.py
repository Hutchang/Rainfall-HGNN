"""Spatial grouping helpers for Component 03 study splits."""

from collections.abc import Iterable
from math import cos, radians
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
) -> dict[str, int]:
    """Assign eligible stations to reproducible latitude/longitude clusters."""
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

    assignments: dict[str, int] = {}
    for index, station in enumerate(eligible):
        label = cast("np.int64", labels[index])
        assignments[station.station_id] = int(label)
    return assignments
