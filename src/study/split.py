"""Spatial grouping helpers for Component 03 study splits."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from math import cos, radians
from pathlib import Path
from random import Random
from typing import cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from src.study.schema import (
    SeedConfig,
    SplitConfig,
    StationRecord,
    StationRoles,
    StudyFold,
    StudySplitOutput,
)

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

    north_south_km = EARTH_RADIUS_KM * np.deg2rad(latitudes - reference_latitude)
    longitude_scale = EARTH_RADIUS_KM * cos(radians(reference_latitude))
    east_west_km = longitude_scale * np.deg2rad(longitudes - reference_longitude)

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


def validate_fold_roles(folds: Sequence[StudyFold], eligible_ids: set[str]) -> None:
    """Enforce Component 03's station-role invariants."""
    all_test_ids: list[str] = []
    for fold in folds:
        train = set(fold.ml.train)
        validation = set(fold.ml.validation)
        test = set(fold.ml.test)

        if train & validation or train & test or validation & test:
            raise ValueError(f"station roles overlap in fold {fold.fold_index}")
        if train | validation | test != eligible_ids:
            raise ValueError(f"fold {fold.fold_index} does not cover every station")

        all_test_ids.extend(fold.ml.test)

    if Counter(all_test_ids) != Counter(eligible_ids):
        raise ValueError("each station must appear in test exactly once")


def assign_fold_roles(
    test_folds: Sequence[Sequence[str]], validation_fraction: float, seed: int
) -> tuple[StudyFold, ...]:
    """Assign non-test stations to validation and train within each fold."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be at least 0 and less than 1")

    all_test_ids = list(chain.from_iterable(test_folds))
    eligible_ids = set(all_test_ids)
    if len(all_test_ids) != len(eligible_ids):
        raise ValueError("each station must appear in test exactly once")

    rng = Random(seed)
    folds: list[StudyFold] = []
    for fold_index, test_ids in enumerate(test_folds):
        test = set(test_ids)
        remaining = sorted(eligible_ids - test)
        rng.shuffle(remaining)

        validation_count = round(len(remaining) * validation_fraction)
        validation = remaining[:validation_count]
        train = remaining[validation_count:]
        folds.append(
            StudyFold(
                fold_index=fold_index,
                ml=StationRoles(
                    train=tuple(sorted(train)),
                    validation=tuple(sorted(validation)),
                    test=tuple(sorted(test)),
                ),
            )
        )

    validate_fold_roles(folds, eligible_ids)
    return tuple(folds)


def load_split_config(path: Path) -> SplitConfig:
    """Load and validate Component 03's split configuration."""
    payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    return SplitConfig.model_validate(payload)


def load_seed_config(path: Path) -> SeedConfig:
    """Load and validate the project's shared seed configuration."""
    payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    return SeedConfig.model_validate(payload)


def build_study_split(
    stations: Iterable[StationRecord],
    config: SplitConfig,
    seeds: SeedConfig,
    created: str,
) -> StudySplitOutput:
    """Run the configured Component 03 splitting pipeline in memory."""
    records = validate_station_records(stations)
    clusters = assign_spatial_clusters(records, config.n_clusters, seeds.split_seed)
    test_folds = create_test_folds(clusters, config.n_folds, seeds.split_seed)
    folds = assign_fold_roles(
        test_folds,
        config.validation_fraction,
        seeds.split_seed,
    )

    return StudySplitOutput(
        name=Path(config.output).stem,
        created=created,
        stations_file=config.stations_file,
        n_eligible=sum(station.use_in_study for station in records),
        seed=seeds.split_seed,
        folds=folds,
    )
