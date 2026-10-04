"""Tests for Component 03's spatial grouping helper."""

import json
import subprocess
import sys
from itertools import chain
from pathlib import Path
from random import Random
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from src.study.schema import SplitConfig, StationRecord
from src.study.split import (
    assign_fold_roles,
    assign_spatial_clusters,
    build_study_split,
    convert_coordinates_to_km,
    create_test_folds,
    load_seed_config,
    load_split_config,
    read_station_records,
    validate_station_records,
)
from src.study.summary import write_split_summary

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


def test_whole_clusters_are_never_divided_between_test_folds():
    cluster_assignments = {
        cluster: [f"area{cluster}-{index}" for index in range(size)]
        for cluster, size in enumerate([3, 7, 2, 5, 4, 6])
    }

    folds = create_test_folds(
        cluster_assignments, n_folds=3, seed=123, assignment="whole_clusters"
    )
    repeated = create_test_folds(
        cluster_assignments, n_folds=3, seed=123, assignment="whole_clusters"
    )
    round_robin = create_test_folds(cluster_assignments, n_folds=3, seed=123)

    assert folds == repeated
    assert folds != round_robin
    assert sorted(chain.from_iterable(folds)) == sorted(
        chain.from_iterable(cluster_assignments.values())
    )
    for fold in folds:
        hidden = set(fold)
        clusters_in_fold = [
            cluster
            for cluster, station_ids in cluster_assignments.items()
            if hidden & set(station_ids)
        ]
        # Two whole clusters per fold: every station of each one, and nothing else.
        assert len(clusters_in_fold) == 2
        assert hidden == {
            station_id
            for cluster in clusters_in_fold
            for station_id in cluster_assignments[cluster]
        }


def test_whole_clusters_refuses_a_cluster_count_the_folds_cannot_share_evenly():
    cluster_assignments = {cluster: [f"station-{cluster}"] for cluster in range(10)}

    with pytest.raises(ValueError, match="10 clusters cannot be dealt evenly to 4"):
        create_test_folds(
            cluster_assignments, n_folds=4, seed=123, assignment="whole_clusters"
        )


def test_blocked_config_hides_no_station_beside_a_visible_clustermate():
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold_blocked.yaml"
    )
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")
    stations = read_station_records(STUDY_DATA / "stations_sydney.csv")
    clusters = assign_spatial_clusters(stations, config.n_clusters, seeds.split_seed)
    cluster_of = {
        station_id: cluster
        for cluster, station_ids in clusters.items()
        for station_id in station_ids
    }

    output = build_study_split(stations, config, seeds, created="2026-10-05")

    assert config.fold_assignment == "whole_clusters"
    assert output.name == "sydney_5fold_blocked"
    for fold in output.folds:
        hidden_clusters = {cluster_of[station_id] for station_id in fold.ml.test}
        visible_clusters = {
            cluster_of[station_id]
            for station_id in fold.ml.train + fold.ml.validation
        }
        assert len(hidden_clusters) == config.n_clusters // config.n_folds
        assert hidden_clusters.isdisjoint(visible_clusters)

    # The inherited rule on the same stations fails the same check in every fold,
    # which is the gap D08.1 closes.
    inherited = build_study_split(
        stations,
        config.model_copy(update={"fold_assignment": "round_robin"}),
        seeds,
        created="2026-10-05",
    )
    for fold in inherited.folds:
        hidden_clusters = {cluster_of[station_id] for station_id in fold.ml.test}
        visible_clusters = {
            cluster_of[station_id]
            for station_id in fold.ml.train + fold.ml.validation
        }
        assert hidden_clusters & visible_clusters


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


def test_split_summary_reports_fold_sizes_and_draws_the_fold_map(tmp_path: Path):
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold_blocked.yaml"
    )
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")
    stations = read_station_records(STUDY_DATA / "stations_sydney.csv")
    output = build_study_split(stations, config, seeds, created="2026-10-05")

    summary_path, figure_path = write_split_summary(
        output, stations, config, tmp_path / "sydney_5fold_blocked.json"
    )

    assert summary_path == tmp_path / "sydney_5fold_blocked.md"
    assert figure_path == tmp_path / "sydney_5fold_blocked.png"
    # A real PNG, not an empty file: the format's fixed eight-byte signature.
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = summary_path.read_text(encoding="utf-8")
    assert "eligible stations: 148; excluded: 14" in summary
    assert "fold assignment: `whole_clusters`" in summary
    assert "(sydney_5fold_blocked.png)" in summary
    for fold in output.folds:
        row = (
            f"| {fold.fold_index} | {len(fold.ml.train)} "
            f"| {len(fold.ml.validation)} | {len(fold.ml.test)} |"
        )
        assert row in summary


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
                f"stations_file: {STUDY_DATA / 'stations_sydney.csv'}",
                "n_folds: 5",
                "validation_fraction: 0.15",
                "n_clusters: 10",
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
    assert json.loads(first_bytes)["n_eligible"] == 148


def test_extra_component_02_columns_are_accepted():
    """Component 02 writes 13 columns; the 5 this component does not use are ignored."""
    stations = read_station_records(STUDY_DATA / "stations_sydney.csv")

    assert len(stations) == 162
    assert sum(station.use_in_study for station in stations) == 148
    assert stations[0].station_id == "560001"


def test_missing_required_column_is_named(tmp_path: Path):
    """A stations.csv missing a required column fails by name, not on the first row."""
    source = (STUDY_DATA / "stations_sydney.csv").read_text(encoding="utf-8")
    broken = tmp_path / "stations.csv"
    _ = broken.write_text(
        source.replace("use_in_study", "eligible", 1), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"missing required stations.csv columns"):
        _ = read_station_records(broken)


def test_split_does_not_depend_on_station_row_order():
    """The same stations and seed give the same folds whatever order the rows arrive in."""
    stations = read_station_records(STUDY_DATA / "stations_sydney.csv")
    shuffled = list(stations)
    Random(99).shuffle(shuffled)
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold.yaml"
    )
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")

    in_file_order = build_study_split(stations, config, seeds, created="2026-09-11")
    in_shuffled_order = build_study_split(shuffled, config, seeds, created="2026-09-11")

    assert in_file_order.folds == in_shuffled_order.folds


def test_excluded_stations_are_recorded_with_their_reason():
    """Every ineligible station is named with its cause, not just subtracted from a count."""
    stations = read_station_records(STUDY_DATA / "stations_sydney.csv")
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold.yaml"
    )
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")

    output = build_study_split(stations, config, seeds, created="2026-09-11")

    dropped = {station.station_id for station in stations if not station.use_in_study}
    assert {exclusion.station_id for exclusion in output.excluded} == dropped
    assert all(exclusion.reason for exclusion in output.excluded)
    assert [exclusion.station_id for exclusion in output.excluded] == sorted(dropped)
    assert output.n_eligible + len(output.excluded) == len(stations)


def test_a_fold_with_no_test_or_validation_stations_is_rejected():
    """Too few stations for the configured folds fails here, not later during training."""
    stations = [
        station(f"station-{index}", -34.0 + index * 0.05, 150.7 + index * 0.05)
        for index in range(3)
    ]
    config = load_split_config(
        REPOSITORY_ROOT / "config" / "splits" / "sydney_5fold.yaml"
    ).model_copy(update={"n_clusters": 2})
    seeds = load_seed_config(REPOSITORY_ROOT / "config" / "seeds.yaml")

    with pytest.raises(ValueError, match=r"fold \d has an empty (test|validation) set"):
        _ = build_study_split(stations, config, seeds, created="2026-09-11")


def test_validation_fraction_of_zero_is_rejected_by_config():
    """A validation_fraction of 0.0 would leave every fold without a validation set."""
    with pytest.raises(ValidationError):
        _ = SplitConfig(
            stations_file="database/standard/stations.csv",
            n_folds=5,
            validation_fraction=0.0,
            n_clusters=10,
            output="config/splits/sydney_5fold.json",
        )
