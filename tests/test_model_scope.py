"""ISS-39: a run folder states its period, gauge positions and node counts by itself."""

import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.models.run import read_settings, run

MAKE_FIXTURE = runpy.run_path(
    str(Path(__file__).parent / "data/models/make_fixture.py")
)["make_fixture"]


def sample(tmp_path, modalities=()):
    return read_settings(MAKE_FIXTURE(tmp_path / "sample", modalities))


def test_a_scored_run_records_period_gauge_extent_and_graph_nodes(tmp_path):
    settings = sample(tmp_path)
    directory = run(settings, tmp_path / "run")
    scope = yaml.safe_load((directory / "config.yaml").read_text())["data_scope"]

    assert scope["period"] == {
        "first_hour_utc": "2022-01-01T00:00:00+00:00",
        "last_hour_utc": "2022-01-01T05:00:00+00:00",
        "hours": 6,
    }
    assert scope["gauges"]["stations"] == 5
    assert scope["gauges"]["latitude"] == [-33.95, -33.75]
    assert scope["gauges"]["longitude"] == [151.0, 151.2]
    assert scope["gauges"]["per_fold"][0] == {"train": 3, "validation": 1, "test": 1}
    # Train graphs hold training gauges; test graphs hold every gauge in the split.
    assert sorted(scope["graph_nodes"]) == [0, 1, 2, 3, 4]
    assert scope["graph_nodes"][0] == {
        "train": {"raingauge": 3},
        "validation": {"raingauge": 4},
        "test": {"raingauge": 5},
    }
    assert "layers" not in scope

    report = (directory / "report.md").read_text()
    assert "## Data scope" in report
    assert (
        "- Period: 2022-01-01T00:00:00+00:00 to 2022-01-01T05:00:00+00:00, 6 hours"
        in report
    )
    assert "| 0 | train | 3 |" in report


def test_upstream_reports_are_copied_and_missing_ones_are_named(tmp_path):
    settings = sample(tmp_path)
    graphs = Path(settings["graphs_dir"])
    (graphs / "report.md").write_text("# Component 04 graph build\n")
    directory = run(dict(settings, fold_subset=[0]), tmp_path / "run")
    reports = yaml.safe_load((directory / "config.yaml").read_text())["data_scope"][
        "upstream_reports"
    ]

    assert (directory / "inputs/graph_report.md").read_text() == (
        "# Component 04 graph build\n"
    )
    assert len(reports["graph_report.md"]["sha256"]) == 64
    # The fixture has no canonical report or split summary; they are named, not skipped.
    assert "sha256" not in reports["standard_report.md"]
    assert "sha256" not in reports["split_summary.md"]
    report = (directory / "report.md").read_text()
    assert "Status: PARTIAL" in report
    assert "No " + str(Path(settings["gauge_file"]).parent / "report.md") in report


def write_nodes(settings, kind, count):
    pd.DataFrame(
        {
            "node_id": range(count),
            "latitude": [-34.0 + 0.1 * n for n in range(count)],
            "longitude": [150.5 + 0.2 * n for n in range(count)],
        }
    ).to_csv(Path(settings["gauge_file"]).parent / f"{kind}_nodes.csv", index=False)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        (2, {"latitude": [-34.0, -33.9], "longitude": [150.5, 150.7]}),
        (3, "UNKNOWN: "),
        (None, "UNKNOWN: no "),
    ],
)
def test_a_sensor_layer_extent_is_recorded_only_from_a_matching_node_file(
    tmp_path, written, expected
):
    settings = sample(tmp_path, ("radar",))
    if written is not None:
        write_nodes(settings, "radar", written)
    directory = run(dict(settings, fold_subset=[1]), tmp_path / "run")
    scope = yaml.safe_load((directory / "config.yaml").read_text())["data_scope"]
    layer = scope["layers"]["radar"]

    assert scope["graph_nodes"][1]["test"] == {"raingauge": 5, "radar": 2}
    assert layer["nodes"] == 2
    if isinstance(expected, dict):
        assert {key: layer[key] for key in expected} == expected
        assert len(layer["node_file"]["sha256"]) == 64
    else:
        assert layer["extent"].startswith(expected)
        assert "latitude" not in layer
    assert "- radar: 2 nodes" in (directory / "report.md").read_text()
