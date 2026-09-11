"""Tests for Component 03's documented station input contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.study.schema import StationRecord
from src.study.split import read_station_records

STUDY_DATA = Path(__file__).parent / "data" / "study"


def valid_station(**overrides: object) -> dict[str, object]:
    station: dict[str, object] = {
        "station_id": "001234",
        "station_no": "0563083",
        "name": "BLACKHEATH G.C.",
        "latitude": -33.635,
        "longitude": 150.285,
        "hours_observed": 17000,
        "use_in_study": True,
        "exclusion_reason": "",
    }
    station.update(overrides)
    return station


def test_validates_station_and_preserves_string_identifiers():
    station = StationRecord.model_validate(valid_station())

    assert station.station_id == "001234"
    assert station.station_no == "0563083"
    assert station.latitude == pytest.approx(-33.635)
    assert station.hours_observed == 17000
    assert station.use_in_study is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", -91.0),
        ("longitude", 181.0),
        ("hours_observed", -1),
    ],
)
def test_rejects_invalid_values(field: str, value: object):
    with pytest.raises(ValidationError):
        _ = StationRecord.model_validate(valid_station(**{field: value}))


def test_rejected_station_requires_an_exclusion_reason():
    with pytest.raises(ValidationError, match="exclusion_reason is required"):
        _ = StationRecord.model_validate(valid_station(use_in_study=False))

    station = StationRecord.model_validate(
        valid_station(use_in_study=False, exclusion_reason="coverage below 0.90")
    )
    assert station.use_in_study is False
    assert station.exclusion_reason == "coverage below 0.90"


def test_included_station_cannot_have_an_exclusion_reason():
    with pytest.raises(ValidationError, match="exclusion_reason must be empty"):
        _ = StationRecord.model_validate(
            valid_station(exclusion_reason="unexpected reason")
        )


def test_reads_the_documented_station_table_without_losing_identifiers():
    stations = read_station_records(STUDY_DATA / "stations.csv")

    assert [station.station_id for station in stations] == [
        "001234",
        "567890",
        "009999",
    ]
    assert stations[0].station_no == "0563083"
    assert stations[2].use_in_study is False
    assert stations[2].exclusion_reason == "coverage below 0.90"
