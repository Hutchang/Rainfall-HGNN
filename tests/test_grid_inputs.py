"""TEST01 (D13.1): the program knows where the archive is, and says so."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from src.canonical.grid_inputs import _gadi_hint, check_inputs, find_archive_dates, main

SAMPLE_ZIP = Path("tests/data/ingestion/radar/71_20220103.prcp-m15.zip")


def _config(**layers):
    return {
        "start": date(2022, 1, 1),
        "end": date(2022, 1, 31),
        "layers": layers,
    }


def test_find_archive_dates_reads_embedded_dates_and_ignores_others(tmp_path):
    (tmp_path / "71_20220103.prcp-m15.zip").write_bytes(b"")
    (tmp_path / "manifest.txt").write_bytes(b"")
    nested = tmp_path / "2022" / "prcp-m15"
    nested.mkdir(parents=True)
    (nested / "71_20220104.prcp-m15.zip").write_bytes(b"")
    assert find_archive_dates(tmp_path) == [date(2022, 1, 3), date(2022, 1, 4)]


def test_check_inputs_missing_root_refuses_and_names_it(tmp_path):
    config = _config(
        radar={"root": str(tmp_path / "no-such-dir")},
        satellite={"root": None},
    )
    with pytest.raises(FileNotFoundError, match=r"radar.*no-such-dir"):
        check_inputs(config)


def test_check_inputs_lists_days_found_on_the_real_sample(tmp_path):
    sample_dir = tmp_path / "radar-sample"
    sample_dir.mkdir()
    (sample_dir / SAMPLE_ZIP.name).write_bytes(SAMPLE_ZIP.read_bytes())
    config = _config(radar={"root": str(sample_dir)}, satellite={"root": None})
    report = check_inputs(config)
    assert "radar" in report
    assert "days found in range: 1 of 31" in report
    assert "2022: 1" in report
    assert "satellite" not in report


def test_check_inputs_missing_gdata_root_names_gadi_and_project():
    config = _config(radar={"root": "/g/data/rq0/rainfields3"})
    with pytest.raises(FileNotFoundError, match=r"Gadi.*rq0.*my\.nci\.org\.au"):
        check_inputs(config)


def test_gadi_hint_empty_for_non_gdata_paths(tmp_path):
    assert _gadi_hint(tmp_path / "no-such-dir") == ""


def test_check_inputs_with_no_roots_configured_does_not_raise():
    config = _config(radar={"root": None}, satellite={"root": None})
    assert "nothing to check" in check_inputs(config)


def test_check_inputs_rejects_reversed_period():
    config = {
        "start": date(2022, 2, 1),
        "end": date(2022, 1, 1),
        "layers": {"radar": {"root": None}},
    }
    with pytest.raises(ValueError, match="after end"):
        check_inputs(config)


def test_cli_check_inputs_missing_root_exits_nonzero(tmp_path, capsys):
    config_path = tmp_path / "grids.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "start": date(2022, 1, 1),
                "end": date(2022, 1, 31),
                "layers": {"radar": {"root": str(tmp_path / "absent")}},
            }
        )
    )
    with pytest.raises(SystemExit) as excinfo:
        main([str(config_path), "--check-inputs"])
    assert excinfo.value.code == 1
    assert "BLOCKED" in capsys.readouterr().err
