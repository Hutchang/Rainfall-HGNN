"""TEST02, TEST03, TEST04 (D13.2): daily radar zips -> hourly node tables."""

import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from src.canonical.grids import build
from src.canonical.radar_archive import (
    QuarterHour,
    _daily_zips_in_range,
    build_radar_grid_data,
    hourly_grids,
    read_daily_zip,
    read_quarter_hour,
    subsampled_flat_indices,
)

REAL_SAMPLE_ZIP = Path("tests/data/ingestion/radar/71_20220103.prcp-m15.zip")

# The real archive's own CF grid-mapping attributes (confirmed against
# tests/data/ingestion/radar/71_20220103.prcp-m15.zip), reused so synthetic
# fixtures exercise the same coordinate derivation as the real reader.
PROJ_ATTRS = {
    "grid_mapping_name": "albers_conical_equal_area",
    "standard_parallel": np.array([-32.2, -35.2]),
    "longitude_of_central_meridian": np.float64(151.209),
    "latitude_of_projection_origin": np.float64(-33.7008),
    "false_easting": np.float64(0.0),
    "false_northing": np.float64(0.0),
    "semi_major_axis": np.float64(6378137.0),
    "semi_minor_axis": np.float64(6356752.31414),
}
TIME_ATTRS = {"units": "seconds since 1970-01-01 00:00:00 UTC", "standard_name": "time"}


def _write_quarter_hour_nc(
    path: Path, start_epoch: float, raw_grid: np.ndarray
) -> None:
    """A minimal but faithful synthetic 15-minute file: the same uint16 /
    scale_factor / add_offset / _FillValue encoding and CF grid mapping the
    real archive uses, so it decodes through the exact same code path.
    """
    size = raw_grid.shape[0]
    xr.Dataset(
        {
            "precipitation": (
                ("y", "x"),
                raw_grid.astype(np.uint16),
                {
                    "grid_mapping": "proj",
                    "units": "kg m-2",
                    "scale_factor": np.float64(0.05),
                    "add_offset": np.float64(-0.05),
                    "_FillValue": np.uint16(0),
                },
            ),
            "proj": ((), np.int8(0), dict(PROJ_ATTRS)),
            "start_time": ((), np.float64(start_epoch), dict(TIME_ATTRS)),
            "valid_time": ((), np.float64(start_epoch + 900), dict(TIME_ATTRS)),
        },
        coords={
            "x": ("x", np.arange(size, dtype=np.float64), {"units": "km"}),
            "y": ("y", np.arange(size, dtype=np.float64), {"units": "km"}),
        },
    ).to_netcdf(path)


def _write_hour_zip(
    zip_path: Path, start: pd.Timestamp, raws: list[np.ndarray]
) -> None:
    """One daily zip holding one synthetic 15-minute file per array in `raws`,
    fifteen minutes apart starting at `start`."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path, "w") as zf:
        tmp_dir = Path(tmp)
        for i, raw in enumerate(raws):
            nc_path = tmp_dir / f"member_{i}.nc"
            _write_quarter_hour_nc(nc_path, int(start.timestamp()) + i * 900, raw)
            zf.write(nc_path, arcname=f"member_{i}.nc")


# ---- TEST02: the real committed sample -------------------------------------


def test_read_quarter_hour_decodes_real_sample_values_and_start_time(tmp_path):
    with zipfile.ZipFile(REAL_SAMPLE_ZIP) as zf:
        zf.extractall(tmp_path)
    quarter = read_quarter_hour(tmp_path / "71_20220103_000000.prcp-m15.nc")
    assert quarter.start == pd.Timestamp("2022-01-02 23:45:00", tz="UTC")
    assert quarter.precipitation[336, 427] == pytest.approx(1.0)
    assert quarter.precipitation[13, 503] == pytest.approx(0.05)


def test_read_daily_zip_reads_both_real_members_in_time_order():
    quarters = read_daily_zip(REAL_SAMPLE_ZIP)
    assert [q.start for q in quarters] == [
        pd.Timestamp("2022-01-02 23:45:00", tz="UTC"),
        pd.Timestamp("2022-01-03 00:00:00", tz="UTC"),
    ]


def test_build_radar_grid_data_on_real_sample_alone_has_no_complete_hour(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / REAL_SAMPLE_ZIP.name).write_bytes(REAL_SAMPLE_ZIP.read_bytes())
    with pytest.raises(ValueError, match="no complete hours"):
        build_radar_grid_data(sample_dir, date(2022, 1, 1), date(2022, 1, 31), 10)


# ---- TEST03: synthetic quarter-hours sum to an hour, or are missing --------


def test_hourly_grids_sums_four_and_propagates_nan():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    quarter_grids = [
        np.array([[np.nan, 0.25], [0.25, 0.25]]),
        np.array([[0.25, 0.25], [0.25, 0.25]]),
        np.array([[0.25, 0.25], [0.25, 0.25]]),
        np.array([[0.25, 0.25], [0.25, 0.25]]),
    ]
    quarters = [
        QuarterHour(hour + pd.Timedelta(minutes=15 * i), grid)
        for i, grid in enumerate(quarter_grids)
    ]
    result = hourly_grids(quarters)
    assert list(result) == [hour]
    np.testing.assert_array_equal(
        np.isnan(result[hour]), [[True, False], [False, False]]
    )
    assert result[hour][0, 1] == pytest.approx(1.0)


def test_hourly_grids_drops_an_incomplete_hour():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    quarters = [
        QuarterHour(hour + pd.Timedelta(minutes=15 * i), np.zeros((2, 2)))
        for i in range(3)
    ]
    assert hourly_grids(quarters) == {}


def test_hourly_grids_rejects_more_than_four_for_one_hour():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    # Five offsets that all floor to the same hour -- an anomalous duplicate,
    # not five files that legitimately span two different hours.
    offsets = [0, 15, 30, 45, 50]
    quarters = [
        QuarterHour(hour + pd.Timedelta(minutes=m), np.zeros((1, 1))) for m in offsets
    ]
    with pytest.raises(ValueError, match="expected at most 4"):
        hourly_grids(quarters)


def test_read_daily_zip_and_hourly_grids_sum_four_synthetic_files(tmp_path):
    """The literal FR02/TEST03 case: four synthetic 15-minute files on disk."""
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    raws = [np.full((2, 2), 6, dtype=np.uint16) for _ in range(4)]
    zip_path = tmp_path / "71_20220105.prcp-m15.zip"
    _write_hour_zip(zip_path, hour, raws)

    hours = hourly_grids(read_daily_zip(zip_path))
    assert list(hours) == [hour]
    np.testing.assert_allclose(hours[hour], np.full((2, 2), 1.0))


def test_read_daily_zip_and_hourly_grids_miss_an_incomplete_synthetic_hour(tmp_path):
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    raws = [np.full((2, 2), 6, dtype=np.uint16) for _ in range(3)]
    zip_path = tmp_path / "71_20220105.prcp-m15.zip"
    _write_hour_zip(zip_path, hour, raws)

    assert hourly_grids(read_daily_zip(zip_path)) == {}


# ---- TEST04: deterministic subsample; report node count matches the table -


def test_subsampled_flat_indices_deterministic_and_sized():
    first = subsampled_flat_indices(20, 5)
    second = subsampled_flat_indices(20, 5)
    assert first == second
    assert len(first) == 16  # (20 / 5) ** 2
    assert first[:2] == [0, 5]


def test_daily_zips_in_range_filters_and_sorts_by_embedded_date(tmp_path):
    (tmp_path / "71_20220102.prcp-m15.zip").write_bytes(b"")
    (tmp_path / "71_20220101.prcp-m15.zip").write_bytes(b"")
    (tmp_path / "71_20220301.prcp-m15.zip").write_bytes(b"")  # outside range
    found = _daily_zips_in_range(tmp_path, date(2022, 1, 1), date(2022, 1, 31))
    assert [p.name for p in found] == [
        "71_20220101.prcp-m15.zip",
        "71_20220102.prcp-m15.zip",
    ]


def test_build_radar_grid_data_end_to_end_subsamples_and_sums(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    full = np.full((4, 4), 6, dtype=np.uint16)  # 0.25 mm everywhere
    first_file_with_a_gap = full.copy()
    first_file_with_a_gap[0, 0] = 0  # fill: node (0, 0) missing in one of four
    _write_hour_zip(
        root / "71_20220105.prcp-m15.zip",
        hour,
        [first_file_with_a_gap, full, full, full],
    )

    grid = build_radar_grid_data(root, date(2022, 1, 1), date(2022, 1, 31), 2)

    assert grid.layer == "radar"
    assert list(grid.timestamps) == [hour]
    assert len(grid.coordinates) == 4  # stride 2 over a 4x4 grid
    assert grid.values.shape == (1, 4, 2)  # [rain_rate, valid], same as satellite
    rain, valid = grid.values[0, :, 0], grid.values[0, :, 1]
    # Node (0, 0): fill in one of the four files -> the model cannot accept a
    # null, so rain_rate is 0 and valid is False, not NaN.
    assert rain[0] == 0.0
    assert valid[0] == 0.0
    np.testing.assert_allclose(rain[1:], [1.0, 1.0, 1.0])
    assert (valid[1:] == 1.0).all()


def test_grids_build_wires_radar_archive_and_report_node_count_matches_table(
    tmp_path,
):
    raw_dir = tmp_path / "raw"
    (raw_dir / "index").mkdir(parents=True)
    pd.DataFrame(columns=["path", "status"]).to_csv(
        raw_dir / "index" / "dummy.csv", index=False
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    full = np.full((4, 4), 6, dtype=np.uint16)
    first_file_with_a_gap = full.copy()
    first_file_with_a_gap[0, 0] = 0  # fill: node (0, 0) invalid this hour
    raws = [first_file_with_a_gap, full, full, full]
    _write_hour_zip(archive_root / "71_20220105.prcp-m15.zip", hour, raws)

    output_dir = tmp_path / "standard"
    config_path = tmp_path / "grids.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(raw_dir),
                "output_dir": str(output_dir),
                "start": date(2022, 1, 1),
                "end": date(2022, 1, 31),
                "layers": {
                    "radar": {
                        "root": str(archive_root),
                        "subsample_stride": 2,
                        "product": "prcp-m15",
                        "units": "mm accumulated per hour",
                        "timestamp_convention": "hour beginning, UTC",
                        "gauge_correction": "yes",
                        "evidence": "synthetic test fixture",
                    }
                },
            }
        )
    )

    build(config_path)

    nodes = pd.read_csv(output_dir / "radar_nodes.csv")
    hourly = pd.read_csv(output_dir / "radar_hourly.csv").sort_values("node_id")
    report = (output_dir / "grid_report.md").read_text()

    assert len(nodes) == 4
    assert "Nodes: 4;" in report
    assert "UNKNOWN" not in report
    # radar_hourly.csv carries a valid column, same as satellite_hourly.csv;
    # the invalid node (0, 0) is rain_rate 0 with valid False, not a null.
    assert list(hourly.columns) == ["node_id", "timestamp", "rain_rate", "valid"]
    by_node = hourly.set_index("node_id")
    assert by_node.loc[0, "rain_rate"] == pytest.approx(0.0)
    assert by_node.loc[0, "valid"] == 0
    np.testing.assert_allclose(by_node.loc[[1, 2, 3], "rain_rate"], 1.0)
    assert (by_node.loc[[1, 2, 3], "valid"] == 1).all()
    assert "Missing rain payloads: 1." in report
