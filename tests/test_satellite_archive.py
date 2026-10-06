"""TEST05 (D13.3, D13.5): Himawari CRRPh 10-minute files -> hourly node tables.

`tests/data/ingestion/satellite/` (D13.5) holds a real, complete hour: six
consecutive files cropped from the real archive to the Sydney window (see
that directory's README for how and why). Most tests here still use
synthetic fixtures for cases the real sample can't isolate on its own (an
incomplete hour, a specific fill cell) -- built with the archive's real,
confirmed encoding (uint16 CRRPh scale/offset/fill, the geostationary CF grid
mapping) so they exercise the same decode path a real file would.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr
import yaml

from src.canonical.grids import build
from src.canonical.satellite_archive import (
    TenMinuteFrame,
    _files_in_range,
    build_satellite_grid_data,
    crop_window,
    hourly_means,
    node_coordinates,
    read_geostationary_grid,
    read_ten_minute_file,
    subsampled_flat_indices,
)

REAL_SAMPLE_DIR = Path("tests/data/ingestion/satellite")
# The real Sydney crop box from config/canonical/grids.yaml, matching how the
# committed sample was cropped -- see tests/data/ingestion/README.md.
REAL_SAMPLE_BOX = {
    "min_lat": -34.3,
    "max_lat": -33.4,
    "min_lon": 150.4,
    "max_lon": 151.4,
    "pad_degrees": 0.25,
}

# The real archive's own CF grid-mapping attributes (confirmed via the public
# THREDDS/OPeNDAP metadata against a real crrph file), reused so synthetic
# fixtures exercise the same coordinate derivation as the real reader.
PROJ_ATTRS = {
    "proj_name": "GEOS141",
    "sweep_angle_axis": "y",
    "false_northing": 0,
    "grid_mapping_name": "geostationary",
    "false_easting": 0,
    "satellite_height": 3.5785863e7,
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.3,
    "perspective_point_height": 3.5785863e7,
    "longitude_of_projection_origin": 140.7,
}
NATIVE_SPACING_M = 2000.0
REAL_TITLE = "NWC GEO Convective Rainfall Rate from Cloud Physical Properties"


def _grid(x_m, y_m):
    return {
        "x_m": np.asarray(x_m, dtype=np.float64),
        "y_m": np.asarray(y_m, dtype=np.float64),
        "proj_attrs": PROJ_ATTRS,
    }


def _write_ten_minute_nc(
    path: Path,
    nominal_time: str,
    raw_grid: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> None:
    """A minimal but faithful synthetic 10-minute file: the same uint16 /
    scale_factor / add_offset / _FillValue encoding and CF geostationary grid
    mapping the real archive uses.
    """
    xr.Dataset(
        {
            "crrph_intensity": (
                ("ny", "nx"),
                raw_grid.astype(np.uint16),
                {
                    "units": "mm/h",
                    "scale_factor": np.float64(0.1),
                    "add_offset": np.float64(0.0),
                    "_FillValue": np.uint16(65535),
                },
            ),
            "geostationary": ((), np.int32(0), dict(PROJ_ATTRS)),
        },
        coords={
            "nx": ("nx", np.asarray(x_m, dtype=np.float64), {"units": "m"}),
            "ny": ("ny", np.asarray(y_m, dtype=np.float64), {"units": "m"}),
        },
        attrs={
            "nominal_product_time": nominal_time,
            # The real archive's global attributes.
            "product_name": "CRRPh",
            "title": REAL_TITLE,
            "source": "NWC/GEO version v2021.2",
        },
    ).to_netcdf(path)


def _write_hour_files(directory: Path, start: pd.Timestamp, raws, x_m, y_m) -> None:
    """One file per array in `raws`, ten minutes apart starting at `start`,
    named the way the real archive names them (only the embedded timestamp
    matters to the reader; the rest of the name is cosmetic)."""
    for i, raw in enumerate(raws):
        stamp = (start + pd.Timedelta(minutes=10 * i)).strftime("%Y%m%dT%H%M%S")
        _write_ten_minute_nc(
            directory / f"S_NWC_CRRPh_HIMA08_HIMA-N-NR_{stamp}Z.nc",
            (start + pd.Timedelta(minutes=10 * i)).isoformat().replace("+00:00", "Z"),
            raw,
            x_m,
            y_m,
        )


# ---- TEST05 (D13.5): the real committed sample ------------------------------


def test_files_in_range_finds_all_six_real_sample_files():
    found = _files_in_range(REAL_SAMPLE_DIR, date(2022, 1, 3), date(2022, 1, 3))
    assert len(found) == 6


def test_read_ten_minute_file_decodes_the_real_sample():
    found = _files_in_range(REAL_SAMPLE_DIR, date(2022, 1, 3), date(2022, 1, 3))
    grid_def = read_geostationary_grid(found[0])
    row_slice, col_slice = crop_window(
        grid_def,
        REAL_SAMPLE_BOX["min_lat"] - REAL_SAMPLE_BOX["pad_degrees"],
        REAL_SAMPLE_BOX["max_lat"] + REAL_SAMPLE_BOX["pad_degrees"],
        REAL_SAMPLE_BOX["min_lon"] - REAL_SAMPLE_BOX["pad_degrees"],
        REAL_SAMPLE_BOX["max_lon"] + REAL_SAMPLE_BOX["pad_degrees"],
    )
    frame = read_ten_minute_file(found[0], row_slice, col_slice)
    assert frame.start == pd.Timestamp("2022-01-03 00:00:00", tz="UTC")
    # A real, mostly-dry hour: some cells are valid (finite), some are the
    # file's own fill value -- both are real, neither is invented.
    assert np.isfinite(frame.intensity).any()
    assert not np.isfinite(frame.intensity).all()
    assert np.nanmax(frame.intensity) < 1000  # sanity: not a raw/undecoded value


def test_build_satellite_grid_data_succeeds_end_to_end_on_the_real_sample():
    """Six real files is a complete hour, unlike radar's two-of-four sample,
    so the full crop/average/subsample pipeline runs successfully here."""
    grid = build_satellite_grid_data(
        REAL_SAMPLE_DIR, date(2022, 1, 3), date(2022, 1, 3), REAL_SAMPLE_BOX, 1
    )
    assert grid.layer == "satellite"
    assert list(grid.timestamps) == [pd.Timestamp("2022-01-03 00:00:00", tz="UTC")]
    assert len(grid.coordinates) > 0
    assert grid.values.shape == (1, len(grid.coordinates), 2)
    # A real hour has both valid and invalid cells (day/night terminator
    # crosses this box), and no rain recorded at this particular hour.
    valid = grid.values[0, :, 1]
    assert 0 < valid.mean() < 1
    np.testing.assert_allclose(grid.values[0, :, 0], 0.0)


def test_real_satellite_files_supply_the_reports_declarations():
    """FR04: product, units and correction come from the files' attributes."""
    grid = build_satellite_grid_data(
        REAL_SAMPLE_DIR, date(2022, 1, 3), date(2022, 1, 3), REAL_SAMPLE_BOX, 1
    )
    provenance = grid.provenance
    assert provenance["product"] == f"CRRPh -- {REAL_TITLE} (NWC/GEO version v2021.2)"
    assert provenance["units"] == "mm/h"
    assert provenance["gauge_correction"] == (
        "contributing_gauges listed in 0 of 6 files read"
    )
    assert "padded by 0.25 degrees" in provenance["crop"]
    assert provenance["subsample_rule"].endswith("crop (stride 1)")


# ---- coordinate derivation --------------------------------------------------


def test_read_geostationary_grid_rejects_unexpected_mapping(tmp_path):
    path = tmp_path / "bad.nc"
    bad_attrs = dict(PROJ_ATTRS)
    bad_attrs["grid_mapping_name"] = "albers_conical_equal_area"
    xr.Dataset(
        {
            "crrph_intensity": (("ny", "nx"), np.zeros((2, 2), dtype=np.uint16), {}),
            "geostationary": ((), np.int32(0), bad_attrs),
        },
        coords={
            "nx": ("nx", [0.0, 2000.0], {"units": "m"}),
            "ny": ("ny", [0.0, -2000.0], {"units": "m"}),
        },
    ).to_netcdf(path)
    with pytest.raises(ValueError, match="grid_mapping_name"):
        read_geostationary_grid(path)


def test_node_coordinates_matches_direct_pyproj_transform():
    x_m = np.array([932000.0, 934000.0])
    y_m = np.array([-3408000.0, -3410000.0])
    grid = _grid(x_m, y_m)
    coords = node_coordinates(grid, slice(0, 2), slice(0, 2))
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_cf(PROJ_ATTRS), "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(x_m[0], y_m[0])
    node_id, (got_lat, got_lon) = coords[0]
    assert node_id == 0
    assert got_lat == pytest.approx(lat)
    assert got_lon == pytest.approx(lon)


def test_crop_window_finds_the_real_sydney_box():
    # The full-scale Himawari grid, built from formula (cheap: no file I/O).
    x_m = -5500000.0 + NATIVE_SPACING_M * np.arange(5500)
    y_m = 5500000.0 - NATIVE_SPACING_M * np.arange(5500)
    grid = _grid(x_m, y_m)
    row_slice, col_slice = crop_window(
        grid, min_lat=-34.55, max_lat=-33.15, min_lon=150.15, max_lon=151.65
    )
    # Independently computed (see the D13.3 session's exploration): the padded
    # Sydney box's four corners project to row 4425.3-4484.6, col 3167.7-3242.0.
    assert (row_slice.start, row_slice.stop) == (4425, 4486)
    assert (col_slice.start, col_slice.stop) == (3167, 3244)


def test_crop_window_rejects_a_box_outside_the_grid():
    x_m = -5500000.0 + NATIVE_SPACING_M * np.arange(10)
    y_m = 5500000.0 - NATIVE_SPACING_M * np.arange(10)
    grid = _grid(x_m, y_m)
    with pytest.raises(ValueError, match="outside the satellite grid"):
        crop_window(
            grid, min_lat=-34.55, max_lat=-33.15, min_lon=150.15, max_lon=151.65
        )


# ---- TEST05: hourly averaging of six 10-minute frames ----------------------


def test_hourly_means_averages_six_and_propagates_nan():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    grids = [
        np.array([[np.nan, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
    ]
    frames = [
        TenMinuteFrame(hour + pd.Timedelta(minutes=10 * i), grid)
        for i, grid in enumerate(grids)
    ]
    result = hourly_means(frames)
    assert list(result) == [hour]
    assert np.isnan(result[hour][0, 0])
    assert result[hour][0, 1] == pytest.approx(1.0)
    assert result[hour][1, 1] == pytest.approx(1.0)


def test_hourly_means_drops_an_incomplete_hour():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    frames = [
        TenMinuteFrame(hour + pd.Timedelta(minutes=10 * i), np.zeros((1, 1)))
        for i in range(5)
    ]
    assert hourly_means(frames) == {}


def test_hourly_means_rejects_more_than_six_for_one_hour():
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    offsets = [0, 10, 20, 30, 40, 50, 55]
    frames = [
        TenMinuteFrame(hour + pd.Timedelta(minutes=m), np.zeros((1, 1)))
        for m in offsets
    ]
    with pytest.raises(ValueError, match="expected at most 6"):
        hourly_means(frames)


def test_subsampled_flat_indices_handles_rectangular_grids():
    first = subsampled_flat_indices(4, 6, 2)
    second = subsampled_flat_indices(4, 6, 2)
    assert first == second
    assert len(first) == 2 * 3  # rows 0,2 x cols 0,2,4
    assert first[:2] == [0, 2]


# ---- end-to-end: synthetic files on disk -----------------------------------


def _tiny_grid_and_matching_box():
    """An 8x8 native grid at real Himawari spacing near Sydney, plus the
    lat/lon box (no padding) of its own inner 4x4 region, computed from all
    four of that region's corners.

    A geostationary grid's row/col axes are not exactly parallel to lines of
    constant latitude/longitude (confirmed while building this fixture: the
    true bounding box of a small rotated region, projected back, extends
    beyond the region's own edges). The extra four rows/cols of margin here
    absorb that overshoot so `crop_window` doesn't run off the array; the
    real archive's full 5500x5500 disk has ample margin around Sydney for
    the same reason, which is why this only matters for a fixture this
    small.
    """
    full_x = -5500000.0 + NATIVE_SPACING_M * np.arange(3214, 3222)  # 8
    full_y = 5500000.0 - NATIVE_SPACING_M * np.arange(4452, 4460)  # 8
    inner_x, inner_y = full_x[2:6], full_y[2:6]  # the region the box targets
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_cf(PROJ_ATTRS), "EPSG:4326", always_xy=True
    )
    corners = [
        (inner_x[0], inner_y[0]),
        (inner_x[0], inner_y[-1]),
        (inner_x[-1], inner_y[0]),
        (inner_x[-1], inner_y[-1]),
    ]
    lons, lats = zip(*(transformer.transform(x, y) for x, y in corners), strict=True)
    box = {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
        "pad_degrees": 0.0,
    }
    return full_x, full_y, box


def test_read_ten_minute_file_and_files_in_range(tmp_path):
    x_m, y_m, _ = _tiny_grid_and_matching_box()
    start = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    _write_hour_files(tmp_path, start, [np.full((8, 8), 6, dtype=np.uint16)], x_m, y_m)
    found = _files_in_range(tmp_path, date(2022, 1, 1), date(2022, 1, 31))
    assert len(found) == 1
    frame = read_ten_minute_file(found[0], slice(0, 8), slice(0, 8))
    assert frame.start == start
    np.testing.assert_allclose(frame.intensity, 0.6)


def test_build_satellite_grid_data_end_to_end_subsamples_and_averages(tmp_path):
    x_m, y_m, box = _tiny_grid_and_matching_box()
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    full = np.full((8, 8), 6, dtype=np.uint16)  # 0.6 mm/h everywhere
    first_frame_with_a_gap = full.copy()
    # (1, 1) is the crop window's own (0, 0) -- see _tiny_grid_and_matching_box.
    first_frame_with_a_gap[1, 1] = 65535  # fill: that node invalid this hour
    raws = [first_frame_with_a_gap] + [full] * 5
    _write_hour_files(tmp_path, hour, raws, x_m, y_m)

    grid = build_satellite_grid_data(
        tmp_path, date(2022, 1, 1), date(2022, 1, 31), box, 2
    )

    assert grid.layer == "satellite"
    assert list(grid.timestamps) == [hour]
    assert len(grid.coordinates) == 9  # stride 2 over the cropped 6x6 window
    assert grid.values.shape == (1, 9, 2)
    rain, valid = grid.values[0, :, 0], grid.values[0, :, 1]
    assert rain[0] == 0.0
    assert valid[0] == 0.0
    np.testing.assert_allclose(rain[1:], 0.6)
    assert (valid[1:] == 1.0).all()


def test_a_satellite_hour_with_five_of_six_files_is_recorded_as_short(tmp_path):
    """Rule 7: the report lists a short hour with how many files it had."""
    x_m, y_m, box = _tiny_grid_and_matching_box()
    full = np.full((8, 8), 6, dtype=np.uint16)
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    _write_hour_files(tmp_path, hour, [full] * 6, x_m, y_m)
    _write_hour_files(tmp_path, hour + pd.Timedelta(hours=1), [full] * 5, x_m, y_m)

    grid = build_satellite_grid_data(
        tmp_path, date(2022, 1, 5), date(2022, 1, 5), box, 2
    )

    assert list(grid.timestamps) == [hour]
    assert grid.provenance["files_per_hour"] == 6
    assert grid.provenance["short_hours"] == {hour + pd.Timedelta(hours=1): 5}


def test_satellite_files_out_of_time_order_stop_the_build(tmp_path):
    """Hours are averaged as their files are read, in filename order. A file
    whose nominal_product_time puts it in an hour already averaged would be
    split off on its own, so it stops the build instead."""
    x_m, y_m, box = _tiny_grid_and_matching_box()
    full = np.full((8, 8), 6, dtype=np.uint16)
    for name_time, nominal in (("0000", "00:00"), ("0010", "01:00"), ("0020", "00:20")):
        _write_ten_minute_nc(
            tmp_path / f"S_NWC_CRRPh_HIMA08_HIMA-N-NR_20220105T{name_time}00Z.nc",
            f"2022-01-05T{nominal}:00Z",
            full,
            x_m,
            y_m,
        )
    with pytest.raises(ValueError, match="not in time order"):
        build_satellite_grid_data(tmp_path, date(2022, 1, 5), date(2022, 1, 5), box, 2)


def test_grids_build_wires_satellite_archive_and_report_node_count_matches_table(
    tmp_path,
):
    raw_dir = tmp_path / "raw"
    (raw_dir / "index").mkdir(parents=True)
    pd.DataFrame(columns=["path", "status"]).to_csv(
        raw_dir / "index" / "dummy.csv", index=False
    )

    x_m, y_m, box = _tiny_grid_and_matching_box()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    hour = pd.Timestamp("2022-01-05 00:00", tz="UTC")
    full = np.full((8, 8), 6, dtype=np.uint16)
    first_frame_with_a_gap = full.copy()
    first_frame_with_a_gap[1, 1] = 65535  # the crop window's own (0, 0)
    raws = [first_frame_with_a_gap] + [full] * 5
    _write_hour_files(archive_root, hour, raws, x_m, y_m)

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
                    "satellite": {
                        "root": str(archive_root),
                        "crop": box,
                        "subsample_stride": 2,
                        "product": "crrph_intensity",
                        "units": "mm/h",
                        "timestamp_convention": "hour beginning, UTC",
                        "gauge_correction": "no",
                        "evidence": "synthetic test fixture",
                    }
                },
            }
        )
    )

    build(config_path)

    nodes = pd.read_csv(output_dir / "satellite_nodes.csv")
    hourly = pd.read_csv(output_dir / "satellite_hourly.csv").sort_values("node_id")
    report = (output_dir / "grid_report.md").read_text()

    assert len(nodes) == 9
    assert "Nodes: 9;" in report
    assert "UNKNOWN" not in report
    assert list(hourly.columns) == ["node_id", "timestamp", "rain_rate", "valid"]
    by_node = hourly.set_index("node_id")
    assert by_node.loc[0, "rain_rate"] == pytest.approx(0.0)
    assert by_node.loc[0, "valid"] == 0
    np.testing.assert_allclose(by_node.loc[[1, 2, 3], "rain_rate"], 0.6)
    assert (by_node.loc[[1, 2, 3], "valid"] == 1).all()
