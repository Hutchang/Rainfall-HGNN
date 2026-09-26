"""FR03 [D13.3]: Himawari CRRPh 10-minute files -> hourly satellite node tables.

Reads `crrph_intensity` (instantaneous convective rain rate, mm/h) from NCI
project `rv74`, not `crrph_accum` or `crr` -- see
`docs/grids/SATELLITE_SOURCE.md` for the investigation and why. Three
decisions recorded there and applied here:

- **Six of six.** An hour needs all six 10-minute files present; fewer is
  missing, not partial -- the same standard FR02 holds radar to, at the
  satellite product's own cadence.
- **Mean, not sum.** `crrph_intensity` is an instantaneous *rate* (mm/h), not
  a per-window accumulation the way radar's `precipitation` is. Averaging the
  six samples approximates the hour's accumulated-equivalent rate; summing
  them would be six times too large and in the wrong units.
- **Crop before subsampling.** The native grid is the full 5500x5500 Himawari
  disk (2 km cells), not pre-cropped the way radar's 512x512 already is. This
  module crops to the configured lat/lon box (padded, so edge nodes keep real
  neighbours) by projecting the box's four corners into the file's own grid
  and taking the bounding row/col window -- a small, deliberate approximation
  for a box this size at this location, since a geostationary projection is
  not exactly axis-aligned with lat/lon.

Coordinates are never stored in the file (confirmed empirically, not
assumed): they are derived from the file's own CF grid mapping
(`geostationary`, not radar's `albers_conical_equal_area`) with `pyproj`, the
same method `src/ingestion/radar/coords.py` uses, kept separate here because
the axis units and grid mapping differ enough that sharing code would need
more abstraction than either module currently earns.

Output feeds `src.canonical.grids.build`: this module only produces a
`GridData` for the satellite layer; CSV shapes, the report and Component 04's
interface are that adapter's existing, unchanged machinery.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from src.canonical.grids import GridData

FILES_PER_HOUR = 6
_DATETIME_RE = re.compile(r"(\d{8}T\d{6})")
_EXPECTED_GRID_MAPPING = "geostationary"
_EXPECTED_AXIS_UNITS = "m"


@dataclass
class TenMinuteFrame:
    """One 10-minute file's cropped intensity grid (mm/h, NaN for missing
    cells) and the UTC start of the window it covers."""

    start: pd.Timestamp
    intensity: np.ndarray


def read_geostationary_grid(nc_path: Path) -> dict:
    """One file's grid definition: x/y coordinates in metres and the
    `geostationary` variable's CF grid-mapping attributes. `decode_cf=False`
    so a units change or an unexpected scale/offset on the axes raises here
    rather than silently scaling every node coordinate.
    """
    with xr.open_dataset(nc_path, decode_cf=False) as ds:
        proj_attrs = dict(ds["geostationary"].attrs)
        if proj_attrs.get("grid_mapping_name") != _EXPECTED_GRID_MAPPING:
            raise ValueError(
                f"{nc_path}: expected grid_mapping_name="
                f"{_EXPECTED_GRID_MAPPING!r}, got "
                f"{proj_attrs.get('grid_mapping_name')!r}"
            )
        return {
            "x_m": _axis_metres(ds["nx"], "nx", nc_path),
            "y_m": _axis_metres(ds["ny"], "ny", nc_path),
            "proj_attrs": proj_attrs,
        }


def _axis_metres(var, name: str, nc_path: Path) -> np.ndarray:
    for attr in ("scale_factor", "add_offset"):
        if attr in var.attrs:
            raise ValueError(
                f"{nc_path}: axis {name} carries {attr}={var.attrs[attr]!r}, "
                "which this reader does not apply"
            )
    units = var.attrs.get("units")
    if units != _EXPECTED_AXIS_UNITS:
        raise ValueError(
            f"{nc_path}: expected axis {name} in {_EXPECTED_AXIS_UNITS!r}, "
            f"got {units!r}"
        )
    return var.values.copy()


def to_wgs84_transformer(proj_attrs: dict) -> pyproj.Transformer:
    crs = pyproj.CRS.from_cf(proj_attrs)
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def node_coordinates(grid: dict, row_slice: slice, col_slice: slice) -> list:
    """(node_id, latitude, longitude) for every cell in the cropped window,
    in row-major (y outer, x inner) order -- the same order
    `crrph_intensity.values[row_slice, col_slice].flatten()` produces.
    """
    transformer = to_wgs84_transformer(grid["proj_attrs"])
    xx, yy = np.meshgrid(grid["x_m"][col_slice], grid["y_m"][row_slice])
    lon, lat = transformer.transform(xx.ravel(), yy.ravel())
    return list(enumerate(zip(lat, lon, strict=True)))


def crop_window(
    grid: dict, min_lat: float, max_lat: float, min_lon: float, max_lon: float
) -> tuple[slice, slice]:
    """Row/col slice bounding the given lat/lon box, found by projecting its
    four corners into the file's own grid. May include a handful of cells
    outside the exact box: a geostationary projection is not perfectly
    axis-aligned with lat/lon, and this is a deliberate approximation rather
    than a precise per-cell mask, accepted because the box is padded anyway.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", pyproj.CRS.from_cf(grid["proj_attrs"]), always_xy=True
    )
    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    x_m, y_m = grid["x_m"], grid["y_m"]
    dx, dy = x_m[1] - x_m[0], y_m[1] - y_m[0]
    rows, cols = [], []
    for lon, lat in corners:
        x, y = transformer.transform(lon, lat)
        cols.append((x - x_m[0]) / dx)
        rows.append((y - y_m[0]) / dy)
    row_start, row_stop = int(np.floor(min(rows))), int(np.ceil(max(rows))) + 1
    col_start, col_stop = int(np.floor(min(cols))), int(np.ceil(max(cols))) + 1
    if row_start < 0 or col_start < 0 or row_stop > len(y_m) or col_stop > len(x_m):
        raise ValueError("crop box falls outside the satellite grid's extent")
    return slice(row_start, row_stop), slice(col_start, col_stop)


def read_ten_minute_file(
    nc_path: Path, row_slice: slice, col_slice: slice
) -> TenMinuteFrame:
    """Decode one 10-minute file's cropped `crrph_intensity` (CF-scaled mm/h,
    NaN where `_FillValue` marks a cell as no-data) and its nominal time.
    Indexing before `.values` reads only the cropped window from disk rather
    than decoding the full 5500x5500 disk first.
    """
    with xr.open_dataset(nc_path) as ds:
        start = pd.Timestamp(ds.attrs["nominal_product_time"])
        intensity = (
            ds["crrph_intensity"]
            .isel(ny=row_slice, nx=col_slice)
            .values.astype(np.float64)
        )
    return TenMinuteFrame(start=start, intensity=intensity)


def hourly_means(frames: list[TenMinuteFrame]) -> dict[pd.Timestamp, np.ndarray]:
    """Average every complete set of six consecutive 10-minute frames into
    one hour-beginning grid. An hour with any other count of files is
    missing, not written. A cell that is NaN in even one contributing frame
    makes that cell's hourly value NaN too; ordinary averaging already does
    this without special handling.
    """
    by_hour: dict[pd.Timestamp, list[np.ndarray]] = {}
    for frame in frames:
        by_hour.setdefault(frame.start.floor("h"), []).append(frame.intensity)
    result = {}
    for hour, grids in by_hour.items():
        if len(grids) == FILES_PER_HOUR:
            result[hour] = np.mean(grids, axis=0)
        elif len(grids) > FILES_PER_HOUR:
            raise ValueError(
                f"{hour}: {len(grids)} ten-minute files, expected at most "
                f"{FILES_PER_HOUR} -- duplicate or misaligned archive files"
            )
    return result


def subsampled_flat_indices(n_rows: int, n_cols: int, stride: int) -> list[int]:
    """Row-major flat indices of a stride subgrid of an `n_rows`-by-`n_cols`
    grid -- deterministic, and not assumed square, unlike radar's cropped
    window. The paper used satellite's native 2 km spacing, so stride 1
    (every cell) is the recorded default; see `config/canonical/grids.yaml`.
    """
    picked_rows = list(range(0, n_rows, stride))
    picked_cols = list(range(0, n_cols, stride))
    return [row * n_cols + col for row in picked_rows for col in picked_cols]


def _files_in_range(root: Path, start: date, end: date) -> list[Path]:
    """Every `.nc` file under `root` whose filename embeds one timestamp in
    [start, end], sorted by that timestamp. Recurses through whatever layout
    the archive uses, the same way `radar_archive._daily_zips_in_range` does.
    """
    found = []
    for path in root.rglob("*.nc"):
        match = _DATETIME_RE.search(path.stem)
        if not match:
            continue
        try:
            timestamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            continue
        if start <= timestamp.date() <= end:
            found.append((timestamp, path))
    return [path for _, path in sorted(found)]


def build_satellite_grid_data(
    root: Path, start: date, end: date, crop: dict, subsample_stride: int
) -> GridData:
    """Read every 10-minute file under `root` in [start, end], crop to
    `crop`'s padded lat/lon box, average complete hours, subsample to the
    stride grid, and return a satellite `GridData`.
    """
    nc_paths = _files_in_range(root, start, end)
    if not nc_paths:
        raise ValueError(
            f"no satellite archive files found under {root} in {start} to {end}"
        )

    grid_def = read_geostationary_grid(nc_paths[0])
    pad = crop.get("pad_degrees", 0.0)
    row_slice, col_slice = crop_window(
        grid_def,
        crop["min_lat"] - pad,
        crop["max_lat"] + pad,
        crop["min_lon"] - pad,
        crop["max_lon"] + pad,
    )

    frames = [read_ten_minute_file(p, row_slice, col_slice) for p in nc_paths]
    hours = hourly_means(frames)
    if not hours:
        raise ValueError(f"no complete hours found under {root} in {start} to {end}")

    n_rows = row_slice.stop - row_slice.start
    n_cols = col_slice.stop - col_slice.start
    indices = subsampled_flat_indices(n_rows, n_cols, subsample_stride)
    coordinates = pd.DataFrame(
        [
            (node_id, lat, lon)
            for node_id, (lat, lon) in node_coordinates(grid_def, row_slice, col_slice)
        ],
        columns=["node_id", "latitude", "longitude"],
    ).iloc[indices].reset_index(drop=True)
    coordinates["node_id"] = np.arange(len(coordinates))

    ordered_hours = sorted(hours)
    rain = np.stack([hours[hour].reshape(-1)[indices] for hour in ordered_hours])
    # The model cannot accept a null, and radar's contract already carries a
    # [value, valid] shape for the same reason: a cell missing in one of its
    # six contributing frames (NaN, per hourly_means) becomes rain_rate 0
    # with valid False, not NaN. An hour's own genuine zero (all six frames
    # agreed, dry) stays 0 with valid True.
    valid = np.isfinite(rain)
    values = np.stack([np.where(valid, rain, 0.0), valid.astype(np.float64)], axis=-1)

    starts = sorted(frame.start for frame in frames)
    native_time_step_minutes = None
    if len(starts) > 1:
        gaps = [b - a for a, b in pairwise(starts)]
        native_time_step_minutes = round(min(gaps).total_seconds() / 60)
    provenance = {
        # D13.4: taken from the files' own attributes, not assumed -- see
        # docs/requirements/D13_SOURCE_GRIDS_REQUIREMENTS.md, FR04. No
        # contributing-gauges entry: satellite is not gauge-corrected.
        "coordinate_system": grid_def["proj_attrs"]["grid_mapping_name"],
        "native_time_step_minutes": native_time_step_minutes,
    }
    return GridData(
        "satellite", pd.DatetimeIndex(ordered_hours), values, coordinates, provenance
    )
