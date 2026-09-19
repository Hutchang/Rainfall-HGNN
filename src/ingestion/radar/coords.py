"""Derive database/raw/radar/radar_node_coords.csv from a Rainfields prcp-m15
NetCDF file's own projection metadata.

Each file carries a real, complete CF grid-mapping definition (Albers Conical Equal
Area, centred on the radar site, WGS84 ellipsoid) plus a fixed 512x512, 0.5 km grid.
That is enough to compute each cell's true latitude/longitude with `pyproj` — no
bounding box or grid spacing is assumed here, only decoded from the source. See
docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md.

Node order matches `precipitation.values.flatten()` (NumPy default: row-major,
iterating the file's own y axis outer and x axis inner) so a `radar_node_coords.csv`
row and the corresponding position in a stored array describe the same cell.
"""

import csv
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

EXPECTED_GRID_MAPPING = "albers_conical_equal_area"
EXPECTED_AXIS_UNITS = "km"


def _axis_metres(var, name: str, nc_path: Path) -> np.ndarray:
    """One axis's values in metres, after checking the file really uses the units
    this module assumes.

    `decode_cf=False` means xarray applies no `scale_factor` or `add_offset`, so a
    file that carried either, or that switched from km to m, would be read at the
    wrong scale silently — every node coordinate off by a constant factor, with no
    exception anywhere. This module's contract is that nothing is assumed, only
    decoded, so the assumption is checked rather than trusted."""
    attrs = var.attrs
    for attr in ("scale_factor", "add_offset"):
        if attr in attrs:
            raise ValueError(
                f"{nc_path}: axis {name} carries {attr}={attrs[attr]!r}, which this "
                "reader does not apply"
            )
    units = attrs.get("units")
    if units != EXPECTED_AXIS_UNITS:
        raise ValueError(
            f"{nc_path}: expected axis {name} in {EXPECTED_AXIS_UNITS!r}, got {units!r}"
        )
    return var.values.copy() * 1000.0


def read_grid(nc_path: Path) -> dict:
    """One file's grid definition: x/y coordinates in metres and the `proj`
    variable's CF grid-mapping attributes."""
    with xr.open_dataset(nc_path, decode_cf=False) as ds:
        proj_attrs = dict(ds["proj"].attrs)
        if proj_attrs.get("grid_mapping_name") != EXPECTED_GRID_MAPPING:
            raise ValueError(
                f"{nc_path}: expected grid_mapping_name={EXPECTED_GRID_MAPPING!r}, "
                f"got {proj_attrs.get('grid_mapping_name')!r}"
            )
        return {
            "x_m": _axis_metres(ds["x"], "x", nc_path),
            "y_m": _axis_metres(ds["y"], "y", nc_path),
            "proj_attrs": proj_attrs,
        }


def _attrs_equal(a: dict, b: dict) -> bool:
    """Attribute-by-attribute equality that tolerates numpy-array-valued attributes
    (e.g. `standard_parallel`), for which `==` on the whole dict is ambiguous."""
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def assert_consistent_grid(grids: list[dict]) -> None:
    """The radar's grid is fixed by its physical siting; every file should describe
    the same one. Fail loudly if a file disagrees rather than silently using it."""
    first = grids[0]
    for g in grids[1:]:
        if not np.array_equal(g["x_m"], first["x_m"]) or not np.array_equal(
            g["y_m"], first["y_m"]
        ):
            raise ValueError("grid x/y coordinates differ between sampled files")
        if not _attrs_equal(g["proj_attrs"], first["proj_attrs"]):
            raise ValueError("projection attributes differ between sampled files")


def to_wgs84_transformer(proj_attrs: dict) -> pyproj.Transformer:
    """Build the exact CRS the file declares (CF grid-mapping attributes -> pyproj),
    rather than assuming a projection or bounding box."""
    crs = pyproj.CRS.from_cf(proj_attrs)
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def node_coordinates(grid: dict) -> list[tuple[int, tuple[float, float]]]:
    """(node_id, latitude, longitude) for every cell, in row-major (y outer, x inner)
    order — the same order `precipitation.values.flatten()` produces."""
    transformer = to_wgs84_transformer(grid["proj_attrs"])
    # yy varies by row (y outer), xx by column (x inner)
    xx, yy = np.meshgrid(grid["x_m"], grid["y_m"])
    lon, lat = transformer.transform(xx.ravel(), yy.ravel())
    return list(enumerate(zip(lat, lon, strict=True)))


def write_node_coords(nc_paths: list[Path], out_csv: Path) -> int:
    """Read one or more sample files (all must agree — see assert_consistent_grid),
    derive real lat/lon per cell from the first, and write radar_node_coords.csv.
    Returns the number of nodes written."""
    grids = [read_grid(p) for p in nc_paths]
    assert_consistent_grid(grids)
    rows = node_coordinates(grids[0])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "latitude", "longitude"])
        for node_id, (lat, lon) in rows:
            w.writerow([node_id, f"{lat:.6f}", f"{lon:.6f}"])
    return len(rows)
