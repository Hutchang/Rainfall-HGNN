"""FR02 [D13.2]: BoM Rainfields3 `prcp-m15` daily zips -> hourly radar node tables.

Reuses `src/ingestion/radar/coords.py`'s CF grid-mapping decoder for node
coordinates -- the fetcher's reader and coordinate derivation, per the
requirements doc (`docs/requirements/D13_SOURCE_GRIDS_REQUIREMENTS.md`).
Precipitation values and each file's own start time are read with ordinary CF
decoding (`scale_factor`/`add_offset` applied, `_FillValue` masked to NaN);
`coords.py`'s `decode_cf=False` mode is for the x/y projection axes only,
kept there because those should never carry a scale/offset.

Filenames embed each 15-minute file's *end* time (its `valid_time`), not its
start: `71_20220103_000000.prcp-m15.nc` covers 23:45-00:00 the day before, not
00:00-00:15 (confirmed against the committed sample). Grouping by each file's
own `start_time`, floored to the hour, is what correctly buckets four files
into one hour-beginning value; grouping by the filename would silently
misassign every file to the wrong hour.

Output feeds `src.canonical.grids.build`: this module only produces a
`GridData` for the radar layer; CSV shapes, the report and Component 04's
interface are that adapter's existing, unchanged machinery.
"""

import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.canonical.grids import GridData
from src.ingestion.radar.coords import node_coordinates, read_grid

FILES_PER_HOUR = 4
_DATE_RE = re.compile(r"(\d{8})")


@dataclass
class QuarterHour:
    """One 15-minute file's precipitation grid (mm, NaN for missing cells)
    and the UTC start of the window it covers."""

    start: pd.Timestamp
    precipitation: np.ndarray


def read_quarter_hour(nc_path: Path) -> QuarterHour:
    """Decode one 15-minute file: CF-scaled precipitation in mm (NaN where
    the file's own `_FillValue` marks a cell as no-data) and its start time.
    """
    with xr.open_dataset(nc_path) as ds:
        start = pd.Timestamp(ds["start_time"].values.item(), tz="UTC")
        precipitation = ds["precipitation"].values.astype(np.float64)
    return QuarterHour(start=start, precipitation=precipitation)


def read_daily_zip(zip_path: Path) -> list[QuarterHour]:
    """Every 15-minute file inside one day's zip, decoded."""
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(name for name in zf.namelist() if name.endswith(".nc"))
        if not members:
            raise ValueError(f"{zip_path}: zip holds no .nc member")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            return [
                read_quarter_hour(Path(zf.extract(member, tmp_dir)))
                for member in members
            ]


def hourly_grids(quarters: list[QuarterHour]) -> dict[pd.Timestamp, np.ndarray]:
    """Sum every complete set of four consecutive quarter-hours into one
    hour-beginning grid. An hour with any other count of files is missing,
    not written -- source-faithful, per the non-negotiable rules. A cell
    that is NaN in even one contributing file makes that cell's hourly value
    NaN too; ordinary elementwise addition already does this without special
    handling.
    """
    by_hour: dict[pd.Timestamp, list[np.ndarray]] = {}
    for quarter in quarters:
        by_hour.setdefault(quarter.start.floor("h"), []).append(quarter.precipitation)
    result = {}
    for hour, grids in by_hour.items():
        if len(grids) == FILES_PER_HOUR:
            result[hour] = np.sum(grids, axis=0)
        elif len(grids) > FILES_PER_HOUR:
            raise ValueError(
                f"{hour}: {len(grids)} quarter-hour files, expected at most "
                f"{FILES_PER_HOUR} -- duplicate or misaligned archive files"
            )
    return result


def subsampled_flat_indices(size: int, stride: int) -> list[int]:
    """Row-major flat indices of a stride-by-stride subgrid of a `size`-by-
    `size` grid -- deterministic: the same size and stride always produce the
    same node list (TEST04), matching `node_coordinates`' row-major order.
    """
    picked = list(range(0, size, stride))
    return [row * size + col for row in picked for col in picked]


def _daily_zips_in_range(root: Path, start: date, end: date) -> list[Path]:
    """Every zip under `root` whose filename embeds one date in [start, end],
    sorted by that date. Recurses through whatever layout the archive uses,
    the same way `src.canonical.grid_inputs.find_archive_dates` does.
    """
    found = []
    for path in root.rglob("*.zip"):
        match = _DATE_RE.search(path.stem)
        if not match:
            continue
        try:
            day = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if start <= day <= end:
            found.append((day, path))
    return [path for _, path in sorted(found)]


def build_radar_grid_data(
    root: Path, start: date, end: date, subsample_stride: int
) -> GridData:
    """Read every daily zip under `root` in [start, end], sum complete hours,
    subsample to the stride grid, and return a radar `GridData`.
    """
    zip_paths = _daily_zips_in_range(root, start, end)
    if not zip_paths:
        raise ValueError(
            f"no radar archive zips found under {root} in {start} to {end}"
        )

    hours: dict[pd.Timestamp, np.ndarray] = {}
    grid_def = None
    for zip_path in zip_paths:
        if grid_def is None:
            with zipfile.ZipFile(zip_path) as zf:
                member = next(n for n in zf.namelist() if n.endswith(".nc"))
                with tempfile.TemporaryDirectory() as tmp:
                    grid_def = read_grid(Path(zf.extract(member, tmp)))
        hours.update(hourly_grids(read_daily_zip(zip_path)))

    if not hours:
        raise ValueError(f"no complete hours found under {root} in {start} to {end}")

    size = next(iter(hours.values())).shape[0]
    if any(grid.shape != (size, size) for grid in hours.values()):
        raise ValueError(f"radar grids under {root} are not all {size}x{size}")
    indices = subsampled_flat_indices(size, subsample_stride)
    coordinates = pd.DataFrame(
        [(node_id, lat, lon) for node_id, (lat, lon) in node_coordinates(grid_def)],
        columns=["node_id", "latitude", "longitude"],
    ).iloc[indices].reset_index(drop=True)
    coordinates["node_id"] = np.arange(len(coordinates))

    ordered_hours = sorted(hours)
    values = np.stack(
        [hours[hour].reshape(-1)[indices] for hour in ordered_hours]
    )[:, :, None]
    return GridData("radar", pd.DatetimeIndex(ordered_hours), values, coordinates)
