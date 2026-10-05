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
    """One 15-minute file's precipitation grid (mm, NaN for missing cells),
    the UTC start and end of the window it covers, and the BoM station_no
    values its `contributing_gauges` attribute lists (D13.4: the size of
    D09's leakage caveat)."""

    start: pd.Timestamp
    precipitation: np.ndarray
    valid_time: pd.Timestamp | None = None
    contributing_gauges: frozenset[int] = frozenset()


def read_quarter_hour(nc_path: Path) -> QuarterHour:
    """Decode one 15-minute file: CF-scaled precipitation in mm (NaN where
    the file's own `_FillValue` marks a cell as no-data), its start/end time,
    and its contributing gauges.
    """
    with xr.open_dataset(nc_path) as ds:
        start = pd.Timestamp(ds["start_time"].values.item(), tz="UTC")
        valid_time = pd.Timestamp(ds["valid_time"].values.item(), tz="UTC")
        precipitation = ds["precipitation"].values.astype(np.float64)
        contributing_gauges = frozenset(
            int(v) for v in ds.attrs.get("contributing_gauges", [])
        )
    return QuarterHour(
        start=start,
        precipitation=precipitation,
        valid_time=valid_time,
        contributing_gauges=contributing_gauges,
    )


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


def hourly_contributing_gauges(
    quarters: list[QuarterHour],
) -> dict[pd.Timestamp, frozenset[int]]:
    """Union of `contributing_gauges` across each complete hour's four files
    (D13.4). Uses the same floor-hour bucketing and four-file completeness
    rule as `hourly_grids`, applied to the same `quarters` list, so this
    always agrees with `hourly_grids` on which hours are complete.
    """
    by_hour: dict[pd.Timestamp, list[frozenset[int]]] = {}
    for quarter in quarters:
        by_hour.setdefault(quarter.start.floor("h"), []).append(
            quarter.contributing_gauges
        )
    return {
        hour: frozenset().union(*gauge_sets)
        for hour, gauge_sets in by_hour.items()
        if len(gauge_sets) == FILES_PER_HOUR
    }


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
    gauges_by_hour: dict[pd.Timestamp, frozenset[int]] = {}
    native_time_step_minutes = None
    grid_def = None
    for zip_path in zip_paths:
        if grid_def is None:
            with zipfile.ZipFile(zip_path) as zf:
                member = next(n for n in zf.namelist() if n.endswith(".nc"))
                with tempfile.TemporaryDirectory() as tmp:
                    grid_def = read_grid(Path(zf.extract(member, tmp)))
        quarters = read_daily_zip(zip_path)
        if native_time_step_minutes is None and quarters:
            delta = quarters[0].valid_time - quarters[0].start
            native_time_step_minutes = round(delta.total_seconds() / 60)
        hours.update(hourly_grids(quarters))
        gauges_by_hour.update(hourly_contributing_gauges(quarters))

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
    rain = np.stack([hours[hour].reshape(-1)[indices] for hour in ordered_hours])
    # The model cannot accept a null: a cell missing in one of its four
    # contributing files (NaN, per hourly_grids) becomes rain_rate 0 with
    # valid False, the same [value, valid] shape satellite already uses,
    # rather than losing the missing/dry distinction or leaking a NaN into
    # the table. An hour's own genuine zero (all four files agreed, dry)
    # stays 0 with valid True.
    valid = np.isfinite(rain)
    values = np.stack([np.where(valid, rain, 0.0), valid.astype(np.float64)], axis=-1)
    provenance = {
        # D13.4: taken from the files' own attributes, not assumed -- see
        # docs/requirements/D13_SOURCE_GRIDS_REQUIREMENTS.md, FR04.
        "coordinate_system": grid_def["proj_attrs"]["grid_mapping_name"],
        "native_time_step_minutes": native_time_step_minutes,
        "contributing_gauges_by_hour": {
            hour: gauges_by_hour[hour]
            for hour in ordered_hours
            if hour in gauges_by_hour
        },
    }
    return GridData(
        "radar", pd.DatetimeIndex(ordered_hours), values, coordinates, provenance
    )
