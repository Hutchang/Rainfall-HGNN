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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.canonical.grids import GridData, single_declaration
from src.ingestion.radar.coords import (
    assert_consistent_grid,
    node_coordinates,
    read_grid,
)

FILES_PER_HOUR = 4
_DATE_RE = re.compile(r"(\d{8})")


@dataclass
class QuarterHour:
    """One 15-minute file's precipitation grid (mm, NaN for missing cells),
    the UTC start and end of the window it covers, and the BoM station_no
    values its `contributing_gauges` attribute lists (D13.4: the size of
    D09's leakage caveat). `title`, `units` and `site` are the file's own
    declarations, for the report (FR04); None where the file has none."""

    start: pd.Timestamp
    precipitation: np.ndarray
    valid_time: pd.Timestamp | None = None
    contributing_gauges: frozenset[int] = frozenset()
    title: str | None = None
    units: str | None = None
    site: str | None = None


def read_quarter_hour(nc_path: Path) -> QuarterHour:
    """Decode one 15-minute file: CF-scaled precipitation in mm (NaN where
    the file's own `_FillValue` marks a cell as no-data), its start/end time,
    its contributing gauges and its own product, units and site attributes.
    """
    with xr.open_dataset(nc_path) as ds:
        start = pd.Timestamp(ds["start_time"].values.item(), tz="UTC")
        valid_time = pd.Timestamp(ds["valid_time"].values.item(), tz="UTC")
        precipitation = ds["precipitation"].values.astype(np.float64)
        contributing_gauges = frozenset(
            int(v) for v in ds.attrs.get("contributing_gauges", [])
        )
        site = ds.attrs.get("station_id")
        if site is not None and "station_name" in ds.attrs:
            site = f"{site} ({ds.attrs['station_name']})"
        title = ds.attrs.get("title")
        units = ds["precipitation"].attrs.get("units")
    return QuarterHour(
        start=start,
        precipitation=precipitation,
        valid_time=valid_time,
        contributing_gauges=contributing_gauges,
        title=title,
        units=units,
        site=site,
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


def _first_member_grid(zip_path: Path) -> dict:
    """The grid definition of one daily zip's first NetCDF member."""
    with zipfile.ZipFile(zip_path) as zf:
        member = min(name for name in zf.namelist() if name.endswith(".nc"))
        with tempfile.TemporaryDirectory() as tmp:
            return read_grid(Path(zf.extract(member, tmp)))


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


def _zip_day(path: Path) -> date | None:
    """The date a daily zip's filename embeds, or None if it embeds none."""
    match = _DATE_RE.search(path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _daily_zips_in_range(root: Path, start: date, end: date) -> list[Path]:
    """Every `prcp-m15` zip under `root` whose filename embeds one date in
    [start, end], sorted by that date. Recurses through whatever layout the
    archive uses, the same way `src.canonical.grid_inputs.find_archive_dates`
    does. A site's year folders also hold `gndrefl` and `prcp-crate` zips,
    which are other products and are never read.
    """
    found = []
    for path in root.rglob("*.prcp-m15.zip"):
        day = _zip_day(path)
        if day is not None and start <= day <= end:
            found.append((day, path))
    return [path for _, path in sorted(found)]


def build_radar_grid_data(
    root: Path, start: date, end: date, subsample_stride: int
) -> GridData:
    """Read every daily zip under `root` for [start, end], sum complete hours,
    subsample to the stride grid, and return a radar `GridData`.

    The archive zips each file under the day its window *ends* (see the
    module docstring), so the 23:00 hour's last file, 23:45-00:00, is in the
    next day's zip. Hours are therefore summed across zips, and the zip for
    the day after `end` is read too, so the period's last hour is complete.
    """
    zip_paths = _daily_zips_in_range(root, start, end + timedelta(days=1))
    if not zip_paths:
        raise ValueError(
            f"no radar archive zips found under {root} in {start} to {end}"
        )
    period_start = pd.Timestamp(start, tz="UTC")
    period_end = pd.Timestamp(end + timedelta(days=1), tz="UTC")

    hours: dict[pd.Timestamp, np.ndarray] = {}
    gauges_by_hour: dict[pd.Timestamp, frozenset[int]] = {}
    pending: dict[pd.Timestamp, list[QuarterHour]] = {}

    def close(closing: list[pd.Timestamp]) -> None:
        """Sum hours no later zip can add to; drop those outside the period
        (the first zip's 23:45 file of the day before `start`, and the extra
        zip's own hours after `end`)."""
        quarters = [
            quarter
            for hour in closing
            for quarter in pending.pop(hour)
            if period_start <= hour < period_end
        ]
        hours.update(hourly_grids(quarters))
        gauges_by_hour.update(hourly_contributing_gauges(quarters))

    # FR04: the report's product, units, site and correction lines come from
    # every file read, not from grids.yaml.
    declared = {"title": set(), "precipitation units": set(), "station_id": set()}
    files_read = files_listing_gauges = 0
    native_time_step_minutes = None
    grid_def = None
    for zip_path in zip_paths:
        quarters = read_daily_zip(zip_path)
        for quarter in quarters:
            declared["title"].add(quarter.title)
            declared["precipitation units"].add(quarter.units)
            declared["station_id"].add(quarter.site)
            files_listing_gauges += bool(quarter.contributing_gauges)
        files_read += len(quarters)
        for name, values in declared.items():
            single_declaration(values, name, root)
        # Each radar site has its own grid (an Albers projection centred on
        # the radar), so a zip whose grid differs is another site's, and its
        # rain must not be stored against the first site's cell positions.
        zip_grid = _first_member_grid(zip_path)
        if grid_def is None:
            grid_def = zip_grid
        try:
            assert_consistent_grid([grid_def, zip_grid])
        except ValueError as error:
            raise ValueError(
                f"{zip_path}: {error}; every zip under {root} must come from "
                "the same radar site -- point the root at one site's folder"
            ) from error
        if native_time_step_minutes is None and quarters:
            delta = quarters[0].valid_time - quarters[0].start
            native_time_step_minutes = round(delta.total_seconds() / 60)
        for quarter in quarters:
            pending.setdefault(quarter.start.floor("h"), []).append(quarter)
        # An hour's four files end 15 to 60 minutes after it starts, so once
        # the zip for the day its last file ends on has been read, no later
        # zip can hold any of them.
        day = _zip_day(zip_path)
        one_hour = pd.Timedelta(hours=1)
        close([hour for hour in pending if (hour + one_hour).date() <= day])
    close(list(pending))

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
    # One value per cell-hour: the paper's gridded nodes carry the cell value
    # only, with no validity flag (docs/source_paper.md, "Node features"). A
    # cell missing in any of its four files stays NaN, is counted in the
    # report and is refused at the graph boundary; adding a flag is ISS-36.
    values = rain[:, :, None]
    provenance = {
        # D13.4: taken from the files' own attributes, not assumed -- see
        # docs/requirements/D13_SOURCE_GRIDS_REQUIREMENTS.md, FR04.
        "coordinate_system": grid_def["proj_attrs"]["grid_mapping_name"],
        "native_time_step_minutes": native_time_step_minutes,
        "product": single_declaration(declared["title"], "title", root),
        "units": single_declaration(
            declared["precipitation units"], "precipitation units", root
        ),
        "site": single_declaration(declared["station_id"], "station_id", root),
        "gauge_correction": (
            f"contributing_gauges listed in {files_listing_gauges} of "
            f"{files_read} files read"
        ),
        "hourly_rule": "sum of four 15-minute files",
        "subsample_rule": (
            f"rows and columns 0, {subsample_stride}, {2 * subsample_stride}, ... "
            f"of the native {size} x {size} grid (stride {subsample_stride})"
        ),
        "contributing_gauges_by_hour": {
            hour: gauges_by_hour[hour]
            for hour in ordered_hours
            if hour in gauges_by_hour
        },
    }
    return GridData(
        "radar", pd.DatetimeIndex(ordered_hours), values, coordinates, provenance
    )
