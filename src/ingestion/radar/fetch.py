"""Component 01, radar source: download BoM Rainfields 3 "Merged Gauge/Radar
Accumulation (15 min)" (product `prcp-m15`) from the NCI THREDDS `rq0` archive.

Stores every daily zip exactly as the server returns it, in the layout from
docs/archive/sprint1/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md:

    <output_dir>/radar/
      <YYYY>/<YYYY-MM-DD>.<product>.zip   one day's ~15-minute NetCDF files, unchanged
      radar_node_coords.csv               node_id, latitude, longitude (see coords.py)

No aggregation, unit conversion or gap-filling happens here — each file already covers
one 15-minute accumulation, not an hour; Component 02 sums four together per hour, the
same way it sums gauge tips. A file that already exists is skipped; to refetch, delete
it first. There is no separate state file, matching the gauge fetcher's resume model.
"""

import argparse
import sys
import tempfile
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.ingestion.common import (
    index_path,
    now,
    report_path,
    sha256_of,
    write_index,
)
from src.ingestion.config import load_config
from src.ingestion.radar.coords import write_node_coords
from src.ingestion.radar.nci import DayNotInArchiveError, NciClient

# The index is rewritten every FLUSH_EVERY days. A multi-hour run that is killed
# outright then still leaves bookkeeping for everything up to the last flush,
# instead of nothing at all.
FLUSH_EVERY = 30


def _log(msg: str) -> None:
    """Print one progress line and flush immediately, so a redirected log (e.g.
    `uv run ... | tee run.log`) shows each day's status as it happens rather than
    only once the whole run finishes and the output buffer is flushed."""
    print(msg, flush=True)


def day_windows(start: date, end: date) -> list[date]:
    """Every calendar day touching [start, end], inclusive."""
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _count_nc_members(path: Path) -> int:
    """Count of `.nc` files inside a day's zip, for the index's `n_records` column.
    The server also packs a `_file_list` manifest entry into every zip; this counts
    only the real data files, not that bookkeeping entry."""
    with zipfile.ZipFile(path) as zf:
        return sum(1 for n in zf.namelist() if n.endswith(".nc"))


def _first_nc_member(zip_path: Path, dest_dir: Path) -> Path:
    """Extract one .nc member from a day's zip, for grid verification only."""
    with zipfile.ZipFile(zip_path) as zf:
        member = next((n for n in zf.namelist() if n.endswith(".nc")), None)
        if member is None:
            raise ValueError(f"{zip_path}: zip holds no .nc member")
        return Path(zf.extract(member, dest_dir))


def fetch_day(
    client: NciClient, site_id: str, product: str, day: date, path: Path, raw_dir: Path
) -> dict:
    """One day's zip: skip if present, else download. Always returns an index row."""
    row = {
        "source": "radar_precip",
        "station_id": "",
        "path": path.relative_to(raw_dir).as_posix(),
        "start": day.isoformat(),
        "end": day.isoformat(),
        "n_records": "",
        "status": "",
        "error": "",
        "retrieved_at": "",
        "sha256": "",
    }
    if path.exists():
        row.update(
            n_records=_count_nc_members(path),
            status="ok",
            retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            sha256=sha256_of(path),
        )
        return row
    try:
        raw = client.daily_zip(site_id, product, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        row.update(
            n_records=_count_nc_members(path),
            status="ok",
            retrieved_at=now(),
            sha256=sha256_of(path),
        )
    except DayNotInArchiveError as e:
        row.update(status="skipped", error=str(e)[:200], retrieved_at=now())
    except (RuntimeError, OSError, zipfile.BadZipFile) as e:
        row.update(status="failed", error=str(e)[:200], retrieved_at=now())
    return row


def run(
    cfg: dict,
    output_dir: Path | None = None,
    limit: int | None = None,
    client: NciClient | None = None,
) -> int:
    """Fetch every radar day for the config. Returns a non-zero count if anything
    failed or the run stopped early. `client` is injectable so tests can run
    offline against a fake.

    Like the gauge fetcher, the index and report are written even when the run is
    interrupted: the zips on disk survive a Ctrl-C, but the index is built in
    memory and would die with the process.
    """
    r = cfg.get("radar") or {}
    if not r:
        _log("radar: not configured (radar section is empty), nothing to do")
        return 0

    raw_dir = Path(output_dir or cfg["output_dir"])
    radar_dir = raw_dir / "radar"
    # Only close a client we made; an injected one belongs to the caller.
    owned_client = client is None
    client = client or NciClient(**r["api"])
    t0 = time.time()

    _log(
        f"config: {cfg['_path']}  site: {r['site_id']} ({r['product']})  "
        f"period: {cfg['start']} to {cfg['end']}"
    )
    days = day_windows(cfg["start"], cfg["end"])
    # `is not None`, so that --limit 0 fetches nothing rather than everything.
    if limit is not None:
        days = days[:limit]

    index, on_disk, fetched, done = [], 0, 0, 0
    ok_paths = []
    try:
        for i, day in enumerate(days, 1):
            path = radar_dir / f"{day.year}" / f"{day.isoformat()}.{r['product']}.zip"
            existed = path.exists()
            row = fetch_day(client, r["site_id"], r["product"], day, path, raw_dir)
            index.append(row)
            if existed:
                on_disk += 1
            elif row["status"] == "ok":
                # Only a download that worked counts as fetched; a failed day is
                # reported as failed, not as progress.
                fetched += 1
            # A zip with no .nc member cannot describe the grid, so it is not a
            # candidate for the coordinate sample.
            if row["status"] == "ok" and int(row["n_records"] or 0) > 0:
                ok_paths.append(path)
            done = i
            if i % FLUSH_EVERY == 0 or i == len(days):
                write_index(index_path(raw_dir, "radar"), index)
                _log(f"  [{i}/{len(days)}] {day.isoformat()}: {row['status']}")
    except KeyboardInterrupt:
        _log("")
        _log(f"interrupted after {done} of {len(days)} days")
    finally:
        if owned_client:
            client.close()
        failures = [row for row in index if row["status"] == "failed"]
        absent = [row for row in index if row["status"] == "skipped"]
        # The index is written before anything that can fail, so a run's
        # bookkeeping survives even if the coordinate derivation throws.
        write_index(index_path(raw_dir, "radar"), index)
        try:
            coords_written = _write_coords_from_samples(
                ok_paths, radar_dir / "radar_node_coords.csv"
            )
        except Exception as error:
            coords_written = 0
            _log(f"radar_node_coords.csv not written: {error}")
        write_report(
            report_path(raw_dir, "radar"),
            cfg,
            r,
            index,
            failures,
            absent,
            on_disk,
            fetched,
            coords_written,
            time.time() - t0,
            done,
            len(days),
        )
        _log("")
        _log(
            f"files: {len(index)} indexed, {fetched} fetched this run, "
            f"{on_disk} already on disk, {len(absent)} not in archive, "
            f"{len(failures)} failed"
        )
        _log(
            f"wrote {index_path(raw_dir, 'radar')} and {report_path(raw_dir, 'radar')}"
        )
    # A run that stopped early is not a success even with nothing failed.
    return len(failures) + (len(days) - done)


def _write_coords_from_samples(ok_paths: list[Path], out_csv: Path) -> int:
    """Derive radar_node_coords.csv from a few successfully-downloaded days, cross-
    checking that they describe the same grid before trusting any of them. Returns
    the node count written, or 0 if no file was available to sample."""
    if not ok_paths or out_csv.exists():
        return 0
    sample = [ok_paths[0], ok_paths[len(ok_paths) // 2], ok_paths[-1]]
    sample = list(dict.fromkeys(sample))  # de-dupe while keeping order
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        nc_paths = [_first_nc_member(p, tmp_dir) for p in sample]
        return write_node_coords(nc_paths, out_csv)


def write_report(
    path,
    cfg,
    r,
    index,
    failures,
    absent,
    on_disk,
    fetched,
    coords_written,
    seconds,
    done,
    total,
) -> None:
    """Write the run report: counts, which days failed and why, which days the
    archive does not hold, how many coordinate nodes were written, and the exact
    command to rerun. `done` short of `total` means the run stopped early, so
    every count below describes only part of the config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [row for row in index if row["status"] == "ok"]
    lines = [
        f"# Radar ingestion report: {cfg['region']}",
        "",
        f"- config: `{cfg['_path']}`",
        f"- period: {cfg['start']} to {cfg['end']} inclusive",
        f"- source: NCI THREDDS `rq0/rainfields3`, site `{r['site_id']}`, "
        f"product `{r['product']}` "
        '("Merged Gauge/Radar Accumulation, 15 min")',
        f"- run finished: {now()}, {seconds / 60:.1f} min",
        "",
    ]
    if done < total:
        lines += [
            f"**This run stopped after {done} of {total} days.** Every count below "
            "describes only that part. Rerun to continue; days already on disk are "
            "skipped.",
            "",
        ]
    lines += [
        "## Counts",
        "",
        f"- days expected: {len(index)}, ok: {len(ok)}, "
        f"not in archive: {len(absent)}, failed: {len(failures)}",
        f"- fetched this run: {fetched}, already on disk: {on_disk}",
        f"- radar_node_coords.csv nodes written this run: {coords_written} "
        "(0 means it already existed, nothing downloaded successfully yet, or the "
        "grid could not be read)",
        "",
        "## Failed days (rerun to retry; only missing files are fetched)",
        "",
    ]
    lines += [f"- {row['path']}: {row['error']}" for row in failures] or ["- none"]
    lines += [
        "",
        "## Days not in the archive (a rerun will not change these)",
        "",
        "Recorded with status `skipped`, so Component 02 reads past them rather than",
        "mistaking them for a failed download.",
        "",
    ]
    lines += [f"- {row['start']}: {row['error']}" for row in absent] or ["- none"]
    lines += [
        "",
        "## Rerun",
        "",
        "```",
        f"uv run python -m src.ingestion.radar.fetch {cfg['_path']}",
        "```",
        "",
        "Files already on disk are skipped. Delete a day's zip to refetch it; delete",
        "`radar/radar_node_coords.csv` to regenerate it from freshly-downloaded days.",
        "",
    ]
    path.write_text("\n".join(lines))


DEFAULT_SAMPLE_DAYS = 5
DEFAULT_SAMPLE_DIR = "local/radar-sample"


def main() -> None:
    """Parse the CLI arguments, load the config, run the fetch, and exit non-zero
    if anything failed. The `python -m src.ingestion.radar.fetch ...` entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="ingestion YAML, e.g. config/ingestion/sydney.yaml")
    ap.add_argument(
        "--output-dir", help="override output_dir from the config (testing)"
    )
    ap.add_argument("--limit", type=int, help="only the first N days (testing)")
    ap.add_argument(
        "--sample",
        nargs="?",
        type=int,
        const=DEFAULT_SAMPLE_DAYS,
        default=None,
        metavar="N",
        help=(
            "quick local run: fetch only the first N days (default "
            f"{DEFAULT_SAMPLE_DAYS}) into '{DEFAULT_SAMPLE_DIR}/' instead of the "
            "real output_dir, so you can try the fetcher without committing to the "
            "full 2021-2022 download. --limit/--output-dir, if also given, win."
        ),
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    limit = args.limit if args.limit is not None else args.sample
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.sample is not None:
        output_dir = Path(DEFAULT_SAMPLE_DIR)
    else:
        output_dir = None

    if args.sample is not None:
        _log(f"--sample: writing to {output_dir}/, limited to {limit} day(s)")
    failures = run(cfg, output_dir, limit)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
