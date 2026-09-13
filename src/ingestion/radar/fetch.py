"""Component 01, radar source: download BoM Rainfields 3 "Merged Gauge/Radar
Accumulation (15 min)" (product `prcp-m15`) from the NCI THREDDS `rq0` archive.

Stores every daily zip exactly as the server returns it, in the layout from
docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md:

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

from src.ingestion.common import now, sha256_of, write_index
from src.ingestion.config import load_config
from src.ingestion.radar.coords import write_node_coords
from src.ingestion.radar.nci import NciClient


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
        member = next(n for n in zf.namelist() if n.endswith(".nc"))
        return Path(zf.extract(member, dest_dir))


def fetch_day(
    client: NciClient, site_id: str, product: str, day: date, path: Path, raw_dir: Path
) -> dict:
    """One day's zip: skip if present, else download. Always returns an index row."""
    row = {
        "source": "radar_precip",
        "station_id": "",
        "path": str(path.relative_to(raw_dir)),
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
    except (RuntimeError, OSError, zipfile.BadZipFile) as e:
        row.update(status="failed", error=str(e)[:200], retrieved_at=now())
    return row


def run(
    cfg: dict,
    output_dir: Path | None = None,
    limit: int | None = None,
    client: NciClient | None = None,
) -> int:
    """Fetch every radar day for the config. Returns the number of failures.
    `client` is injectable so tests can run offline against a fake."""
    r = cfg.get("radar") or {}
    if not r:
        _log("radar: not configured (radar section is empty), nothing to do")
        return 0

    raw_dir = Path(output_dir or cfg["output_dir"])
    radar_dir = raw_dir / "radar"
    client = client or NciClient(**r["api"])
    t0 = time.time()

    _log(
        f"config: {cfg['_path']}  site: {r['site_id']} ({r['product']})  "
        f"period: {cfg['start']} to {cfg['end']}"
    )
    days = day_windows(cfg["start"], cfg["end"])
    if limit:
        days = days[:limit]

    index, skipped, fetched = [], 0, 0
    ok_paths = []
    for i, day in enumerate(days, 1):
        path = radar_dir / f"{day.year}" / f"{day.isoformat()}.{r['product']}.zip"
        existed = path.exists()
        row = fetch_day(client, r["site_id"], r["product"], day, path, raw_dir)
        index.append(row)
        skipped += existed
        fetched += not existed
        if row["status"] == "ok":
            ok_paths.append(path)
        if i % 30 == 0 or i == len(days):
            _log(f"  [{i}/{len(days)}] {day.isoformat()}: {row['status']}")

    failures = [row for row in index if row["status"] == "failed"]
    coords_written = _write_coords_from_samples(ok_paths, radar_dir / "radar_node_coords.csv")

    write_index(raw_dir / "index.csv", index)
    write_report(
        raw_dir / "report.md", cfg, r, index, failures, skipped, fetched,
        coords_written, time.time() - t0,
    )
    _log(
        f"\nfiles: {len(index)} indexed, {fetched} fetched this run, "
        f"{skipped} already on disk, {len(failures)} failed"
    )
    _log(f"wrote {raw_dir / 'index.csv'} and {raw_dir / 'report.md'}")
    return len(failures)


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
    path, cfg, r, index, failures, skipped, fetched, coords_written, seconds
) -> None:
    """Write `report.md`: a human summary of the run — counts, which days failed
    and why, how many coordinate nodes were written this run, and the exact
    command to rerun it."""
    ok = [row for row in index if row["status"] == "ok"]
    lines = [
        f"# Radar ingestion report: {cfg['region']}",
        "",
        f"- config: `{cfg['_path']}`",
        f"- period: {cfg['start']} to {cfg['end']} inclusive",
        f'- source: NCI THREDDS `rq0/rainfields3`, site `{r["site_id"]}`, product `{r["product"]}` ("Merged Gauge/Radar Accumulation, 15 min")',
        f"- run finished: {now()}, {seconds / 60:.1f} min",
        "",
        "## Counts",
        "",
        f"- days expected: {len(index)}, ok: {len(ok)}, failed: {len(failures)}",
        f"- fetched this run: {fetched}, already on disk: {skipped}",
        f"- radar_node_coords.csv nodes written this run: {coords_written} (0 means it already existed or nothing downloaded successfully yet)",
        "",
        "## Failed days (rerun the command to retry; only missing files are fetched)",
        "",
    ]
    lines += [f"- {row['path']}: {row['error']}" for row in failures] or ["- none"]
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
