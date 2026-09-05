"""Component 01, gauge source: download BoM tip and daily-total series to disk.

Stores every API response exactly as returned, gzipped, in the layout from
docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md:

    <output_dir>/gauge/
      stations.json                         getStationList response, unchanged
      series.csv                            station_id, ts_id, series_name, parameter
      tips/<station_id>/<YYYY-MM>.json.gz   getTimeseriesValues response, unchanged
      daily/<station_id>/<YYYY>.json.gz     getTimeseriesValues response, unchanged

No aggregation, timezone conversion, rounding or gap-filling happens here. A file
that already exists is skipped; to refetch, delete it first.
"""

import argparse
import calendar
import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

from src.ingestion.config import load_config
from src.ingestion.gauge.kiwis import KiwisClient


def _log(msg: str) -> None:
    """Print and flush, so `tail -f` on a redirected log shows progress live."""
    print(msg, flush=True)


def find_series(client: KiwisClient, station_id: str, names: dict) -> dict:
    """Return {"tips": ts_id, "daily": ts_id} for one station.
    Fails with a concrete message if either named series is absent."""
    _, rows = client.series_list(station_id=station_id)
    by_name = {r["ts_name"]: r for r in rows}
    found, missing = {}, []
    for key, name in names.items():
        if name in by_name:
            found[key] = by_name[name]["ts_id"]
        else:
            missing.append(name)
    if missing:
        raise LookupError(
            f"station {station_id}: no series named {missing}; has {sorted(by_name)}"
        )
    return found


def month_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    """Calendar months touching [start, end], as (YYYY-MM, first_day, last_day)."""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        last = date(y, m, calendar.monthrange(y, m)[1])
        out.append((f"{y:04d}-{m:02d}", max(first, start), min(last, end)))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def year_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    """Calendar years touching [start, end], as (YYYY, first_day, last_day)."""
    return [
        (f"{y:04d}", max(date(y, 1, 1), start), min(date(y, 12, 31), end))
        for y in range(start.year, end.year + 1)
    ]


def write_raw(path: Path, raw: bytes) -> str:
    """Gzip the bytes to path, never overwriting. Returns the sha256 of the gzip file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to refetch")
    path.parent.mkdir(parents=True, exist_ok=True)
    # gzip embeds a timestamp and the file name in its header; zero both so that
    # identical response bytes always give an identical file and sha256.
    with (
        path.open("wb") as out,
        gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as fh,
    ):
        fh.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_values(client: KiwisClient, ts_id: str, start: date, end: date, path: Path):
    """Download one series window to path. Returns (n_records, sha256)."""
    raw, n = client.values(ts_id, start.isoformat(), end.isoformat())
    return n, write_raw(path, raw)


# ---------------------------------------------------------------------------
# The run: every station in the box, every window, with resume and an index.
# ---------------------------------------------------------------------------

INDEX_COLUMNS = [
    "source",
    "station_id",
    "path",
    "start",
    "end",
    "n_records",
    "status",
    "error",
    "retrieved_at",
    "sha256",
]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_records(path: Path) -> int:
    with gzip.open(path) as fh:
        return len(json.load(fh)[0]["data"])


def load_or_fetch_stations(
    client: KiwisClient, bounds: dict, gauge_dir: Path
) -> list[dict]:
    """stations.json is fetched once. On a rerun it is read back so the station set
    cannot drift between runs; delete it to start over."""
    path = gauge_dir / "stations.json"
    if path.exists():
        with path.open("rb") as fh:
            payload = json.load(fh)
        header, *rows = payload
        return [dict(zip(header, r)) for r in rows]
    raw, rows = client.station_list(bounds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return rows


def load_or_fetch_series(
    client: KiwisClient, stations: list[dict], names: dict, gauge_dir: Path
) -> tuple[dict, dict]:
    """series.csv maps station_id -> the two ts_ids. Reused on rerun like stations.json.
    Returns ({station_id: {"tips": ts_id, "daily": ts_id}}, {station_id: error})."""
    path = gauge_dir / "series.csv"
    found, missing = {}, {}
    if path.exists():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                key = next(k for k, v in names.items() if v == row["series_name"])
                found.setdefault(row["station_id"], {})[key] = row["ts_id"]
        return found, missing

    rows = []
    for i, st in enumerate(stations, 1):
        sid = st["station_id"]
        try:
            ids = find_series(client, sid, names)
        except (LookupError, RuntimeError) as e:
            missing[sid] = str(e)
            _log(f"  [{i}/{len(stations)}] {sid} {st['station_name']}: NO SERIES ({e})")
            continue
        found[sid] = ids
        for key, ts_id in ids.items():
            rows.append(
                {
                    "station_id": sid,
                    "ts_id": ts_id,
                    "series_name": names[key],
                    "parameter": "Rainfall",
                }
            )
        _log(
            f"  [{i}/{len(stations)}] {sid} {st['station_name']}: tips={ids['tips']} daily={ids['daily']}"
        )
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["station_id", "ts_id", "series_name", "parameter"]
        )
        w.writeheader()
        w.writerows(rows)
    return found, missing


def fetch_window(client, source, sid, ts_id, label, start, end, path, raw_dir) -> dict:
    """One file: skip if present, else download. Always returns an index row."""
    row = {
        "source": source,
        "station_id": sid,
        "path": str(path.relative_to(raw_dir)),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_records": "",
        "status": "",
        "error": "",
        "retrieved_at": "",
        "sha256": "",
    }
    if path.exists():
        row.update(
            n_records=_count_records(path),
            status="ok",
            retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            sha256=_sha256(path),
        )
        return row
    try:
        n, sha = fetch_values(client, ts_id, start, end, path)
        row.update(n_records=n, status="ok", retrieved_at=_now(), sha256=sha)
    except (RuntimeError, OSError) as e:
        row.update(status="failed", error=str(e)[:200], retrieved_at=_now())
    return row


def run(
    cfg: dict,
    output_dir: Path | None = None,
    limit: int | None = None,
    client: KiwisClient | None = None,
) -> int:
    """Fetch every gauge file for the config. Returns the number of failures.
    `client` is injectable so tests can run offline against a fake."""
    g = cfg["gauge"]
    raw_dir = Path(output_dir or cfg["output_dir"])
    gauge_dir = raw_dir / "gauge"
    client = client or KiwisClient(**g["api"])
    t0 = time.time()

    _log(
        f"config: {cfg['_path']}  region: {cfg['region']}  period: {cfg['start']} to {cfg['end']}"
    )
    stations = load_or_fetch_stations(client, g["bounds"], gauge_dir)
    if limit:
        stations = stations[:limit]
    _log(
        f"stations in box: {len(stations)}"
        + (f" (limited to {limit})" if limit else "")
    )

    _log("series lookup:")
    series, no_series = load_or_fetch_series(client, stations, g["series"], gauge_dir)

    months = month_windows(cfg["start"], cfg["end"])
    years = year_windows(cfg["start"], cfg["end"])
    index, skipped, fetched = [], 0, 0
    for i, st in enumerate(stations, 1):
        sid = st["station_id"]
        if sid not in series:
            continue
        before = len([r for r in index if r["retrieved_at"]])
        for label, start, end in months:
            path = gauge_dir / "tips" / sid / f"{label}.json.gz"
            existed = path.exists()
            index.append(
                fetch_window(
                    client,
                    "gauge_tips",
                    sid,
                    series[sid]["tips"],
                    label,
                    start,
                    end,
                    path,
                    raw_dir,
                )
            )
            skipped += existed
            fetched += not existed
        for label, start, end in years:
            path = gauge_dir / "daily" / sid / f"{label}.json.gz"
            existed = path.exists()
            index.append(
                fetch_window(
                    client,
                    "gauge_daily",
                    sid,
                    series[sid]["daily"],
                    label,
                    start,
                    end,
                    path,
                    raw_dir,
                )
            )
            skipped += existed
            fetched += not existed
        failed_here = sum(r["status"] == "failed" for r in index[before:])
        _log(
            f"  [{i}/{len(stations)}] {sid} {st['station_name']}: {len(months) + len(years)} files, {failed_here} failed"
        )

    failures = [r for r in index if r["status"] == "failed"]
    write_index(raw_dir / "index.csv", index)
    write_report(
        raw_dir / "report.md",
        cfg,
        stations,
        no_series,
        index,
        failures,
        skipped,
        fetched,
        time.time() - t0,
    )
    _log(
        f"\nfiles: {len(index)} indexed, {fetched} fetched this run, {skipped} already on disk, {len(failures)} failed"
    )
    _log(f"stations with no usable series: {len(no_series)}")
    _log(f"wrote {raw_dir / 'index.csv'} and {raw_dir / 'report.md'}")
    return len(failures) + len(no_series)


def write_index(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_report(
    path, cfg, stations, no_series, index, failures, skipped, fetched, seconds
) -> None:
    ok = [r for r in index if r["status"] == "ok"]
    lines = [
        f"# Gauge ingestion report: {cfg['region']}",
        "",
        f"- config: `{cfg['_path']}`",
        f"- period: {cfg['start']} to {cfg['end']} inclusive",
        f"- bounding box: {cfg['gauge']['bounds']}",
        f"- series: tips `{cfg['gauge']['series']['tips']}`, daily `{cfg['gauge']['series']['daily']}`",
        f"- run finished: {_now()}, {seconds / 60:.1f} min",
        "",
        "## Counts",
        "",
        f"- stations in box: {len(stations)}",
        f"- stations with both series: {len(stations) - len(no_series)}",
        f"- stations with no usable series: {len(no_series)}",
        f"- files expected: {len(index)}, ok: {len(ok)}, failed: {len(failures)}",
        f"- fetched this run: {fetched}, already on disk: {skipped}",
        f"- tip records on disk: {sum(int(r['n_records']) for r in ok if r['source'] == 'gauge_tips'):,}",
        f"- daily records on disk: {sum(int(r['n_records']) for r in ok if r['source'] == 'gauge_daily'):,}",
        "",
        "## Stations with no usable series (excluded, with cause)",
        "",
    ]
    lines += [f"- {sid}: {why}" for sid, why in no_series.items()] or ["- none"]
    lines += [
        "",
        "## Failed files (rerun the command to retry; only missing files are fetched)",
        "",
    ]
    lines += [f"- {r['path']}: {r['error']}" for r in failures] or ["- none"]
    lines += [
        "",
        "## Rerun",
        "",
        "```",
        f"uv run python -m src.ingestion.gauge.fetch {cfg['_path']}",
        "```",
        "",
        "Files already on disk are skipped. Delete a file to refetch it; delete",
        "`gauge/stations.json` or `gauge/series.csv` to redo the station or series lookup.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="ingestion YAML, e.g. config/ingestion/sydney.yaml")
    ap.add_argument(
        "--output-dir", help="override output_dir from the config (testing)"
    )
    ap.add_argument("--limit", type=int, help="only the first N stations (testing)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    failures = run(cfg, Path(args.output_dir) if args.output_dir else None, args.limit)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
