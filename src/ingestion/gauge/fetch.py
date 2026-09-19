"""Component 01, gauge source: download BoM tip and daily-total series to disk.

Stores every API response exactly as returned, gzipped, in the layout from
docs/archive/sprint1/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md:

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
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

from src.ingestion.common import index_path, now, report_path, sha256_of, write_index
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
    return sha256_of(path)


def fetch_values(client: KiwisClient, ts_id: str, start: date, end: date, path: Path):
    """Download one series window to path. Returns (n_records, sha256)."""
    raw, n = client.values(ts_id, start.isoformat(), end.isoformat())
    return n, write_raw(path, raw)


# ---------------------------------------------------------------------------
# The run: every station in the box, every window, with resume and an index.
# ---------------------------------------------------------------------------

# The index is rewritten every FLUSH_EVERY stations. A multi-hour run that is
# killed outright then still leaves bookkeeping for everything up to the last
# flush, instead of nothing at all.
FLUSH_EVERY = 25


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


SERIES_COLUMNS = ["station_id", "ts_id", "series_name", "parameter"]


def read_series_csv(path: Path, names: dict) -> dict:
    """Series ids cached by an earlier run, as {station_id: {"tips": ts_id, ...}}.

    Only stations with every series this config asks for are returned. A station
    with some series present is treated as not yet looked up, so a run that stopped
    part-way through writing the file cannot be mistaken for a completed lookup.
    Series names the config no longer asks for are ignored rather than raising.
    """
    if not path.exists():
        return {}
    by_name = {v: k for k, v in names.items()}
    cached: dict[str, dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = by_name.get(row["series_name"])
            if key is not None:
                cached.setdefault(row["station_id"], {})[key] = row["ts_id"]
    return {sid: ids for sid, ids in cached.items() if set(ids) == set(names)}


def write_series_csv(path: Path, found: dict, names: dict) -> None:
    """Rewrite series.csv from the stations whose lookup succeeded, sorted so the
    file is identical between runs regardless of the order stations were added."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SERIES_COLUMNS)
        w.writeheader()
        w.writerows(
            {
                "station_id": sid,
                "ts_id": found[sid][key],
                "series_name": names[key],
                "parameter": "Rainfall",
            }
            for sid in sorted(found)
            for key in names
        )


def load_or_fetch_series(
    client: KiwisClient, stations: list[dict], names: dict, gauge_dir: Path
) -> tuple[dict, dict]:
    """series.csv maps station_id -> the two ts_ids.
    Returns ({station_id: {"tips": ts_id, "daily": ts_id}}, {station_id: reason}).

    series.csv caches the lookup, but only for the stations where it succeeded, and
    every station missing from it is looked up again. That matters because the
    lookup is one request per station with no per-station file to mark it done: if a
    single transient error were cached as "already looked up", the station would be
    absent from every later run with nothing on disk recording why. Re-deriving
    costs one request per unresolved station and keeps the exclusion list, the
    report and the exit code true on every run.
    """
    path = gauge_dir / "series.csv"
    found = read_series_csv(path, names)
    missing: dict[str, str] = {}
    todo = [st for st in stations if st["station_id"] not in found]
    _log(f"  {len(found)} cached in series.csv, {len(todo)} to look up")

    for i, st in enumerate(todo, 1):
        sid = st["station_id"]
        prefix = f"  [{i}/{len(todo)}] {sid} {st['station_name']}"
        try:
            found[sid] = find_series(client, sid, names)
        except LookupError as e:
            # The station genuinely lacks a series we need. A rerun will not change
            # this; it needs a decision about the station, so say so rather than
            # implying a retry will fix it.
            missing[sid] = f"no required series: {e}"
            _log(f"{prefix}: NO SERIES ({e})")
        except RuntimeError as e:
            # The API call itself failed after KiwisClient exhausted its retries.
            # Transient, so the next run retries it automatically.
            missing[sid] = f"lookup failed, will retry on rerun: {e}"
            _log(f"{prefix}: LOOKUP FAILED, will retry on rerun ({e})")
        else:
            _log(f"{prefix}: tips={found[sid]['tips']} daily={found[sid]['daily']}")

    if todo:
        write_series_csv(path, found, names)
    return found, missing


def skipped_row(source, sid, path, start, end, reason, raw_dir) -> dict:
    """An expected file that was never requested, because the station has no series
    ids. `status` is neither ok nor failed: nothing was downloaded, and retrying this
    one file cannot help. Component 02 reads only `status == "ok"` rows, so these are
    inert downstream while keeping the station and its cause visible in the index."""
    return {
        "source": source,
        "station_id": sid,
        "path": path.relative_to(raw_dir).as_posix(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_records": "",
        "status": "skipped",
        "error": reason[:200],
        "retrieved_at": "",
        "sha256": "",
    }


def fetch_window(client, source, sid, ts_id, label, start, end, path, raw_dir) -> dict:
    """One file: skip if present, else download. Always returns an index row."""
    row = {
        "source": source,
        "station_id": sid,
        "path": path.relative_to(raw_dir).as_posix(),
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
            sha256=sha256_of(path),
        )
        return row
    try:
        n, sha = fetch_values(client, ts_id, start, end, path)
        row.update(n_records=n, status="ok", retrieved_at=now(), sha256=sha)
    except (RuntimeError, OSError) as e:
        row.update(status="failed", error=str(e)[:200], retrieved_at=now())
    return row


def run(
    cfg: dict,
    output_dir: Path | None = None,
    limit: int | None = None,
    client: KiwisClient | None = None,
) -> int:
    """Fetch every gauge file for the config. Returns the number of failures plus
    the stations never reached, so an interrupted run also exits non-zero.
    The index and report are written even when the run is interrupted.
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
    index, skipped, fetched, done = [], 0, 0, 0
    try:
        for i, st in enumerate(stations, 1):
            sid = st["station_id"]
            if sid not in series:
                # No ts_ids, so nothing for this station can be requested. Write a row
                # per expected file anyway, with the cause, so the station cannot drop
                # out of the index between runs and Component 02 can see the whole
                # expected set rather than inferring it from a directory listing.
                reason = no_series[sid]
                for source, subdir, windows in (
                    ("gauge_tips", "tips", months),
                    ("gauge_daily", "daily", years),
                ):
                    for label, start, end in windows:
                        index.append(
                            skipped_row(
                                source,
                                sid,
                                gauge_dir / subdir / sid / f"{label}.json.gz",
                                start,
                                end,
                                reason,
                                raw_dir,
                            )
                        )
                _log(
                    f"  [{i}/{len(stations)}] {sid} {st['station_name']}: "
                    f"{len(months) + len(years)} files not requested, {reason}"
                )
                done = i
                continue
            before = len(index)
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
            done = i
            if i % FLUSH_EVERY == 0:
                write_index(index_path(raw_dir, "gauge"), index)
    except KeyboardInterrupt:
        # The downloads themselves survive a Ctrl-C, but the index is built in
        # memory and dies with the process. Catch it here so the finally below
        # still describes what did get fetched.
        _log(f"\ninterrupted after {done} of {len(stations)} stations")
    finally:
        failures = [r for r in index if r["status"] == "failed"]
        write_index(index_path(raw_dir, "gauge"), index)
        write_report(
            report_path(raw_dir, "gauge"),
            cfg,
            stations,
            no_series,
            index,
            failures,
            skipped,
            fetched,
            time.time() - t0,
            done,
        )
        _log(
            f"\nfiles: {len(index)} indexed, {fetched} fetched this run, {skipped} already on disk, {len(failures)} failed"
        )
        _log(f"stations with no usable series: {len(no_series)}")
        _log(
            f"wrote {index_path(raw_dir, 'gauge')} and {report_path(raw_dir, 'gauge')}"
        )
    # A run that stopped early is not a success even with nothing failed, so the
    # stations it never reached count towards the exit code.
    return len(failures) + len(no_series) + (len(stations) - done)


def write_report(
    path, cfg, stations, no_series, index, failures, skipped, fetched, seconds, done
) -> None:
    """Write the run report. `done` is the number of stations fully processed; when it is
    short of the station count the report says so, because every count below then
    describes only part of the config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in index if r["status"] == "ok"]
    never_requested = [r for r in index if r["status"] == "skipped"]
    names_by_id = {st["station_id"]: st["station_name"] for st in stations}
    per_station = len(month_windows(cfg["start"], cfg["end"])) + len(
        year_windows(cfg["start"], cfg["end"])
    )
    lines = [
        f"# Gauge ingestion report: {cfg['region']}",
        "",
    ]
    if done < len(stations):
        lines += [
            f"**Incomplete run: {done} of {len(stations)} stations processed.** Every"
            " count below covers only those stations, not the whole config. Rerun to"
            " carry on; files already on disk are skipped.",
            "",
        ]
    lines += [
        f"- config: `{cfg['_path']}`",
        f"- period: {cfg['start']} to {cfg['end']} inclusive",
        f"- bounding box: {cfg['gauge']['bounds']}",
        f"- series: tips `{cfg['gauge']['series']['tips']}`, daily `{cfg['gauge']['series']['daily']}`",
        f"- run finished: {now()}, {seconds / 60:.1f} min",
        "",
        "## Counts",
        "",
        f"- stations in box: {len(stations)}",
        f"- stations processed this run: {done}",
        f"- stations with both series: {len(stations) - len(no_series)}",
        f"- stations with no usable series: {len(no_series)}",
        f"- files expected for the whole config: {len(stations) * per_station}",
        (
            f"- files in this index: {len(index)}, ok: {len(ok)},"
            f" failed: {len(failures)}, never requested: {len(never_requested)}"
        ),
        f"- fetched this run: {fetched}, already on disk: {skipped}",
        f"- tip records on disk: {sum(int(r['n_records']) for r in ok if r['source'] == 'gauge_tips'):,}",
        f"- daily records on disk: {sum(int(r['n_records']) for r in ok if r['source'] == 'gauge_daily'):,}",
        "",
        "## Stations with no usable series (excluded, with cause)",
        "",
    ]
    if no_series:
        lines += [
            (
                f"{len(no_series)} stations have no series ids, so none of their "
                f"{len(never_requested)} expected files were requested. They are in "
                "the index with status `skipped`, not readable by Component 02."
            ),
            "",
            "Rerun to retry the ones marked *will retry*; those are transient API",
            "failures. A station with *no required series* will not change on a rerun",
            "and needs a decision about the station itself.",
            "",
        ]
    lines += [
        f"- {sid} {names_by_id.get(sid, '')}: {why}"
        for sid, why in sorted(no_series.items())
    ] or ["- none"]
    # Stations whose every file came back empty: real API answers (status ok), stored
    # and indexed, but a reader of this report should not have to compute it.
    by_station = {}
    for r in ok:
        by_station.setdefault(r["station_id"], []).append(int(r["n_records"]))
    silent = sorted(sid for sid, c in by_station.items() if sum(c) == 0)
    partial = sorted(sid for sid, c in by_station.items() if 0 in c and sum(c) > 0)
    lines += [
        "",
        "## Stations with no records in the period",
        "",
        f"{len(silent)} stations returned zero records for every file (tips and daily).",
        "They are fetched and indexed like any other; deciding whether they are dry,",
        "dead or absent is Component 02's job.",
        "",
    ]
    lines += [f"- {sid} {names_by_id.get(sid, '')}" for sid in silent] or ["- none"]
    lines += [
        "",
        f"{len(partial)} further stations have some empty months but records elsewhere.",
    ]
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
