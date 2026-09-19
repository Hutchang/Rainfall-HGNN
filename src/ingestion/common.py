"""Shared index/report plumbing for Component 01 fetchers (gauge, radar, ...).

Every fetcher writes the same row shape, but each writes its **own** file under
`<raw_dir>/index/`: `gauge.csv`, `radar.csv`, and so on. One row per file the
fetcher expected to produce, whether or not the fetch succeeded, so Component 02
can tell "no rain" apart from "download failed" without listing the directory.

Splitting by fetcher rather than sharing one `index.csv` means no fetcher can
truncate another's rows, and two fetchers can run at the same time. The `source`
column stays, because one fetcher can emit several sources (the gauge fetcher
writes both `gauge_tips` and `gauge_daily`).

The `path` column is relative to `raw_dir`, not to the index file, so a path
means the same thing in every index.

`status` is one of:
  ok       the file is on disk and Component 02 may read it.
  failed   a download was attempted and broke. A rerun may fix it.
  skipped  the file is known not to be obtainable; a rerun will not change that.
           `error` carries the reason. `retrieved_at` is empty when nothing was
           ever requested, and set when we asked and the source said no.
Component 02 reads only `ok` rows, so everything else is inert downstream while
staying visible here.
"""

import csv
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

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


def now() -> str:
    """The current UTC time, formatted the way every `retrieved_at` column expects."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path) -> str:
    """The sha256 of a file already on disk, for the index's `sha256` column."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_path(raw_dir: Path, fetcher: str) -> Path:
    """Where one fetcher writes its index rows, e.g. `<raw_dir>/index/radar.csv`."""
    return Path(raw_dir) / "index" / f"{fetcher}.csv"


def report_path(raw_dir: Path, fetcher: str) -> Path:
    """Where one fetcher writes its human-readable run report."""
    return Path(raw_dir) / "report" / f"{fetcher}.md"


def write_index(path: Path, rows: list[dict]) -> None:
    """Write one fetcher's index file via a temp file and one rename, so a kill
    part-way through a flush leaves the previous index intact rather than a
    truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
