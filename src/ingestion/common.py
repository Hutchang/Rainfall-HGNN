"""Shared index/report plumbing for Component 01 fetchers (gauge, radar, ...).

Every source writes the same `index.csv` shape into `database/raw/index.csv`:
one row per file it expected to produce, whether or not the fetch succeeded, so
Component 02 can tell "no rain" apart from "download failed" without listing the
directory. See docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md.
"""

import csv
import hashlib
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
    """The sha256 of a file already on disk, for the `index.csv` `sha256` column."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_index(path: Path, rows: list[dict]) -> None:
    """Write one `index.csv`: a header row plus one row per file a fetcher expected
    to produce, whether or not it succeeded."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)
