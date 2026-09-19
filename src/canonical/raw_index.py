"""Read Component 01's index, whichever fetchers wrote it.

Each fetcher writes its own file under `<raw_dir>/index/` — `gauge.csv`,
`radar.csv`, and so on — so that no fetcher can truncate another's rows and two
can run at once. From Component 02's side they are one table: every row has the
same columns, and `source` says which feed it came from.

The `path` column is relative to `raw_dir` in every file, so a path means the
same thing no matter which index it was read from.
"""

from pathlib import Path

import pandas as pd


def index_files(raw_dir: Path) -> list[Path]:
    """Every per-fetcher index file, sorted. Falls back to a single legacy
    `<raw_dir>/index.csv` when no `index/` directory exists, so a raw tree fetched
    before the per-fetcher split still reads."""
    raw_dir = Path(raw_dir)
    index_dir = raw_dir / "index"
    if index_dir.is_dir():
        return sorted(index_dir.glob("*.csv"))
    legacy = raw_dir / "index.csv"
    return [legacy] if legacy.exists() else []


def read_index(raw_dir: Path, **read_csv_kwargs) -> pd.DataFrame:
    """Component 01's manifest: one row per raw file, with its download status.

    `read_csv_kwargs` is passed through, because callers disagree about how to type
    the columns: the gauge reader wants station ids as strings and normal NA
    handling, the grid adapter wants every column as a literal string."""
    paths = index_files(raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"no Component 01 index under {Path(raw_dir) / 'index'} and no "
            f"{Path(raw_dir) / 'index.csv'}: nothing has been fetched yet"
        )
    kwargs = {"dtype": {"station_id": str}} | read_csv_kwargs
    frames = [pd.read_csv(p, **kwargs) for p in paths]
    return pd.concat(frames, ignore_index=True)
