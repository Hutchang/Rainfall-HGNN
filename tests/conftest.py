"""Shared fixtures for the baseline-reconciliation tests.

Data location resolution (NFR-002 — no machine-specific absolute paths):

  1. $SYDNEY_GAUGE_ROOT if set
  2. ../COMP3888-niramay/database/australia   (the niramay worktree, BR-003 layout)
  3. ./database/australia                     (if the data has been copied in)

If none resolve, every dependent test skips with an explicit message rather
than failing spuriously (NFR-003: fail loudly, but do not fake a pass).

The same data is recoverable from this repository's own history:
    git ls-tree -r --name-only cc23a25 -- database/australia
"""

import glob
import os
from pathlib import Path

import pytest


def _candidate_roots():
    env = os.environ.get("SYDNEY_GAUGE_ROOT")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent.parent
    yield here.parent / "COMP3888-niramay" / "database" / "australia"
    yield here / "database" / "australia"


def _resolve_root():
    for c in _candidate_roots():
        if (c / "station_metadata.csv").is_file() and (c / "raw").is_dir():
            return c
    return None


DATA_ROOT = _resolve_root()

_SKIP = pytest.mark.skipif(
    DATA_ROOT is None,
    reason=(
        "Sydney gauge data not found. Set SYDNEY_GAUGE_ROOT, or create the "
        "niramay worktree: git worktree add ../COMP3888-niramay niramay-clone"
    ),
)


@pytest.fixture(scope="session")
def data_root():
    if DATA_ROOT is None:
        pytest.skip("Sydney gauge data not available")
    return DATA_ROOT


@pytest.fixture(scope="session")
def raw_files(data_root):
    files = sorted(glob.glob(str(data_root / "raw" / "*.csv")))
    if not files:
        pytest.skip("no raw station CSVs found")
    return files


@pytest.fixture(scope="session")
def gauge_long(raw_files):
    """All raw per-station observations concatenated (long format)."""
    import pandas as pd

    df = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture(scope="session")
def gauge_wide(gauge_long):
    """Pivoted [T x N] frame, as data/loader.py builds it."""
    wide = gauge_long.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    wide.columns = wide.columns.astype(str)  # match loader.py:48
    return wide


@pytest.fixture(scope="session")
def station_meta(data_root, gauge_wide):
    """Metadata filtered to stations with data — replicates data/loader.py:41-56,
    including its lack of de-duplication."""
    import pandas as pd

    meta = pd.read_csv(
        data_root / "station_metadata.csv",
        header=None,
        names=["name", "id", "network", "latitude", "longitude"],
    )
    meta["id"] = meta["id"].astype(str)
    meta["latitude"] = meta["latitude"].astype(float)
    meta["longitude"] = meta["longitude"].astype(float)
    cols = {str(c) for c in gauge_wide.columns}
    meta = meta[meta["id"].isin(cols)].copy().reset_index(drop=True)
    meta["order"] = range(len(meta))
    return meta


def pytest_collection_modifyitems(items):
    """Skip tests that need the legacy dataset when it is absent. Tests that do not
    use the data_root fixture (directly or through another fixture) still run."""
    if DATA_ROOT is None:
        for item in items:
            if "data_root" in getattr(item, "fixturenames", ()):
                item.add_marker(_SKIP)
