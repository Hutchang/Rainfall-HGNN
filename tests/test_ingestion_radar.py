"""Component 01, radar source: acceptance tests from
docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md "Done when".

Offline tests use a fake NCI client and the committed sample under
tests/data/ingestion/radar/. Network tests hit the live NCI THREDDS archive and are
skipped unless NCI_NETWORK=1 is set:

    NCI_NETWORK=1 uv run python -m pytest tests/test_ingestion_radar.py -v
"""

import csv
import io
import os
import shutil
import zipfile
from datetime import date
from pathlib import Path

import pytest
import xarray as xr

from src.ingestion.common import INDEX_COLUMNS
from src.ingestion.config import load_config
from src.ingestion.radar import fetch
from src.ingestion.radar.coords import read_grid, to_wgs84_transformer
from src.ingestion.radar.nci import NciClient

SAMPLE_ZIP = (
    Path(__file__).parent / "data" / "ingestion" / "radar" / "71_20220103.prcp-m15.zip"
)
CONFIG = Path(__file__).parent.parent / "config" / "ingestion" / "sydney.yaml"

network = pytest.mark.skipif(
    not os.environ.get("NCI_NETWORK"), reason="set NCI_NETWORK=1 to hit the live API"
)


# ---------------------------------------------------------------- fake client
class FakeClient:
    """Stands in for NciClient. Records every call; fails on demand; returns the
    committed real sample zip's bytes for every successful day."""

    def __init__(self, fail_days=()):
        self.calls = []
        self.fail = set(fail_days)

    def daily_zip(self, site_id, product, day):
        self.calls.append((site_id, product, day))
        if day in self.fail:
            raise RuntimeError("HTTP 504")
        return SAMPLE_ZIP.read_bytes()


def small_cfg(tmp_path):
    cfg = load_config(CONFIG)
    cfg["start"], cfg["end"] = date(2022, 1, 1), date(2022, 1, 3)  # 3 days
    cfg["output_dir"] = str(tmp_path / "raw")
    return cfg


# ---------------------------------------------------------------- windows
def test_day_windows_cover_period_inclusive():
    days = fetch.day_windows(date(2022, 1, 30), date(2022, 2, 2))
    assert days == [date(2022, 1, 30), date(2022, 1, 31), date(2022, 2, 1), date(2022, 2, 2)]


# ---------------------------------------------------------------- the run
def test_run_writes_layout_index_and_coords(tmp_path):
    cfg = small_cfg(tmp_path)
    failures = fetch.run(cfg, client=FakeClient())
    raw = Path(cfg["output_dir"])

    assert failures == 0
    for day in ("2022-01-01", "2022-01-02", "2022-01-03"):
        assert (raw / "radar" / "2022" / f"{day}.prcp-m15.zip").exists()

    with (raw / "index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == INDEX_COLUMNS
    assert len(rows) == 3
    assert all(
        r["source"] == "radar_precip"
        and r["station_id"] == ""
        and r["status"] == "ok"
        and r["n_records"] == "2"  # the committed sample has 2 .nc members
        and len(r["sha256"]) == 64
        for r in rows
    )

    coords = raw / "radar" / "radar_node_coords.csv"
    assert coords.exists()
    with coords.open() as fh:
        coord_rows = list(csv.DictReader(fh))
    assert len(coord_rows) == 512 * 512
    assert list(coord_rows[0].keys()) == ["node_id", "latitude", "longitude"]
    assert coord_rows[0]["node_id"] == "0"

    assert (raw / "report.md").read_text().startswith("# Radar ingestion report")


def test_second_run_skips_existing_files_and_makes_no_downloads(tmp_path):
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient())
    second = FakeClient()
    fetch.run(cfg, client=second)
    assert second.calls == []
    with (Path(cfg["output_dir"]) / "index.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 3, "index still covers every expected day"


def test_failed_download_is_indexed_and_run_exits_nonzero(tmp_path):
    cfg = small_cfg(tmp_path)
    client = FakeClient(fail_days={date(2022, 1, 2)})
    failures = fetch.run(cfg, client=client)
    assert failures == 1
    with (Path(cfg["output_dir"]) / "index.csv").open() as fh:
        rows = {r["path"]: r for r in csv.DictReader(fh)}
    bad = rows["radar/2022/2022-01-02.prcp-m15.zip"]
    assert bad["status"] == "failed" and "504" in bad["error"] and bad["sha256"] == ""
    assert not (Path(cfg["output_dir"]) / "radar/2022/2022-01-02.prcp-m15.zip").exists()
    assert (
        "radar/2022/2022-01-02.prcp-m15.zip"
        in (Path(cfg["output_dir"]) / "report.md").read_text()
    )
    # rerun retries only the failed day
    retry = FakeClient()
    assert fetch.run(cfg, client=retry) == 0
    assert [c[2] for c in retry.calls] == [date(2022, 1, 2)]


def test_empty_radar_config_is_a_no_op(tmp_path):
    cfg = small_cfg(tmp_path)
    cfg["radar"] = {}
    assert fetch.run(cfg, client=FakeClient()) == 0
    assert not (Path(cfg["output_dir"]) / "index.csv").exists()


# ---------------------------------------------------------------- coordinates
def test_committed_sample_is_an_unmodified_api_response():
    with zipfile.ZipFile(SAMPLE_ZIP) as zf:
        names = zf.namelist()
        assert names == [
            "71_20220103_000000.prcp-m15.nc",
            "71_20220103_001500.prcp-m15.nc",
        ]
        raw = zf.read(names[0])
    tmp = SAMPLE_ZIP.parent / "_tmp_check.nc"
    tmp.write_bytes(raw)
    try:
        ds = xr.open_dataset(tmp)
        assert ds.attrs["station_id"] == 71
        assert ds.attrs["station_name"] == "T_Hills"
        assert "Merged Gauge/Radar" in ds.attrs["title"]
        assert ds["precipitation"].shape == (512, 512)
    finally:
        tmp.unlink()


def test_coordinate_transform_resolves_near_radar_origin():
    """The centre of the grid must resolve close to the radar's own declared
    origin (lat -33.7008, lon 151.209) — proves the CF projection is being read
    and applied correctly, not assumed."""
    with zipfile.ZipFile(SAMPLE_ZIP) as zf:
        name = zf.namelist()[0]
        tmp = SAMPLE_ZIP.parent / "_tmp_check2.nc"
        tmp.write_bytes(zf.read(name))
    try:
        grid = read_grid(tmp)
        transformer = to_wgs84_transformer(grid["proj_attrs"])
        lon, lat = transformer.transform(0.0, 0.0)  # the projection's own origin
        assert lat == pytest.approx(-33.7008, abs=0.01)
        assert lon == pytest.approx(151.209, abs=0.01)
    finally:
        tmp.unlink()


# ---------------------------------------------------------------- live API
@network
def test_live_fetch_one_day_matches_expected_product():
    """Doc 01 provenance check: the live file for site 71 really is the gauge-
    corrected Rainfields3 product, and its contributing gauges can be cross-checked
    against the Sydney station list for leakage."""
    cfg = load_config(CONFIG)
    client = NciClient(**cfg["radar"]["api"])
    raw = client.daily_zip(cfg["radar"]["site_id"], cfg["radar"]["product"], date(2022, 1, 1))
    tmp_dir = Path(__file__).parent / "_tmp_live"
    tmp_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # the real daily zip's first entry is a `_file_list` manifest, not data
            name = next(n for n in zf.namelist() if n.endswith(".nc"))
            path = Path(zf.extract(name, tmp_dir))
        ds = xr.open_dataset(path)
        assert ds.attrs["station_name"] == "T_Hills"
        assert "Merged Gauge/Radar" in ds.attrs["title"]
        assert len(ds.attrs["contributing_gauges"]) > 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
