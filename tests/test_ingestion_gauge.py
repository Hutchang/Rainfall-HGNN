"""Component 01, gauge source: acceptance tests from
docs/requirements/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md "Done when".

Offline tests use a fake client and the committed sample under tests/data/ingestion/.
Network tests hit the live BoM API and are skipped unless BOM_NETWORK=1 is set:

    BOM_NETWORK=1 uv run python -m pytest tests/test_ingestion_gauge.py -v
"""

import csv
import gzip
import json
import os
from datetime import date
from pathlib import Path

import pytest

from src.ingestion.config import load_config
from src.ingestion.gauge import fetch
from src.ingestion.gauge.kiwis import KiwisClient

SAMPLE = Path(__file__).parent / "data" / "ingestion"
CONFIG = Path(__file__).parent.parent / "config" / "ingestion" / "sydney.yaml"

network = pytest.mark.skipif(
    not os.environ.get("BOM_NETWORK"), reason="set BOM_NETWORK=1 to hit the live API"
)


# ---------------------------------------------------------------- fake client
class FakeClient:
    """Stands in for KiwisClient. Records every call; fails on demand."""

    def __init__(self, fail_paths=()):
        self.calls = []
        self.fail = set(fail_paths)

    def station_list(self, bounds=None, station_no=None):
        self.calls.append("station_list")
        header = [
            "station_id",
            "station_no",
            "station_name",
            "station_latitude",
            "station_longitude",
        ]
        rows = [
            ["1", "563083", "A", "-33.9", "151.0"],
            ["2", "563083", "B", "-33.6", "150.3"],
        ]
        return json.dumps([header] + rows).encode(), [
            dict(zip(header, r)) for r in rows
        ]

    def series_list(self, station_id=None, ts_name=None):
        self.calls.append(f"series_list:{station_id}")
        rows = [
            {
                "station_id": station_id,
                "ts_id": f"{station_id}01",
                "ts_name": "DMQaQc.Merged.AsStored.1",
            },
            {
                "station_id": station_id,
                "ts_id": f"{station_id}02",
                "ts_name": "DMQaQc.Merged.DailyTotal.09HR",
            },
        ]
        return b"", rows

    def values(self, ts_id, start, end):
        self.calls.append(f"values:{ts_id}:{start}")
        if (ts_id, start) in self.fail:
            raise RuntimeError("HTTP 504")
        data = [
            [f"{start}T09:00:00.000+10:00", 0.0, 10],
            [f"{start}T10:00:00.000+10:00", 0.5, 10],
        ]
        payload = [
            {
                "ts_id": ts_id,
                "rows": "2",
                "columns": "Timestamp,Value,Quality Code",
                "data": data,
            }
        ]
        return json.dumps(payload).encode(), 2


def small_cfg(tmp_path):
    cfg = load_config(CONFIG)
    cfg["start"], cfg["end"] = date(2022, 1, 1), date(2022, 2, 28)  # 2 months, 1 year
    cfg["output_dir"] = str(tmp_path / "raw")
    return cfg


# ---------------------------------------------------------------- windows
def test_month_windows_cover_period_and_clip_to_it():
    w = fetch.month_windows(date(2021, 1, 1), date(2022, 12, 31))
    assert len(w) == 24
    assert w[0] == ("2021-01", date(2021, 1, 1), date(2021, 1, 31))
    assert w[-1] == ("2022-12", date(2022, 12, 1), date(2022, 12, 31))
    clipped = fetch.month_windows(date(2021, 3, 15), date(2021, 4, 10))
    assert clipped == [
        ("2021-03", date(2021, 3, 15), date(2021, 3, 31)),
        ("2021-04", date(2021, 4, 1), date(2021, 4, 10)),
    ]


def test_year_windows():
    assert fetch.year_windows(date(2021, 1, 1), date(2022, 12, 31)) == [
        ("2021", date(2021, 1, 1), date(2021, 12, 31)),
        ("2022", date(2022, 1, 1), date(2022, 12, 31)),
    ]


# ---------------------------------------------------------------- raw writer
def test_write_raw_is_byte_faithful_reproducible_and_never_overwrites(tmp_path):
    p = tmp_path / "a.json.gz"
    sha1 = fetch.write_raw(p, b'{"x": 1}')
    with gzip.open(p) as fh:
        assert fh.read() == b'{"x": 1}'
    with pytest.raises(FileExistsError):
        fetch.write_raw(p, b"other")
    sha2 = fetch.write_raw(tmp_path / "b.json.gz", b'{"x": 1}')
    assert sha1 == sha2, "same bytes must give the same sha256 (gzip mtime is zeroed)"


# ---------------------------------------------------------------- the run
def test_run_writes_layout_index_and_report(tmp_path):
    cfg = small_cfg(tmp_path)
    client = FakeClient()
    failures = fetch.run(cfg, client=client)
    raw = Path(cfg["output_dir"])

    assert failures == 0
    assert (raw / "gauge" / "stations.json").exists()
    assert (raw / "gauge" / "series.csv").exists()
    for sid in ("1", "2"):
        assert (raw / "gauge" / "tips" / sid / "2022-01.json.gz").exists()
        assert (raw / "gauge" / "tips" / sid / "2022-02.json.gz").exists()
        assert (raw / "gauge" / "daily" / sid / "2022.json.gz").exists()

    with (raw / "index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == fetch.INDEX_COLUMNS
    assert len(rows) == 2 * (2 + 1)
    assert all(
        r["status"] == "ok" and r["n_records"] == "2" and len(r["sha256"]) == 64
        for r in rows
    )
    assert (raw / "report.md").read_text().startswith("# Gauge ingestion report")


def test_second_run_skips_existing_files_and_makes_no_value_calls(tmp_path):
    cfg = small_cfg(tmp_path)
    first = FakeClient()
    fetch.run(cfg, client=first)
    second = FakeClient()
    fetch.run(cfg, client=second)
    assert not [
        c
        for c in second.calls
        if c.startswith(("values", "station_list", "series_list"))
    ]
    with (Path(cfg["output_dir"]) / "index.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 6, (
            "index still covers every expected file"
        )


def test_failed_download_is_indexed_and_run_exits_nonzero(tmp_path):
    cfg = small_cfg(tmp_path)
    client = FakeClient(fail_paths={("101", "2022-02-01")})
    failures = fetch.run(cfg, client=client)
    assert failures == 1
    with (Path(cfg["output_dir"]) / "index.csv").open() as fh:
        rows = {r["path"]: r for r in csv.DictReader(fh)}
    bad = rows["gauge/tips/1/2022-02.json.gz"]
    assert bad["status"] == "failed" and "504" in bad["error"] and bad["sha256"] == ""
    assert not (Path(cfg["output_dir"]) / "gauge/tips/1/2022-02.json.gz").exists()
    assert (
        "gauge/tips/1/2022-02.json.gz"
        in (Path(cfg["output_dir"]) / "report.md").read_text()
    )
    # rerun retries only the failed file
    retry = FakeClient()
    assert fetch.run(cfg, client=retry) == 0
    assert [c for c in retry.calls if c.startswith("values")] == [
        "values:101:2022-02-01"
    ]


def test_stations_sharing_a_public_number_are_kept_separate(tmp_path):
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient())
    with (Path(cfg["output_dir"]) / "gauge" / "series.csv").open() as fh:
        ids = {r["station_id"] for r in csv.DictReader(fh)}
    assert ids == {"1", "2"}, "two station_ids with the same station_no both fetched"


# ---------------------------------------------------------------- committed sample
def test_committed_sample_is_an_unmodified_api_response():
    with gzip.open(SAMPLE / "gauge/tips/31531755/2022-01.json.gz") as fh:
        tips = json.load(fh)
    assert set(tips[0]) == {"ts_id", "rows", "columns", "data"}
    assert tips[0]["columns"] == "Timestamp,Value,Quality Code"
    assert int(tips[0]["rows"]) == len(tips[0]["data"]) == 390
    assert tips[0]["data"][0] == ["2022-01-01T03:00:00.000+10:00", 0.0, 10]
    assert all(t[0].endswith("+10:00") for t in tips[0]["data"]), "fixed UTC+10 offset"

    with gzip.open(SAMPLE / "gauge/daily/31531755/2022.json.gz") as fh:
        daily = json.load(fh)
    assert len(daily[0]["data"]) == 365
    assert all(d[0][11:19] == "09:00:00" for d in daily[0]["data"]), (
        "rain day ends 09:00"
    )


# ---------------------------------------------------------------- config
def test_config_rejects_missing_series_and_inverted_bounds(tmp_path):
    text = CONFIG.read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("  series:\n", "  serie:\n"))
    with pytest.raises(ValueError, match="gauge section is missing"):
        load_config(bad)
    bad.write_text(text.replace("min_lat: -34.3", "min_lat: -33.0"))
    with pytest.raises(ValueError, match="min must be less than max"):
        load_config(bad)


# ---------------------------------------------------------------- live API
@network
def test_storm_check_station_568163_january_2023():
    """Doc 01 'Done when': 50 tips 19:00-20:00 AEST on 4 Jan 2023 (25.0 mm),
    adjacent 09:00 daily totals 4.0 and 30.0 mm."""
    cfg = load_config(CONFIG)
    client = KiwisClient(**cfg["gauge"]["api"])
    _, st = client.station_list(station_no="568163")
    assert len(st) == 1
    ids = fetch.find_series(client, st[0]["station_id"], cfg["gauge"]["series"])
    raw, _ = client.values(ids["tips"], "2023-01-04", "2023-01-05")
    tips = [d for d in json.loads(raw)[0]["data"] if d[0].startswith("2023-01-04T19")]
    assert len(tips) == 50 and sum(d[1] for d in tips) == 25.0
    raw, _ = client.values(ids["daily"], "2023-01-04", "2023-01-05")
    assert [d[1] for d in json.loads(raw)[0]["data"]] == [4.0, 30.0]


@network
def test_collision_check_station_no_563083_is_two_stations():
    cfg = load_config(CONFIG)
    client = KiwisClient(**cfg["gauge"]["api"])
    _, rows = client.station_list(station_no="563083")
    got = {r["station_id"]: r["station_name"] for r in rows}
    assert got == {"7730155": "BLACKHEATH G.C.", "49645132": "NORTH RICHMOND WFP"}
