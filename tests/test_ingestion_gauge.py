"""Component 01, gauge source: acceptance tests from
docs/archive/sprint1/01_DATA_SOURCE_INGESTION_REQUIREMENTS.md "Done when".

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

from src.ingestion.common import INDEX_COLUMNS
from src.ingestion.config import load_config
from src.ingestion.gauge import fetch
from src.ingestion.gauge.kiwis import KiwisClient

SAMPLE = Path(__file__).parent / "data" / "ingestion"
CONFIG = Path(__file__).parent.parent / "config" / "ingestion" / "sydney.yaml"

# The offline tests own their config instead of reading config/ingestion/sydney.yaml.
# That file is a live project setting: its bounds and period change as the study
# region is redrawn, and a test that mutates it silently stops testing anything.
CONFIG_YAML = """
region: testregion
start: 2022-01-01
end: 2022-02-28
output_dir: database/raw
gauge:
  bounds:
    min_lat: -34.3
    max_lat: -33.4
    min_lon: 150.4
    max_lon: 151.4
  api:
    base_url: https://example.invalid/waterdata/services
    request_delay_seconds: 0
    timeout_seconds: 5
    user_agent: test-agent/0.1
  series:
    tips: DMQaQc.Merged.AsStored.1
    daily: DMQaQc.Merged.DailyTotal.09HR
radar: {}
satellite: {}
"""


@pytest.fixture
def config_text():
    """The config as text, for tests that break it on purpose."""
    return CONFIG_YAML


def _without(text, old, new):
    """Replace and prove it happened. A replace() that matches nothing leaves a valid
    config behind, so the negative test passes without testing anything - which is how
    the inverted-bounds case quietly stopped working."""
    assert old in text, f"{old!r} is not in the config fixture"
    return text.replace(old, new)


network = pytest.mark.skipif(
    not os.environ.get("BOM_NETWORK"), reason="set BOM_NETWORK=1 to hit the live API"
)


# ---------------------------------------------------------------- fake client
class FakeClient:
    """Stands in for KiwisClient. Records every call; fails on demand for use in
    negative/edge cases.

    fail_paths:  (ts_id, start) pairs whose values() call raises, as a download would.
    fail_series: station_ids whose series_list() raises RuntimeError, as KiwisClient
                 does when it exhausts its retries — a transient lookup failure.
    no_series:   station_ids that return no matching series at all, which is
                 permanent and makes find_series raise LookupError.
    """

    def __init__(self, fail_paths=(), fail_series=(), no_series=()):
        self.calls = []
        self.fail = set(fail_paths)
        self.fail_series = set(fail_series)
        self.no_series = set(no_series)

    def station_list(self, bounds=None, station_no=None):
        self.calls.append("station_list")
        header = [
            "station_id",
            "station_no",
            "station_name",
            "station_latitude",
            "station_longitude",
        ]
        # In the BoM real data, these two rain gauges are in distinct location
        # but are keyed by the same id. Negative test is designed to flag this.

        rows = [
            ["1", "563083", "A", "-33.9", "151.0"],
            ["2", "563083", "B", "-33.6", "150.3"],
        ]
        return json.dumps([header] + rows).encode(), [
            dict(zip(header, r)) for r in rows
        ]

    def series_list(self, station_id=None, ts_name=None):
        self.calls.append(f"series_list:{station_id}")
        if station_id in self.fail_series:
            raise RuntimeError("getTimeseriesList failed after 3 attempts: HTTP 504")
        if station_id in self.no_series:
            return b"", []
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
    path = tmp_path / "ingestion.yaml"
    path.write_text(CONFIG_YAML)
    cfg = load_config(path)
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

    with (raw / "index" / "gauge.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == INDEX_COLUMNS
    assert len(rows) == 2 * (2 + 1)
    assert all(
        r["status"] == "ok" and r["n_records"] == "2" and len(r["sha256"]) == 64
        for r in rows
    )
    assert (
        (raw / "report" / "gauge.md").read_text().startswith("# Gauge ingestion report")
    )


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
    with (Path(cfg["output_dir"]) / "index" / "gauge.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 6, (
            "index still covers every expected file"
        )


def test_failed_download_is_indexed_and_run_exits_nonzero(tmp_path):
    cfg = small_cfg(tmp_path)
    client = FakeClient(fail_paths={("101", "2022-02-01")})
    failures = fetch.run(cfg, client=client)
    assert failures == 1
    with (Path(cfg["output_dir"]) / "index" / "gauge.csv").open() as fh:
        rows = {r["path"]: r for r in csv.DictReader(fh)}
    bad = rows["gauge/tips/1/2022-02.json.gz"]
    assert bad["status"] == "failed" and "504" in bad["error"] and bad["sha256"] == ""
    assert not (Path(cfg["output_dir"]) / "gauge/tips/1/2022-02.json.gz").exists()
    assert (
        "gauge/tips/1/2022-02.json.gz"
        in (Path(cfg["output_dir"]) / "report" / "gauge.md").read_text()
    )
    # rerun retries only the failed file
    retry = FakeClient()
    assert fetch.run(cfg, client=retry) == 0
    assert [c for c in retry.calls if c.startswith("values")] == [
        "values:101:2022-02-01"
    ]


def _index(cfg):
    with (Path(cfg["output_dir"]) / "index" / "gauge.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_transient_series_failure_is_retried_on_the_next_run(tmp_path):
    """Station 2's series lookup hits a 504 on the first run; the API is healthy on
    the second. FakeClient returns two stations and small_cfg asks for 2 tip months
    plus 1 daily year, so a complete run is 2 stations x 3 files = 6 index rows.

    Run 1 must exit non-zero and still leave station 2 visible in the index.
    Run 2 must re-ask ONLY station 2: re-asking station 1 would mean the cache was
    abandoned, and not re-asking station 2 is the original bug, where one transient
    error was cached as "already looked up" and the station silently disappeared
    from every later run.
    """
    cfg = small_cfg(tmp_path)

    first = FakeClient(fail_series={"2"})
    assert fetch.run(cfg, client=first) == 1, "an unresolved station must exit non-zero"
    assert {r["station_id"] for r in _index(cfg)} == {"1", "2"}, (
        "station 2 stays in the index even though nothing was downloaded for it"
    )

    second = FakeClient()
    assert fetch.run(cfg, client=second) == 0
    assert "series_list:2" in second.calls, "the failed lookup must be retried"
    assert "series_list:1" not in second.calls, "a cached station must not be re-asked"

    rows = _index(cfg)
    assert all(r["status"] == "ok" for r in rows), (
        "run 1's skipped rows are replaced by real downloads, not left behind"
    )
    assert len(rows) == 2 * (2 + 1), "2 stations x (2 tip months + 1 daily year)"
    assert len({(r["station_id"], r["path"]) for r in rows}) == len(rows), (
        "one row per file, so the retry did not append a second row for station 2"
    )


def test_station_with_no_series_is_indexed_reported_and_kept_non_zero(tmp_path):
    """A permanent exclusion must stay visible on every run, not just the first."""
    cfg = small_cfg(tmp_path)

    for run_number in (1, 2):
        assert fetch.run(cfg, client=FakeClient(no_series={"2"})) == 1, (
            f"run {run_number}: a station with no required series exits non-zero"
        )
        rows = [r for r in _index(cfg) if r["station_id"] == "2"]
        assert len(rows) == 3, f"run {run_number}: a row per expected file"
        assert {r["status"] for r in rows} == {"skipped"}
        assert all("no required series" in r["error"] for r in rows)
        assert all(r["sha256"] == "" and r["n_records"] == "" for r in rows), (
            "nothing was downloaded, so there is no hash and no record count"
        )
        report = (Path(cfg["output_dir"]) / "report" / "gauge.md").read_text()
        assert "- stations with no usable series: 1" in report
        assert "never requested: 3" in report
        assert "2 B: no required series" in report


def test_skipped_rows_are_invisible_to_component_02(tmp_path):
    """Component 02 selects status == "ok", so a skipped row must never reach it."""
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient(no_series={"2"}))
    rows = _index(cfg)
    assert [r["station_id"] for r in rows if r["status"] == "ok"] == ["1", "1", "1"]
    assert {r["station_id"] for r in rows} == {"1", "2"}, (
        "the skipped station is still on the record for a human and for 01's report"
    )


def test_partially_written_series_csv_is_not_trusted(tmp_path):
    """A crash mid-write can leave a station with one of its two series. That is
    not a completed lookup, so the station is looked up again rather than fetched
    with a missing ts_id."""
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient())
    path = Path(cfg["output_dir"]) / "gauge" / "series.csv"

    rows = [r for r in csv.DictReader(path.open(newline=""))]
    keep = [
        r
        for r in rows
        if not (r["station_id"] == "1" and "DailyTotal" in r["series_name"])
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fetch.SERIES_COLUMNS)
        w.writeheader()
        w.writerows(keep)

    assert fetch.read_series_csv(path, cfg["gauge"]["series"]) == {
        "2": {"tips": "201", "daily": "202"}
    }, "station 1 is incomplete and must not count as cached"

    third = FakeClient()
    assert fetch.run(cfg, client=third) == 0
    assert "series_list:1" in third.calls and "series_list:2" not in third.calls


def test_series_csv_keeps_its_documented_columns(tmp_path):
    """Series.csv designed to exactly these four columns, so
    the header is a contract with any future reader. Adding a column here
    must fail this test and force the doc to change with it."""
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient())
    path = Path(cfg["output_dir"]) / "gauge" / "series.csv"
    with path.open(newline="") as fh:
        assert next(csv.reader(fh)) == fetch.SERIES_COLUMNS
    assert fetch.SERIES_COLUMNS == [
        "station_id",
        "ts_id",
        "series_name",
        "parameter",
    ], "these four names are published in 01's requirements doc"


def test_series_csv_is_written_once_per_newly_resolved_station(tmp_path):
    """series.csv is rewritten whole, so it must change when a station resolves and
    be left alone when nothing does.

    Run 1 leaves station 2 unresolved. Run 2 resolves it, so the file must change.
    Run 3 resolves nothing, so the file must not be touched at all — checked by
    modification time, because write_series_csv sorts by station_id and a pointless
    rewrite would therefore produce byte-identical content that a content
    comparison could not detect.
    """
    cfg = small_cfg(tmp_path)
    fetch.run(cfg, client=FakeClient(fail_series={"2"}))
    path = Path(cfg["output_dir"]) / "gauge" / "series.csv"
    after_first = path.read_bytes()
    assert [r["station_id"] for r in csv.DictReader(path.open(newline=""))] == [
        "1",
        "1",
    ], "only the station whose lookup succeeded is cached"

    fetch.run(cfg, client=FakeClient())
    assert path.read_bytes() != after_first, "station 2 resolved and must be added"
    after_second, mtime = path.read_bytes(), path.stat().st_mtime_ns

    fetch.run(cfg, client=FakeClient())
    assert path.stat().st_mtime_ns == mtime, "nothing resolved, so nothing was written"
    assert path.read_bytes() == after_second


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
def test_config_rejects_missing_series_and_inverted_bounds(tmp_path, config_text):
    bad = tmp_path / "bad.yaml"
    bad.write_text(_without(config_text, "  series:\n", "  serie:\n"))
    with pytest.raises(ValueError, match="gauge section is missing"):
        load_config(bad)
    bad.write_text(_without(config_text, "min_lat: -34.3", "min_lat: -33.0"))
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
