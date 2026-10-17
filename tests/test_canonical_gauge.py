"""Component 02, gauge half: acceptance tests from docs/archive/sprint1/02_CANONICAL_DATA_REQUIREMENTS.md
"Done when", plus the plausibility and station rules. Offline: synthetic records and the
committed Component 01 sample under tests/data/ingestion/.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.canonical.config import load_config
from src.canonical.gauge import raw
from src.canonical.gauge.build import load_gauge_hourly, load_stations
from src.canonical.gauge.stations import build_station_table
from src.canonical.gauge.verify import judgeable_hours, rain_day_end, verify_days

CONFIG = Path(__file__).parent.parent / "config" / "canonical" / "standard.yaml"
SAMPLE = Path(__file__).parent / "data" / "ingestion"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def tips_df(rows):
    """rows: (station_id, local timestamp 'YYYY-MM-DD HH:MM:SS', value, quality)."""
    df = pd.DataFrame(rows, columns=["station_id", "timestamp", "value", "quality"])
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize("Etc/GMT-10")
        .dt.tz_convert("UTC")
    )
    return df


def daily_df(rows):
    """rows: (station_id, local rain-day end 'YYYY-MM-DD 09:00:00', total, quality)."""
    return tips_df(rows)


# ---------------------------------------------------------------- rain day and hours
def test_rain_day_boundary_is_0900_local_exclusive_at_end():
    """The day is [start, end), so a tip at exactly 09:00:00 opens the day ending 24
    hours later. It has to, because the hourly grid is 24 buckets labelled by their
    start and covering [t, t+1h), whereas a (start, end] day spans 25 of those, so its
    boundary tip verifies one day and lands in the next one's grid.

    See the `rain_day_end` docstring in src/canonical/gauge/verify.py."""
    ts = pd.Series(
        pd.to_datetime(
            ["2022-01-05 08:59:59", "2022-01-05 09:00:00", "2022-01-05 09:00:01"]
        )
    )
    ts = ts.dt.tz_localize("Etc/GMT-10").dt.tz_convert("UTC")
    ends = (rain_day_end(ts) + pd.Timedelta(hours=10)).dt.strftime("%Y-%m-%d %H:%M")
    assert ends.tolist() == ["2022-01-05 09:00", "2022-01-06 09:00", "2022-01-06 09:00"]


def test_tip_at_1059_lands_in_10_hour_and_1100_in_11_hour(cfg):
    tips = tips_df(
        [("s", "2022-01-05 10:59:59", 0.5, 10), ("s", "2022-01-05 11:00:00", 0.5, 10)]
    )
    daily = daily_df([("s", "2022-01-06 09:00:00", 1.0, 10)])
    hourly, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert days["verdict"].tolist() == ["verified"]
    local = hourly.set_index(
        hourly["timestamp"].dt.tz_convert("Etc/GMT-10").dt.strftime("%H:%M")
    )["rainfall_mm"]
    assert local["10:00"] == 0.5 and local["11:00"] == 0.5 and local["12:00"] == 0.0


def test_verified_day_gives_24_rows_and_unverified_gives_none(cfg):
    tips = tips_df(
        [("s", "2022-01-05 12:00:00", 0.5, 10), ("s", "2022-01-06 12:00:00", 0.5, 10)]
    )
    daily = daily_df(
        [("s", "2022-01-06 09:00:00", 0.5, 10), ("s", "2022-01-07 09:00:00", 3.0, 10)]
    )
    hourly, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert dict(zip(days["day_end"].dt.strftime("%m-%d"), days["verdict"])) == {
        "01-05": "verified",
        "01-06": "mismatch",
    }
    assert len(hourly) == 24
    assert hourly["timestamp"].min() == pd.Timestamp("2022-01-04 23:00", tz="UTC")
    assert hourly["timestamp"].is_monotonic_increasing and hourly["timestamp"].is_unique


@pytest.mark.parametrize(
    "changeover", ["2021-04-04", "2021-10-03", "2022-04-03", "2022-10-02"]
)
def test_daylight_saving_changeovers_give_distinct_ordered_utc_hours(cfg, changeover):
    """BoM serves fixed +10:00 so no local-clock seam exists; this pins that the
    conversion never reintroduces one."""
    # 02:30 local is before 09:00, so it belongs to the rain day ending that same morning
    tips = tips_df([("s", f"{changeover} 02:30:00", 0.5, 10)])
    daily = daily_df([("s", f"{changeover} 09:00:00", 0.5, 10)])
    hourly, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2021, 1, 1),
        date(2022, 12, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert days["verdict"].tolist() == ["verified"]
    assert len(hourly) == 24 and hourly["timestamp"].is_unique
    assert (hourly["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_markers_and_ignored_codes_do_not_count_and_gaps_are_unverifiable(cfg):
    tips = tips_df(
        [
            ("s", "2022-01-05 09:00:00", 0.0, 10),  # marker
            ("s", "2022-01-05 12:00:00", 0.5, 210),  # ignored code
            ("s", "2022-01-05 13:00:00", 0.5, 10),
            ("s", "2022-01-06 12:00:00", None, -1),  # gap marker
        ]
    )
    daily = daily_df(
        [("s", "2022-01-06 09:00:00", 0.5, 10), ("s", "2022-01-07 09:00:00", None, -1)]
    )
    hourly, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    v = dict(zip(days["day_end"].dt.strftime("%m-%d"), days["verdict"]))
    assert v == {"01-05": "verified", "01-06": "unverifiable"}
    assert hourly["rainfall_mm"].sum() == 0.5


def test_tolerance_is_one_increment_per_station(cfg):
    tips = tips_df(
        [("a", "2022-01-05 12:00:00", 0.2, 10)] * 3
        + [("b", "2022-01-05 12:00:00", 0.5, 10)] * 3
    )
    daily = daily_df(
        [("a", "2022-01-06 09:00:00", 0.8, 10), ("b", "2022-01-06 09:00:00", 1.0, 10)]
    )
    _, days, inc = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert inc.to_dict() == {"a": 0.2, "b": 0.5}
    assert dict(zip(days["station_id"], days["verdict"])) == {
        "a": "verified",
        "b": "verified",
    }
    daily2 = daily_df(
        [("a", "2022-01-06 09:00:00", 0.9, 10)]
    )  # 0.3 off at a 0.2 station
    _, days2, _ = verify_days(
        tips[tips.station_id == "a"],
        daily2,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert days2["verdict"].tolist() == ["mismatch"]


# ---------------------------------------------------------------- plausibility caps
def test_each_plausibility_cap_rejects_the_whole_day(cfg):
    record = tips_df([("s", "2022-01-05 12:00:00", 5.0, 10)])
    burst = tips_df(
        [
            ("s", f"2022-01-05 12:{m:02d}:{sec:02d}", 0.5, 10)
            for m in range(60)
            for sec in range(0, 60, 10)
        ]
    )
    big = tips_df(
        [
            ("s", f"2022-01-05 {h:02d}:{m:02d}:00", 0.5, 10)
            for h in range(10, 24)
            for m in range(60)
        ]
    )
    for tips, total, reason in [
        (record, 5.0, "record>"),
        (burst, 180.0, "hour>"),
        (big, 420.0, "day>"),
    ]:
        daily = daily_df([("s", "2022-01-06 09:00:00", total, 10)])
        hourly, days, _ = verify_days(
            tips,
            daily,
            cfg["quality"],
            cfg["plausibility"],
            date(2022, 1, 1),
            date(2022, 1, 31),
            default_increment=cfg["default_tip_increment_mm"],
        )
        assert days["verdict"].tolist() == ["implausible"], reason
        assert days["reason"].iloc[0].startswith(reason)
        assert hourly.empty


# ---------------------------------------------------------------- committed sample
def test_sample_month_verifies_against_bom_daily_totals(cfg):
    index = pd.DataFrame(
        [
            {
                "source": "gauge_tips",
                "station_id": "31531755",
                "path": "gauge/tips/31531755/2022-01.json.gz",
                "start": "2022-01-01",
                "end": "2022-01-31",
                "status": "ok",
            },
            {
                "source": "gauge_daily",
                "station_id": "31531755",
                "path": "gauge/daily/31531755/2022.json.gz",
                "start": "2022-01-01",
                "end": "2022-12-31",
                "status": "ok",
            },
            {
                "source": "gauge_tips",
                "station_id": "31531755",
                "path": "gauge/tips/31531755/2022-02.json.gz",
                "start": "2022-02-01",
                "end": "2022-02-28",
                "status": "failed",
            },
        ]
    )
    tips = raw.read_tips(
        SAMPLE, index
    )  # the failed row must not be opened (it does not exist)
    daily = raw.read_daily(SAMPLE, index)
    assert tips["timestamp"].dt.tz is not None and str(tips["timestamp"].dt.tz) == "UTC"
    hourly, days, inc = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert inc.to_dict() == {"31531755": 0.5}
    assert days["verdict"].value_counts().get("verified", 0) >= 25, days[
        "verdict"
    ].value_counts()
    assert (hourly.groupby(hourly["timestamp"].dt.floor("D")).size() <= 24).all()
    assert len(hourly) == 24 * (days["verdict"] == "verified").sum()


# ---------------------------------------------------------------- station table
def stations_df():
    return pd.DataFrame(
        {
            "station_id": ["a", "b", "c", "d", "e"],
            "station_no": ["563083", "563083", "567163", "567163", "999"],
            "station_name": ["A", "B", "C", "D", "E"],
            "latitude": [-33.80, -33.63, -33.77, -33.77, -33.90],
            "longitude": [150.70, 150.28, 150.67, 150.6701, 151.10],
        }
    )


def test_collision_pairs_stay_four_rows_and_colocation_is_flagged(cfg):
    n = 24 * 100
    hourly = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": s,
                    "timestamp": pd.date_range(
                        "2022-01-01", periods=n, freq="h", tz="UTC"
                    ),
                    "rainfall_mm": 0.0,
                }
            )
            for s in ["a", "b", "c", "d"]
        ]
    )
    days = pd.DataFrame(
        {
            "station_id": ["a"],
            "day_end": [pd.Timestamp("2022-01-02 23:00", tz="UTC")],
            "verdict": ["verified"],
            "tip_mm": [1.0],
            "daily_quality": [10],
        }
    )
    st = build_station_table(
        stations_df(),
        hourly,
        days,
        pd.Series({"a": 0.5}, name="tip_increment_mm"),
        n,
        cfg,
    )
    assert len(st) == 5 and st["station_id"].is_unique
    assert st.set_index("station_id").loc[["a", "b", "c", "d"], "use_in_study"].all()
    assert (
        st.set_index("station_id").loc["e", "exclusion_reason"]
        == "no records in period"
    )
    coloc = st.set_index("station_id")["colocated_with"]
    assert (
        coloc["c"] == "d" and coloc["d"] == "c" and coloc["a"] == ""
    )  # c-d are 9 m apart


def test_coverage_and_dead_gauge_rules(cfg):
    n = 24 * 100
    hours = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    hourly = pd.concat(
        [
            pd.DataFrame({"station_id": "a", "timestamp": hours, "rainfall_mm": 0.0}),
            pd.DataFrame(
                {"station_id": "e", "timestamp": hours[: n // 2], "rainfall_mm": 0.0}
            ),  # 50% coverage
        ]
    )
    # a: 95 consecutive verified-dry days; c and d (and b) wet on those days, all within 50 km of a
    ends = pd.date_range("2022-01-01 23:00", periods=95, freq="D", tz="UTC")
    days = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": "a",
                    "day_end": ends,
                    "verdict": "verified",
                    "tip_mm": 0.0,
                    "daily_quality": 10,
                }
            ),
            *[
                pd.DataFrame(
                    {
                        "station_id": s,
                        "day_end": ends,
                        "verdict": "verified",
                        "tip_mm": 2.0,
                        "daily_quality": 10,
                    }
                )
                for s in "bcd"
            ],
        ]
    )
    st = build_station_table(
        stations_df(),
        hourly,
        days,
        pd.Series(dtype=float, name="tip_increment_mm"),
        n,
        cfg,
    ).set_index("station_id")
    assert st.loc["a", "exclusion_reason"].startswith(
        "dead gauge: 95 verified-dry days"
    )
    assert st.loc["e", "exclusion_reason"].startswith("coverage 0.5 <")
    assert not st.loc["a", "use_in_study"] and not st.loc["e", "use_in_study"]


# ---------------------------------------------------------------- output conventions
def test_written_tables_round_trip_with_string_ids_and_utc_offsets(tmp_path):
    from src.canonical.gauge.build import _fmt_ts

    h = pd.DataFrame(
        {
            "station_id": ["0012"],
            "timestamp": pd.to_datetime(["2022-01-01 00:00"], utc=True),
            "rainfall_mm": [0.5],
        }
    )
    h["timestamp"] = _fmt_ts(h["timestamp"])
    h.to_csv(tmp_path / "gauge_hourly.csv", index=False)
    assert (tmp_path / "gauge_hourly.csv").read_text().splitlines()[
        1
    ] == "0012,2022-01-01 00:00:00+00:00,0.5"
    back = load_gauge_hourly(tmp_path / "gauge_hourly.csv")
    assert (
        back["station_id"].iloc[0] == "0012" and str(back["timestamp"].dt.tz) == "UTC"
    )

    st = pd.DataFrame(
        {
            "station_id": ["0012"],
            "station_no": ["563083"],
            "colocated_with": [""],
            "use_in_study": [True],
        }
    )
    st.to_csv(tmp_path / "stations.csv", index=False)
    back = load_stations(tmp_path / "stations.csv")
    assert back["station_id"].iloc[0] == "0012" and back["colocated_with"].iloc[0] == ""


def test_load_stations_keeps_numbers_numeric_and_blanks_empty(tmp_path):
    """`keep_default_na=False` is what makes an empty exclusion_reason "" (empty string)
    instead of NaN, but it also suppresses numeric inference on any column holding a blank cell.

    For example, if a station (gauge) doesn't have a tip increment to record in one
    period, the blank cell becomes "" instead of NaN, but the whole numeric column
    (thanks to keep_default_na = False) gets converted to a string
    """
    p = tmp_path / "stations.csv"
    p.write_text(
        "station_id,station_no,name,latitude,longitude,hours_observed,coverage,"
        "days_verified,days_rejected,tip_increment_mm,colocated_with,use_in_study,"
        "exclusion_reason\n"
        "0012,563083,Alpha,-33.8,151.1,17000,0.9705,700,10,0.5,,True,\n"
        "0013,563083,Beta,-33.6,150.2,0,0.0,0,0,,,False,no records in period\n"
    )
    st = load_stations(p)
    assert st["station_id"].tolist() == ["0012", "0013"]  # leading zero survives
    assert st["use_in_study"].dtype == bool
    assert st[st["use_in_study"]]["station_id"].tolist() == ["0012"]
    for col in ("latitude", "longitude", "coverage", "hours_observed"):
        assert pd.api.types.is_numeric_dtype(st[col]), col
    assert pd.api.types.is_numeric_dtype(st["tip_increment_mm"])
    assert st["tip_increment_mm"].iloc[0] == 0.5
    assert pd.isna(st["tip_increment_mm"].iloc[1])  # no tips, so no increment
    assert st["exclusion_reason"].tolist() == ["", "no records in period"]


# ---------------------------------------------------------------- the core invariant
def test_every_verified_days_hours_sum_to_its_verified_total(cfg):
    """The invariant the component exists to provide: an hour is written only when an
    independent BoM total backs it, so a verified day's 24 rows must add up to the tip
    total that was checked against BoM.

    Tips at exactly 09:00:00 local are the case that breaks this if the rain day and
    the hourly grid disagree by an hour: the day is [start, end) and the grid is 24
    buckets covering [t, t+1h), so a boundary tip must open the day whose grid it
    lands in. Before that was true, such a tip verified one day and was written into
    the next, or was dropped outright if the next day did not verify.
    """
    tips = tips_df(
        [
            ("s", "2022-01-05 09:00:00", 0.5, 10),  # opens the day ending 01-06 09:00
            ("s", "2022-01-05 23:30:00", 0.5, 10),  # mid-day
            ("s", "2022-01-06 08:59:59", 0.5, 10),  # closes that same day
            ("s", "2022-01-06 09:00:00", 0.5, 10),  # opens the following day
        ]
    )
    daily = daily_df(
        [
            ("s", "2022-01-06 09:00:00", 1.5, 10),
            ("s", "2022-01-07 09:00:00", 0.5, 10),
        ]
    )
    hourly, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    verified = days[days["verdict"] == "verified"]
    assert len(verified) == 2, days[["day_end", "tip_mm", "daily_mm", "verdict"]]
    for row in verified.itertuples():
        window = hourly[
            (hourly["timestamp"] >= row.day_end - pd.Timedelta(hours=24))
            & (hourly["timestamp"] < row.day_end)
        ]
        assert len(window) == 24, row.day_end
        assert window["rainfall_mm"].sum() == pytest.approx(row.tip_mm), row.day_end
    # and nothing leaked out of the two days altogether
    assert hourly["rainfall_mm"].sum() == pytest.approx(2.0)


def test_judgeable_hours_comes_from_the_period_not_the_data():
    """The coverage denominator. Deriving it from the rows that happened to verify
    lets one station's outage inflate every other station's coverage, and makes two
    runs over the same period incomparable."""
    assert judgeable_hours(date(2021, 1, 1), date(2022, 12, 31)) == 17_496
    assert judgeable_hours(date(2022, 1, 1), date(2022, 1, 31)) == 24 * 30
    # a period too short to hold one complete rain day judges nothing
    assert judgeable_hours(date(2022, 1, 1), date(2022, 1, 1)) == 0


def test_default_increment_is_configured_not_hard_coded(cfg):
    """A station with no positive tip has no increment to infer, so the comparison
    tolerance falls back to `default_tip_increment_mm`. The requirements doc forbids
    hard-coding 0.5 mm in the module, so the value has to change the verdict."""
    tips = tips_df([("s", "2022-01-05 12:00:00", 0.0, 10)])  # marker only
    daily = daily_df([("s", "2022-01-06 09:00:00", 0.4, 10)])
    args = (tips, daily, cfg["quality"], cfg["plausibility"])
    period = (date(2022, 1, 1), date(2022, 1, 31))
    _, loose, _ = verify_days(*args, *period, default_increment=0.5)
    _, tight, _ = verify_days(*args, *period, default_increment=0.1)
    assert loose["verdict"].tolist() == ["verified"]  # |0 - 0.4| <= 0.5
    assert tight["verdict"].tolist() == ["mismatch"]  # |0 - 0.4| >  0.1
    assert cfg["default_tip_increment_mm"] == 0.5


def test_days_table_carries_the_evidence_for_each_cap(cfg):
    """days.csv is the evidence behind every hour written or withheld, so the values
    the plausibility caps fired on have to be in it."""
    tips = tips_df([("s", "2022-01-05 12:00:00", 5.0, 10)])
    daily = daily_df([("s", "2022-01-06 09:00:00", 5.0, 10)])
    _, days, _ = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        date(2022, 1, 1),
        date(2022, 1, 31),
        default_increment=cfg["default_tip_increment_mm"],
    )
    assert {"max_record_mm", "max_hourly_mm"} <= set(days.columns)
    row = days.iloc[0]
    assert row["verdict"] == "implausible"  # BoM agrees, but 5 mm is not one tip
    assert row["max_record_mm"] == 5.0 and row["max_hourly_mm"] == 5.0


def test_period_from_index_fails_loudly_when_there_are_no_tip_rows():
    index = pd.DataFrame(
        [("gauge_daily", "1", "d", "ok", "")],
        columns=["source", "station_id", "path", "status", "error"],
    )
    with pytest.raises(ValueError, match="no gauge_tips rows"):
        raw.period_from_index(index)


def test_restrict_to_bounds_needs_the_ingestion_box(tmp_path):
    """The Sydney-box filter fails at startup, naming the key, without a box to use."""
    ingestion = tmp_path / "ingestion.yaml"
    ingestion.write_text("region: synthetic-test\n")
    settings = yaml.safe_load(CONFIG.read_text())
    settings.update(ingestion_config=str(ingestion), restrict_to_bounds=True)
    path = tmp_path / "standard.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=r"restrict_to_bounds needs gauge\.bounds"):
        load_config(path)
    settings["restrict_to_bounds"] = False
    path.write_text(yaml.safe_dump(settings))
    assert load_config(path)["restrict_to_bounds"] is False
