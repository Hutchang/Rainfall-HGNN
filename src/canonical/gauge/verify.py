"""Dry versus missing: verify each station's rain days against BoM's daily totals and
turn the verified ones into hourly rows.

Requirements doc 02, "The three rules that carry the science", rule 1. For each
station and BoM rain day (09:00 to 09:00 local standard time, UTC+10):

  - count the tips whose quality code is in the config's `count` list;
  - compare their sum with BoM's own daily total for that day;
  - if they agree within one tip increment the day is VERIFIED and all 24 hours are
    written, 0.0 where no tips fell;
  - otherwise nothing is written for that day and the reason is recorded.

A missing hour is therefore an absent row. Component 04 reads absence as the
model's validity flag, so this function decides what the model treats as known.
When in doubt, reject: a wrongly verified day puts confident zeros into the model,
a wrongly rejected day only makes it slightly blinder.

Timestamps in and out are timezone-aware UTC. BoM's fixed +10:00 offset means the
rain-day boundary is always 23:00 UTC of the previous calendar day.
"""

from datetime import date

import numpy as np
import pandas as pd

LOCAL_OFFSET = pd.Timedelta(hours=10)  # BoM serves fixed UTC+10, no daylight saving
RAIN_DAY_END_HOUR = 9  # local

DAY_COLUMNS = [
    "station_id",
    "day_end",  # UTC timestamp of the end of the rain day (09:00 local)
    "n_tips",
    "tip_mm",
    "daily_mm",
    "daily_quality",
    "verdict",  # verified | mismatch | unverifiable | no_daily_record | implausible
    "reason",  # for implausible: record>max | hour>max | day>max
]


def rain_day_end(ts_utc: pd.Series) -> pd.Series:
    """The UTC timestamp of the 09:00-local rain-day boundary at or after each time.
    
    BoM records rain-days from 09:00 -> 09:00, not midnight --> midnight. Every tip needs 
    a label saying which rain day it falls in and we've chosen the rain day's end.

    Pandas can only round to natural boundaries, so .ceil("D") 'round up to the next
    day' will round to midnight, not 09:00.

    This function shifts back 9 hours so pandas can round correctly and 09:00 will
    correctly map to a midnight, or a new day. 
    
    A tip at exactly 09:00:00 belongs to the day ending then: the day is (start, end]."""
    local = ts_utc + LOCAL_OFFSET
    shifted = local - pd.Timedelta(hours=RAIN_DAY_END_HOUR)

    # days = 1 offset to advance to the next day boundary. We take the floor of the 
    # current day (shifted 9 hours back) and THEN advance by a day, putting us in TODAY's
    # bucket, not tomorrow's

    end_local = shifted.dt.floor("D") + pd.Timedelta(days=1, hours=RAIN_DAY_END_HOUR)
    return end_local - LOCAL_OFFSET


def tip_increments(tips: pd.DataFrame) -> pd.Series:
    """Millimetres per tip, per station: the smallest positive counted value seen.
    BoM's series metadata does not state it (see docs/ingestion/README.md)."""
    pos = tips[tips["value"] > 0]
    return pos.groupby("station_id")["value"].min().rename("tip_increment_mm")


def verify_days(
    tips: pd.DataFrame,
    daily: pd.DataFrame,
    quality: dict,
    plausibility: dict,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns (hourly, days, increments).

    hourly: station_id, timestamp (UTC), rainfall_mm, one row per verified hour.
    days:   one row per station-day that could be judged, with its verdict.
    increments: mm per tip per station.

    A day that passes the daily check but breaks a plausibility cap (a single record,
    an hour, or the day above the configured maximum) is "implausible" and rejected:
    BoM's total is computed from the same tips, so a stuck gauge fools both sides.

    Only rain days whose full 24 hours lie inside [start, end] are judged. The
    period's first 9 and last 15 local hours belong to rain days that straddle the
    boundary and are never written.
    """
    q_tips, q_daily = quality["tips"], quality["daily"]

    # ---- tips: keep counted codes with a value; markers (0.0) contribute nothing
    t = tips[tips["quality"].isin(q_tips["count"]) & tips["value"].notna()].copy()
    increments = tip_increments(t)
    t["day_end"] = rain_day_end(t["timestamp"])
    by_day = (
        t[t["value"] > 0]
        .groupby(["station_id", "day_end"])
        .agg(n_tips=("value", "size"), tip_mm=("value", "sum"))
    )

    # ---- daily totals: BoM's record for the rain day ending at this timestamp
    d = daily.copy()
    d["day_end"] = d["timestamp"]
    trusted = d["quality"].isin(q_daily["trusted"]) & d["value"].notna()
    dry = d["quality"].isin(q_daily["dry"])
    d["daily_mm"] = np.where(trusted, d["value"], np.where(dry, 0.0, np.nan))
    d["daily_ok"] = trusted | dry
    d = d.rename(columns={"quality": "daily_quality"})
    d = d.set_index(["station_id", "day_end"])[
        ["daily_mm", "daily_quality", "daily_ok"]
    ]

    # ---- judge every day that has a daily record or tips, inside the period
    first_end = (
        pd.Timestamp(start, tz="UTC")
        + pd.Timedelta(hours=24)
        - LOCAL_OFFSET
        + pd.Timedelta(hours=RAIN_DAY_END_HOUR)
    )
    last_end = (
        pd.Timestamp(end, tz="UTC")
        + pd.Timedelta(hours=24)
        - LOCAL_OFFSET
        + pd.Timedelta(hours=RAIN_DAY_END_HOUR)
        - pd.Timedelta(days=1)
    )
    days = d.join(by_day, how="outer").reset_index()
    days = days[(days["day_end"] >= first_end) & (days["day_end"] <= last_end)].copy()
    days["n_tips"] = days["n_tips"].fillna(0).astype(int)
    days["tip_mm"] = days["tip_mm"].fillna(0.0)
    days["daily_ok"] = days["daily_ok"].fillna(False).astype(bool)
    days = days.merge(increments, on="station_id", how="left")
    tol = days["tip_increment_mm"].fillna(
        0.5
    )  # a station with no tips at all: assume 0.5

    # per-day maxima for the plausibility caps
    pos = t[t["value"] > 0].copy()
    pos["hour"] = pos["timestamp"].dt.floor("h")
    max_record = (
        pos.groupby(["station_id", "day_end"])["value"].max().rename("max_record_mm")
    )
    max_hour = (
        pos.groupby(["station_id", "day_end", "hour"])["value"]
        .sum()
        .groupby(["station_id", "day_end"])
        .max()
        .rename("max_hourly_mm")
    )
    days = days.merge(max_record, on=["station_id", "day_end"], how="left")
    days = days.merge(max_hour, on=["station_id", "day_end"], how="left")

    no_daily = days["daily_quality"].isna()
    agree = (days["tip_mm"] - days["daily_mm"]).abs() <= tol
    bad_record = days["max_record_mm"] > plausibility["max_record_mm"]
    bad_hour = days["max_hourly_mm"] > plausibility["max_hourly_mm"]
    bad_day = days["tip_mm"] > plausibility["max_daily_mm"]
    days["verdict"] = np.select(
        [no_daily, ~days["daily_ok"], bad_record | bad_hour | bad_day, agree],
        ["no_daily_record", "unverifiable", "implausible", "verified"],
        default="mismatch",
    )
    days["reason"] = np.select(
        [bad_record, bad_hour, bad_day],
        [
            "record>" + str(plausibility["max_record_mm"]),
            "hour>" + str(plausibility["max_hourly_mm"]),
            "day>" + str(plausibility["max_daily_mm"]),
        ],
        default="",
    )
    days = (
        days[DAY_COLUMNS + ["tip_increment_mm"]]
        .sort_values(["station_id", "day_end"])
        .reset_index(drop=True)
    )

    # ---- hourly rows for verified days: 24 per day, tips summed by UTC hour, else 0.0
    ok = days[days["verdict"] == "verified"][["station_id", "day_end"]]
    t_ok = t[t["value"] > 0].merge(ok, on=["station_id", "day_end"], how="inner")
    t_ok["timestamp"] = t_ok["timestamp"].dt.floor("h")
    wet = t_ok.groupby(["station_id", "timestamp"])["value"].sum().rename("rainfall_mm")

    starts = (ok["day_end"] - pd.Timedelta(hours=24)).to_numpy(dtype="datetime64[ns]")
    grid = pd.DataFrame(
        {
            "station_id": np.repeat(ok["station_id"].to_numpy(), 24),
            "timestamp": (
                starts[:, None] + np.arange(24) * np.timedelta64(1, "h")
            ).ravel(),
        }
    )
    grid["timestamp"] = grid["timestamp"].dt.tz_localize("UTC")
    hourly = grid.merge(wet, on=["station_id", "timestamp"], how="left")
    hourly["rainfall_mm"] = hourly["rainfall_mm"].fillna(0.0)
    hourly = hourly.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return hourly, days, increments
