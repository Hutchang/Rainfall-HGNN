"""One row per station, including the rejected ones (requirements doc 02, job 4).

Rules that set use_in_study = false, each with an exclusion_reason:
  - no records in the period at all (the "silent" stations from Component 01);
  - coverage below min_coverage: too few verified hours over the period;
  - dead gauge: dead_gauge_days consecutive verified-dry days during which at least
    dead_gauge_neighbours stations within dead_gauge_radius_km recorded rain.

Co-located stations (closer than colocated_m) are NOT excluded here. They are flagged
in colocated_with so Component 03 can keep them in the same fold.
"""

import numpy as np
import pandas as pd

STATION_COLUMNS = [
    "station_id",
    "station_no",
    "name",
    "latitude",
    "longitude",
    "hours_observed",
    "coverage",
    "days_verified",
    "days_rejected",
    "tip_increment_mm",
    "colocated_with",
    "use_in_study",
    "exclusion_reason",
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def distance_matrix(st: pd.DataFrame) -> pd.DataFrame:
    lat, lon = st["latitude"].to_numpy(), st["longitude"].to_numpy()
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return pd.DataFrame(d, index=st["station_id"], columns=st["station_id"])


def dead_gauge_runs(days: pd.DataFrame, dist: pd.DataFrame, cfg: dict) -> dict:
    """{station_id: (run_start, run_end, n_days)} for the longest qualifying dry run.
    A run is consecutive calendar rain-days, all verified and all dry, during which at
    least `dead_gauge_neighbours` stations within `dead_gauge_radius_km` recorded rain
    on at least one day."""
    v = days[days["verdict"] == "verified"]
    wet = (
        v.pivot(index="day_end", columns="station_id", values="tip_mm") > 0
    )  # NaN -> False
    found = {}
    for sid in wet.columns:
        near = dist.loc[sid]
        near = near[
            (near > 0) & (near <= cfg["dead_gauge_radius_km"])
        ].index.intersection(wet.columns)
        wet_neighbours = wet[near].sum(axis=1) >= cfg["dead_gauge_neighbours"]
        s = v[v["station_id"] == sid].set_index("day_end")["tip_mm"].sort_index()
        dry = s == 0
        # break runs at any gap in consecutive days or any wet day
        gap = s.index.to_series().diff() != pd.Timedelta(days=1)
        run_id = (gap | ~dry).cumsum()
        best = None
        for _, grp in s[dry].groupby(run_id[dry]):
            if (
                len(grp) >= cfg["dead_gauge_days"]
                and wet_neighbours.reindex(grp.index).fillna(False).any()
            ) and (best is None or len(grp) > best[2]):
                best = (grp.index.min(), grp.index.max(), len(grp))
        if best:
            found[sid] = best
    return found


def build_station_table(
    stations: pd.DataFrame,
    hourly: pd.DataFrame,
    days: pd.DataFrame,
    increments: pd.Series,
    period_hours: int,
    cfg: dict,
) -> pd.DataFrame:
    st = stations.rename(columns={"station_name": "name"}).copy()
    hours = hourly.groupby("station_id").size().rename("hours_observed")
    verified = (
        days[days["verdict"] == "verified"]
        .groupby("station_id")
        .size()
        .rename("days_verified")
    )
    rejected = (
        days[days["verdict"].isin(["mismatch", "implausible"])]
        .groupby("station_id")
        .size()
        .rename("days_rejected")
    )
    st = st.merge(hours, on="station_id", how="left").merge(
        verified, on="station_id", how="left"
    )
    inc = increments.rename("tip_increment_mm").rename_axis("station_id").reset_index()
    st = st.merge(rejected, on="station_id", how="left").merge(
        inc, on="station_id", how="left"
    )
    for c in ("hours_observed", "days_verified", "days_rejected"):
        st[c] = st[c].fillna(0).astype(int)
    st["coverage"] = (st["hours_observed"] / period_hours).round(4)

    dist = distance_matrix(st)
    active = set(st.loc[st["hours_observed"] > 0, "station_id"])
    coloc = {}
    for sid in active:
        near = dist.loc[sid]
        partners = [
            p
            for p in near[(near > 0) & (near * 1000 <= cfg["colocated_m"])].index
            if p in active
        ]
        coloc[sid] = ";".join(sorted(partners))
    st["colocated_with"] = st["station_id"].map(coloc).fillna("")

    dead = dead_gauge_runs(days, dist, cfg)

    silent = st["hours_observed"] == 0
    low = st["coverage"] < cfg["min_coverage"]
    is_dead = st["station_id"].isin(dead)
    st["use_in_study"] = ~(silent | low | is_dead)
    st["exclusion_reason"] = np.select(
        [silent, low, is_dead],
        [
            "no records in period",
            "coverage " + st["coverage"].astype(str) + f" < {cfg['min_coverage']}",
            st["station_id"].map(
                lambda s: (
                    f"dead gauge: {dead[s][2]} verified-dry days {dead[s][0].date()} to {dead[s][1].date()} while neighbours recorded rain"
                    if s in dead
                    else ""
                )
            ),
        ],
        default="",
    )
    return st[STATION_COLUMNS].sort_values("station_id").reset_index(drop=True)
