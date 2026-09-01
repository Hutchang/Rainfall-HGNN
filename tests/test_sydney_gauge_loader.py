"""T-001 — Sydney gauge loader.

Verifies the loader contract for Niramay's bundled Sydney gauge data:
non-empty records, consistent timestamp parsing, finite coordinates,
documented rainfall validity rules, and stable station IDs.

Findings encoded here are documented in
docs/archive/baseline-reconciliation/AUDIT_NIRAMAY.md sections 8 and 9.

Tests marked xfail record known defects in the inherited hand export and metadata.
The new source/canonical components replace that data rather than mutating this legacy
regression fixture.
"""

import re

import numpy as np
import pandas as pd
import pytest

# ── Records and schema ───────────────────────────────────────────────────────


def test_raw_files_present(raw_files):
    assert len(raw_files) == 173, f"expected 173 station files, found {len(raw_files)}"


def test_schema_is_stable(gauge_long):
    assert list(gauge_long.columns) == ["timestamp", "station_id", "rainfall_mm"]


def test_records_non_empty(gauge_long):
    assert len(gauge_long) > 0
    assert gauge_long["rainfall_mm"].notna().any()


def test_filename_matches_in_file_station_id(raw_files):
    """The station id in each file must match its filename."""
    bad = []
    for f in raw_files:
        sid = re.search(r"(\d+)_rainfall_hourly\.csv$", f).group(1)
        inside = set(pd.read_csv(f, nrows=50)["station_id"].astype(str))
        if inside != {sid}:
            bad.append((f, inside))
    assert not bad, f"filename/in-file station_id mismatch: {bad[:3]}"


# ── Timestamps ───────────────────────────────────────────────────────────────


def test_timestamps_parse_consistently(gauge_long):
    assert gauge_long["timestamp"].notna().all(), "unparseable timestamps present"


def test_temporal_cadence_is_hourly(gauge_long):
    """The inherited Sydney observations are hourly and must not be rescaled."""
    one = gauge_long[gauge_long["station_id"] == gauge_long["station_id"].iloc[0]]
    deltas = one.sort_values("timestamp")["timestamp"].diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(hours=1)], f"non-hourly cadence: {deltas[:5]}"


def test_timestamps_are_timezone_naive(gauge_long):
    """Documents the inherited source defect; Component 02 replaces it with UTC."""
    assert gauge_long["timestamp"].dt.tz is None


def test_no_duplicate_station_time_records(gauge_long):
    """A duplicate (timestamp, station_id) breaks pivot() in BOTH loaders."""
    dup = gauge_long.duplicated(subset=["timestamp", "station_id"]).sum()
    assert dup == 0, f"{dup} duplicate (timestamp, station_id) rows"


def test_pivot_succeeds(gauge_wide):
    assert gauge_wide.shape[0] == 43_824
    assert gauge_wide.shape[1] == 173


# ── Rainfall validity ────────────────────────────────────────────────────────


def test_no_negative_rainfall(gauge_long):
    neg = (gauge_long["rainfall_mm"] < 0).sum()
    assert neg == 0, f"{neg} negative rainfall values"


def test_missing_rate_is_low(gauge_long):
    rate = gauge_long["rainfall_mm"].isna().mean()
    assert rate < 0.01, f"missing rate {rate:.2%} exceeds 1%"


def test_uptime_filter_is_a_noop(gauge_wide):
    """Every station exceeds the 0.9 threshold, so the filter has never been
    exercised. AUDIT_NIRAMAY.md section 8.5."""
    uptime = gauge_wide.notna().mean(axis=0)
    assert (uptime >= 0.9).all()
    assert uptime.min() > 0.99


@pytest.mark.xfail(
    reason="LEGACY DATA REGRESSION — AUDIT_NIRAMAY.md section 8.2 supersession "
    "note. Direct BoM comparison proved time- and station-varying export "
    "distortion; no global scale factor can repair rainfall_mm. Components "
    "01/02 replace it with source-faithful tips and verified true millimetres.",
    strict=False,
)
def test_units_are_plausibly_millimetres(gauge_long):
    """Sydney climatology is ~1210 mm/yr; western stations 700-900 mm/yr.
    A plausible mm dataset should not exceed ~2500 mm/yr median."""
    yearly = (
        gauge_long.assign(y=gauge_long["timestamp"].dt.year)
        .groupby(["station_id", "y"])["rainfall_mm"]
        .sum()
    )
    median_annual = yearly.groupby("station_id").mean().median()
    assert median_annual < 2500, (
        f"median station annual total {median_annual:,.0f} is not consistent "
        f"with millimetres for Sydney"
    )


# ── Station identity and coordinates ─────────────────────────────────────────


def test_all_data_stations_have_metadata(gauge_wide, station_meta):
    data_ids = {str(c) for c in gauge_wide.columns}
    assert data_ids - set(station_meta["id"]) == set()


def test_coordinates_are_finite(station_meta):
    assert np.isfinite(station_meta["latitude"]).all()
    assert np.isfinite(station_meta["longitude"]).all()


def test_coordinates_are_in_greater_sydney(station_meta):
    assert station_meta["latitude"].between(-35.0, -33.0).all()
    assert station_meta["longitude"].between(149.0, 152.0).all()


@pytest.mark.xfail(
    reason="LEGACY DATA REGRESSION — AUDIT_NIRAMAY.md section 8.1. station_metadata.csv holds "
    "425 rows for 409 unique ids; data/loader.py:53 filters with .isin() "
    "but never de-duplicates, so the graph gets 188 nodes for 173 real "
    "stations. Components 01/02 key the replacement data on unique BoM "
    "internal station_id.",
    strict=False,
)
def test_station_ids_are_unique_after_filtering(station_meta, gauge_wide):
    """One metadata row per station with data."""
    assert len(station_meta) == gauge_wide.shape[1], (
        f"{len(station_meta)} metadata rows for {gauge_wide.shape[1]} stations "
        f"-> {len(station_meta) - gauge_wide.shape[1]} phantom graph nodes"
    )


@pytest.mark.xfail(
    reason="LEGACY PUBLIC-ID REGRESSION — AUDIT_NIRAMAY.md section 8.1. Public "
    "numbers 563083 and 567163 each identify two real stations. The BoM "
    "API provides distinct internal station_id values, which components "
    "01/02 use instead of de-duplicating public numbers.",
    strict=False,
)
def test_duplicate_ids_refer_to_the_same_place(station_meta):
    """Where an id appears twice, both rows must describe the same location.
    Ids failing this are collisions needing external (BOM) resolution."""
    bad = []
    for sid, grp in station_meta[
        station_meta.duplicated(subset=["id"], keep=False)
    ].groupby("id"):
        r = grp.reset_index(drop=True)
        km = np.hypot(
            (r.latitude[0] - r.latitude[1]) * 111.0,
            (r.longitude[0] - r.longitude[1])
            * 111.0
            * np.cos(np.radians(r.latitude.mean())),
        )
        if km > 1.0:
            bad.append((sid, round(km, 2), list(r["name"])))
    assert not bad, f"id collisions between different stations: {bad}"
