"""Component 02, gauge half: build the standard gauge tables from Component 01's raw files.

    uv run python -m src.canonical.gauge.build config/canonical/standard.yaml

Writes to <output_dir>:
  gauge_hourly.csv   station_id, timestamp (UTC, with offset), rainfall_mm; one row per
                     verified hour. A missing hour is an absent row.
  stations.csv       one row per station in the box, rejected ones included, with
                     use_in_study and exclusion_reason.
  days.csv           one row per station rain-day with its verdict (for review).
  report.md          counts, exclusions with cause, rejected days, run settings.

Two conventions every reader depends on (requirements doc 02):
  - timestamps are written as ISO8601 with the UTC offset, e.g. 2022-01-01 00:00:00+00:00;
  - station_id must be read as a string: pd.read_csv(path, dtype={"station_id": str}).
Use load_gauge_hourly / load_stations below rather than remembering that.
"""

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.canonical.config import load_config
from src.canonical.gauge import raw
from src.canonical.gauge.stations import build_station_table
from src.canonical.gauge.verify import verify_days

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%z"


def load_gauge_hourly(path: Path) -> pd.DataFrame:
    """The loader every downstream component should use."""
    df = pd.read_csv(path, dtype={"station_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_stations(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"station_id": str, "station_no": str, "colocated_with": str},
        keep_default_na=False,
    )


def period_from_index(index: pd.DataFrame) -> tuple:
    t = index[index["source"] == "gauge_tips"]
    return pd.Timestamp(t["start"].min()).date(), pd.Timestamp(t["end"].max()).date()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - best effort provenance only
        return "unknown"


def _fmt_ts(s: pd.Series) -> pd.Series:
    out = s.dt.strftime(TIMESTAMP_FORMAT)
    return out.str.replace(r"\+0000$", "+00:00", regex=True)


def run(cfg: dict) -> dict:
    t0 = time.time()
    raw_dir, out = Path(cfg["raw_dir"]), Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    index = raw.read_index(raw_dir)
    stations = raw.read_stations(raw_dir)
    tips = raw.read_tips(raw_dir, index)
    daily = raw.read_daily(raw_dir, index)
    start, end = period_from_index(index)
    print(
        f"read {len(tips):,} tips, {len(daily):,} daily records, {len(stations)} stations; period {start} to {end}",
        flush=True,
    )

    hourly, days, increments = verify_days(
        tips, daily, cfg["quality"], cfg["plausibility"], start, end
    )
    period_hours = int(hourly["timestamp"].nunique())
    st = build_station_table(stations, hourly, days, increments, period_hours, cfg)

    # ---- write, with the two conventions
    h = hourly.copy()
    h["timestamp"] = _fmt_ts(h["timestamp"])
    h.to_csv(out / "gauge_hourly.csv", index=False, float_format="%.2f")
    st.to_csv(out / "stations.csv", index=False)
    d = days.copy()
    d["day_end"] = _fmt_ts(d["day_end"])
    d.to_csv(out / "days.csv", index=False, float_format="%.2f")

    summary = write_report(
        out / "report.md",
        cfg,
        start,
        end,
        period_hours,
        st,
        days,
        hourly,
        increments,
        time.time() - t0,
    )
    print(
        f"wrote {out}/gauge_hourly.csv ({len(hourly):,} rows), stations.csv ({len(st)} rows), days.csv, report.md",
        flush=True,
    )
    return summary


def write_report(
    path, cfg, start, end, period_hours, st, days, hourly, increments, seconds
) -> dict:
    usable = st[st["use_in_study"]]
    excluded = st[~st["use_in_study"]]
    verdicts = days["verdict"].value_counts()
    judged = int(days["daily_quality"].notna().sum())
    imp = days[days["verdict"] == "implausible"]
    reasons = (
        excluded["exclusion_reason"]
        .str.split(":")
        .str[0]
        .str.replace(r"coverage .*", "coverage below threshold", regex=True)
        .value_counts()
    )
    lines = [
        "# Standard gauge dataset report",
        "",
        f"- config: `{cfg['_path']}`; code revision `{git_revision()}`; built {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} in {seconds:.0f} s",
        f"- source: `{cfg['raw_dir']}` (Component 01 output)",
        (
            f"- period: {start} to {end}; judgeable window {period_hours:,} hours per station "
            "(rain days fully inside the period; the first 9 and last 15 local hours are never written)"
        ),
        f"- quality codes counted as tips: {cfg['quality']['tips']['count']}; ignored: {cfg['quality']['tips']['ignore']}; gap markers: {cfg['quality']['tips']['gap']}",
        f"- daily totals trusted: {cfg['quality']['daily']['trusted']}; dry: {cfg['quality']['daily']['dry']}; unverifiable: {cfg['quality']['daily']['unverifiable']}",
        f"- plausibility caps: record {cfg['plausibility']['max_record_mm']} mm, hour {cfg['plausibility']['max_hourly_mm']} mm, day {cfg['plausibility']['max_daily_mm']} mm",
        f"- station rules: min_coverage {cfg['min_coverage']}, dead gauge {cfg['dead_gauge_days']} days with {cfg['dead_gauge_neighbours']} wet neighbours within {cfg['dead_gauge_radius_km']} km",
        "",
        "## Stations",
        "",
        f"- in box: {len(st)}",
        f"- use_in_study: {len(usable)}",
        f"- excluded: {len(excluded)}",
    ]
    lines += [f"  - {r}: {n}" for r, n in reasons.items()]
    lines += [
        f"- co-located pairs flagged for Component 03: {int((st['colocated_with'] != '').sum()) // 2}",
        "- tip increment inferred per station: "
        + ", ".join(
            f"{v} mm at {n}" for v, n in increments.value_counts().sort_index().items()
        ),
        "",
        "## Days",
        "",
        f"- station rain-days judged: {len(days):,}",
    ]
    lines += [f"- {v}: {n:,}" for v, n in verdicts.items()]
    lines += [
        f"- verified share of days with a usable BoM total: {verdicts.get('verified', 0) / judged:.1%}",
        "- implausible by cap: "
        + ", ".join(f"{r} ({n})" for r, n in imp["reason"].value_counts().items()),
        "",
        "## Hours",
        "",
        f"- rows in gauge_hourly.csv: {len(hourly):,} across {hourly['station_id'].nunique()} stations (all stations with verified days, usable or not)",
        f"- rows at use_in_study stations: {int(hourly['station_id'].isin(usable['station_id']).sum()):,}",
        f"- wet hours: {int((hourly['rainfall_mm'] > 0).sum()):,}; max hourly value {hourly['rainfall_mm'].max():.1f} mm",
        "",
        "## Excluded stations, with cause",
        "",
    ]
    lines += [
        f"- {r.station_id} {r['name']}: {r.exclusion_reason}"
        for _, r in excluded.iterrows()
    ]
    lines += ["", "## Rejected days (mismatch or implausible), per station", ""]
    rej = (
        days[days["verdict"].isin(["mismatch", "implausible"])]
        .groupby(["station_id", "verdict"])
        .size()
        .unstack(fill_value=0)
    )
    names = st.set_index("station_id")["name"]
    lines += [
        f"- {sid} {names.get(sid, '')}: "
        + ", ".join(f"{c} {n}" for c, n in row.items() if n)
        for sid, row in rej.sort_values(
            rej.columns.tolist(), ascending=False
        ).iterrows()
    ]
    lines += ["", "Full per-day verdicts are in `days.csv`.", ""]
    path.write_text("\n".join(lines))
    return {
        "usable": len(usable),
        "excluded": len(excluded),
        "hourly_rows": len(hourly),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    run(load_config(args.config))
    sys.exit(0)


if __name__ == "__main__":
    main()
