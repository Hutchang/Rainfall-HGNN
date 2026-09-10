"""Profile Component 01's raw gauge data before cleaning: what is there, what quality
BoM assigns it, and where the stations are. Writes figures and a Markdown summary
under <output_dir>/profile/. Grows "after" views once the build exists.

    uv run python -m src.canonical.gauge.profile config/canonical/standard.yaml
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.canonical.config import load_config
from src.canonical.gauge import raw
from src.canonical.gauge.build import load_stations
from src.canonical.gauge.verify import verify_days

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill): categorical slots 1-3, text and surface tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"  # status colour, reserved for rejected data
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#b3b2ad", "#fcfcfb", "#e8e7e3"
QUALITY_NAMES = {
    10: "10 A best",
    90: "90 B compromised",
    110: "110 C estimate",
    130: "130 no rainfall",
    140: "140 E unknown",
    210: "210 F missing/incorrect",
    -1: "-1 (null value)",
    254: "254 (undocumented)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "grid.color": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def fig_quality(tips: pd.DataFrame, out: Path) -> dict:
    """Horizontal bars: tip records per BoM quality code, with share and station count."""
    g = tips.groupby("quality").agg(
        records=("value", "size"), stations=("station_id", "nunique")
    )
    g = g.sort_values("records")
    share = g["records"] / g["records"].sum()
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(g) + 1.5))
    y = np.arange(len(g))
    ax.barh(y, g["records"], color=BLUE, height=0.55)
    ax.set_yticks(y, [QUALITY_NAMES.get(q, str(q)) for q in g.index])
    ax.set_xscale("log")
    ax.set_xlabel("tip records (log scale)")
    ax.grid(axis="x", alpha=0.6)
    for yi, (n, s, st) in enumerate(zip(g["records"], share, g["stations"])):
        ax.text(
            n * 1.15,
            yi,
            f"{n:,}  ({s:.1%}, {st} stations)",
            va="center",
            color=INK2,
            fontsize=9,
        )
    ax.set_xlim(right=g["records"].max() * 8)
    ax.set_title(
        "Raw tip records by BoM quality code", loc="left", color=INK, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out / "quality_codes.png", dpi=140)
    plt.close(fig)
    return {
        int(q): (int(n), float(s), int(st))
        for q, n, s, st in zip(g.index, g["records"], share, g["stations"])
    }


def station_status(index: pd.DataFrame) -> pd.Series:
    """complete / partial / silent from the 01 index: months with and without records."""
    t = index[index["source"] == "gauge_tips"].groupby("station_id")["n_records"]
    empty, total = t.apply(lambda s: (s == 0).sum()), t.size()
    return pd.Series(
        np.where(empty == 0, "complete", np.where(empty == total, "silent", "partial")),
        index=empty.index,
    )


def fig_map(
    stations: pd.DataFrame,
    status: pd.Series,
    bounds: dict,
    out: Path,
    name: str,
    title: str,
    classes: dict,
) -> None:
    """Zoomed map of the box. `classes` maps label -> (colour, marker); status gives each station's label."""
    df = stations.set_index("station_id").join(status.rename("status"))
    fig, ax = plt.subplots(figsize=(9, 8))
    for label, (colour, marker) in classes.items():
        sub = df[df["status"] == label]
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=34,
            c=colour,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=0.8,
            label=f"{label} ({len(sub)})",
            zorder=3,
        )
    b = bounds
    ax.add_patch(
        plt.Rectangle(
            (b["min_lon"], b["min_lat"]),
            b["max_lon"] - b["min_lon"],
            b["max_lat"] - b["min_lat"],
            fill=False,
            ec=INK2,
            lw=1,
            ls="--",
            zorder=2,
        )
    )
    pad = 0.05
    ax.set_xlim(b["min_lon"] - pad, b["max_lon"] + pad)
    ax.set_ylim(b["min_lat"] - pad, b["max_lat"] + pad)
    ax.set_aspect(1 / np.cos(np.radians((b["min_lat"] + b["max_lat"]) / 2)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.6)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=140)
    plt.close(fig)


def fig_random_day(
    tips: pd.DataFrame, daily: pd.DataFrame, cfg: dict, out: Path, seed: int = 123
) -> dict:
    """One rain day at one station: cumulative counted tips through the day against BoM's
    daily total. The dry-versus-missing rule, drawn. Seeded so the report is reproducible."""
    q = cfg["quality"]
    rng = np.random.default_rng(seed)
    wet = daily[(daily["value"] > 5) & daily["quality"].isin(q["daily"]["trusted"])]
    pick = wet.iloc[rng.integers(len(wet))]
    sid, day_end, total = pick["station_id"], pick["timestamp"], float(pick["value"])
    day_start = day_end - pd.Timedelta(hours=24)
    t = tips[
        (tips["station_id"] == sid)
        & (tips["timestamp"] > day_start)
        & (tips["timestamp"] <= day_end)
        & tips["quality"].isin(q["tips"]["count"])
        & (tips["value"] > 0)
    ].sort_values("timestamp")
    cum = t["value"].cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(
        t["timestamp"], cum, where="post", color=BLUE, lw=2, label="tips, cumulative"
    )
    ax.axhline(
        total, color=ORANGE, lw=2, ls="--", label=f"BoM daily total {total:.1f} mm"
    )
    ax.set_xlim(day_start, day_end)
    ax.set_ylabel("mm")
    ax.set_xlabel("UTC")
    ax.grid(alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    local_day = (day_end - pd.Timedelta(hours=10)).strftime("%Y-%m-%d")
    verdict = "verified" if abs(cum.iloc[-1] - total) <= 0.5 else "NOT verified"
    ax.set_title(
        f"Station {sid}, rain day ending 09:00 local {local_day}: "
        f"{len(t)} tips = {cum.iloc[-1]:.1f} mm, {verdict}",
        loc="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "random_day.png", dpi=140)
    plt.close(fig)
    return {
        "station_id": sid,
        "day_end_utc": str(day_end),
        "tips": len(t),
        "tips_mm": float(cum.iloc[-1]),
        "daily_mm": total,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Diagnostics: the figures that justify the day-level rules
# ---------------------------------------------------------------------------


def fig_value_profile(
    tips: pd.DataFrame, quality: dict, cap: float, names: pd.Series, out: Path
) -> pd.DataFrame:
    """One bar per station: its largest single counted record, sorted. A tipping bucket
    never records more than its increment, so bars sit at 0.5 (or 0.2, 0.1, 1.0).
    Anything above the cap is a logger storing totals as if they were tips."""
    t = tips[tips["quality"].isin(quality["tips"]["count"]) & (tips["value"] > 0)]
    prof = t.groupby("station_id")["value"].agg(
        distinct="nunique", max_record="max", records="size"
    )
    prof = prof.sort_values(["max_record", "distinct"])
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(prof))
    over = prof["max_record"] > cap
    ax.bar(x[~over], prof["max_record"][~over], color=BLUE, width=0.8)
    ax.bar(x[over], prof["max_record"][over], color=CRITICAL, width=0.8)
    ax.axhline(cap, color=CRITICAL, lw=1, ls="--")
    ax.text(0, cap * 1.25, f"max_record_mm = {cap}", color=CRITICAL, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("largest single record, mm (log)")
    ax.set_xlabel(f"{len(prof)} stations with counted records, sorted")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.6)
    ax.text(
        x[over][0] - 1,
        prof["max_record"].max(),
        f"{int(over.sum())} stations above the cap, listed in profile.md",
        ha="right",
        va="top",
        fontsize=9,
        color=INK2,
    )
    ax.set_ylim(top=prof["max_record"].max() * 3)
    ax.set_title(
        "Is it a tipping bucket? Largest single record per station",
        loc="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "value_profile.png", dpi=140)
    plt.close(fig)
    return prof


def fig_daily_totals(days: pd.DataFrame, plaus: dict, out: Path) -> None:
    """Every station-day with rain: date against tip total, coloured by verdict. Outliers
    sit alone above the caps; non-tipping stations scatter oddly; floods are columns."""
    wet = days[days["tip_mm"] > 0]
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = [
        ("verified", BLUE, 6, 0.25),
        ("mismatch", ORANGE, 14, 0.9),
        ("implausible", CRITICAL, 18, 0.9),
    ]
    for verdict, colour, size, alpha in layers:
        sub = wet[wet["verdict"] == verdict]
        ax.scatter(
            sub["day_end"],
            sub["tip_mm"],
            s=size,
            c=colour,
            alpha=alpha,
            linewidths=0,
            label=f"{verdict} ({len(sub):,})",
            zorder=3,
        )
    ax.axhline(plaus["max_daily_mm"], color=CRITICAL, lw=1, ls="--", zorder=2)
    ax.text(
        wet["day_end"].min(),
        plaus["max_daily_mm"] * 1.15,
        f"max_daily_mm = {plaus['max_daily_mm']}",
        color=CRITICAL,
        fontsize=9,
    )
    ax.set_yscale("log")
    ax.set_ylabel("rain-day total from tips, mm (log)")
    ax.grid(alpha=0.6)
    ax.legend(loc="upper left", frameon=False, markerscale=1.5)
    ax.set_title(
        "Every station rain-day, 2021-2022, by verdict",
        loc="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "daily_totals.png", dpi=140)
    plt.close(fig)


def fig_heatmap(days: pd.DataFrame, stations: pd.DataFrame, out: Path) -> None:
    """Station by rain-day, one panel per year. Cell colour is the verdict: blues scaled
    by rainfall for verified days, orange mismatch, red implausible, grey unverifiable.
    Silent stations are omitted; rows sorted by share of verified days."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    d = days.copy()
    # code: 0 unverifiable, 1 mismatch, 2 implausible, 3.. verified with rainfall bins
    bins = [
        0,
        0.01,
        1,
        5,
        20,
        50,
        np.inf,
    ]  # dry, trace, light, moderate, heavy, extreme
    rain_bin = np.digitize(d["daily_mm"].fillna(0), bins[1:], right=False)
    d["code"] = np.select(
        [
            d["verdict"] == "unverifiable",
            d["verdict"] == "mismatch",
            d["verdict"] == "implausible",
        ],
        [0, 1, 2],
        default=3 + rain_bin,
    )
    keep = d.groupby("station_id")["verdict"].apply(lambda v: (v == "verified").any())
    d = d[d["station_id"].isin(keep[keep].index)]
    share = d.groupby("station_id")["verdict"].apply(lambda v: (v == "verified").mean())
    order = share.sort_values(ascending=False).index
    grid = d.pivot(index="station_id", columns="day_end", values="code").reindex(order)
    blues = [
        matplotlib.colormaps["Blues"](v) for v in (0.08, 0.25, 0.42, 0.6, 0.78, 0.98)
    ]
    cmap = ListedColormap([MUTED, ORANGE, CRITICAL, *blues])
    years = sorted({c.year for c in grid.columns})
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 5.2 * len(years)))
    for ax, year in zip(np.atleast_1d(axes), years):
        cols = [c for c in grid.columns if c.year == year]
        sub = grid[cols]
        ax.imshow(
            sub.to_numpy(dtype=float),
            aspect="auto",
            cmap=cmap,
            vmin=-0.5,
            vmax=8.5,
            interpolation="nearest",
        )
        ticks = [i for i, c in enumerate(cols) if c.day == 1 or (i == 0)]
        ax.set_xticks(ticks, [cols[i].strftime("%b") for i in ticks], fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel(f"{len(sub)} stations, most complete at top")
        ax.set_title(str(year), loc="left", color=INK, fontsize=11)
    legend = [
        Patch(color=MUTED, label="unverifiable (no BoM total)"),
        Patch(color=ORANGE, label="mismatch"),
        Patch(color=CRITICAL, label="implausible"),
    ]
    legend += [
        Patch(
            color=c,
            label=f"verified, {lo}" + (f"-{hi} mm" if np.isfinite(hi) else "+ mm"),
        )
        for c, lo, hi in zip(blues, bins[:-1], bins[1:])
    ]
    legend[3] = Patch(color=blues[0], label="verified, dry")
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Station by rain-day: verdict and rainfall (silent stations omitted)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "heatmap.png", dpi=140)
    plt.close(fig)


def fig_stations_after(
    standard_dir: Path, bounds: dict, out: Path
) -> pd.DataFrame | None:
    """The 'after' map: stations.csv from the build, coloured by use_in_study and reason."""
    path = standard_dir / "stations.csv"
    if not path.exists():
        return None
    st = load_stations(path)
    reason = (
        st["exclusion_reason"]
        .str.split(":")
        .str[0]
        .str.replace(r"coverage .*", "coverage below threshold", regex=True)
    )
    status = pd.Series(
        np.where(st["use_in_study"].astype(str) == "True", "use_in_study", reason),
        index=st["station_id"],
    )
    classes = {
        "use_in_study": (BLUE, "o"),
        "coverage below threshold": (ORANGE, "o"),
        "dead gauge": (CRITICAL, "D"),
        "no records in period": (MUTED, "s"),
    }
    fig_map(
        st.rename(columns={"name": "station_name"}),
        status,
        bounds,
        out,
        "stations_after",
        "Stations after Component 02 rules",
        classes,
    )
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = Path(cfg["raw_dir"])
    out = Path(cfg["output_dir"]) / "profile"
    out.mkdir(parents=True, exist_ok=True)
    with open(cfg["ingestion_config"]) as fh:
        bounds = yaml.safe_load(fh)["gauge"]["bounds"]

    index = raw.read_index(raw_dir)
    stations = raw.read_stations(raw_dir)
    print("reading tips...", flush=True)
    tips = raw.read_tips(raw_dir, index)
    daily = raw.read_daily(raw_dir, index)
    print(
        f"{len(tips):,} tip records, {len(daily):,} daily records, {len(stations)} stations",
        flush=True,
    )

    qstats = fig_quality(tips, out)
    status = station_status(index)
    fig_map(
        stations,
        status,
        bounds,
        out,
        "stations_raw",
        "Stations in the box, by records returned for 2021-2022",
        {"complete": (BLUE, "o"), "partial": (ORANGE, "o"), "silent": (MUTED, "s")},
    )
    day = fig_random_day(tips, daily, cfg, out)

    start, end = raw.period_from_index(index)
    hourly, days, inc = verify_days(
        tips,
        daily,
        cfg["quality"],
        cfg["plausibility"],
        start,
        end,
        default_increment=cfg["default_tip_increment_mm"],
    )
    names = stations.set_index("station_id")["station_name"]
    prof = fig_value_profile(
        tips, cfg["quality"], cfg["plausibility"]["max_record_mm"], names, out
    )
    fig_daily_totals(days, cfg["plausibility"], out)
    fig_heatmap(days, stations, out)
    verdicts = days["verdict"].value_counts()
    non_tip = prof[prof["max_record"] > cfg["plausibility"]["max_record_mm"]]

    counts = status.value_counts()
    lines = [
        "# Raw gauge data profile",
        "",
        f"Source: `{raw_dir}` (Component 01 output). Config: `{cfg['_path']}`.",
        "",
        "## What is there",
        "",
        f"- stations in box: {len(stations)}",
        f"- complete (records every month): {counts.get('complete', 0)}",
        f"- partial: {counts.get('partial', 0)}",
        f"- silent (no records at all): {counts.get('silent', 0)}",
        f"- tip records: {len(tips):,}; daily records: {len(daily):,}",
        "",
        "![stations](stations_raw.png)",
        "",
        "## BoM quality codes on tip records",
        "",
        "| Code | Records | Share | Stations | Policy |",
        "|---|---|---|---|---|",
    ]
    q = cfg["quality"]["tips"]
    policy = {
        **{c: "counted, then verified daily" for c in q["count"]},
        **{c: "ignored" for c in q["ignore"]},
        **{c: "gap marker" for c in q["gap"]},
    }
    for code, (n, s, st) in sorted(qstats.items(), key=lambda kv: -kv[1][0]):
        lines.append(
            f"| {QUALITY_NAMES.get(code, code)} | {n:,} | {s:.1%} | {st} | {policy.get(code, 'UNDECIDED')} |"
        )
    lines += [
        "",
        "![quality](quality_codes.png)",
        "",
        "## One rain day, the verification rule drawn",
        "",
        (
            f"Station {day['station_id']}, rain day ending {day['day_end_utc']}: "
            f"{day['tips']} counted tips = {day['tips_mm']:.1f} mm against BoM's daily "
            f"total {day['daily_mm']:.1f} mm: **{day['verdict']}**."
        ),
        "",
        "![day](random_day.png)",
        "",
    ]
    lines += [
        "## Day verification (dry versus missing) and plausibility",
        "",
        f"Period {start} to {end}. {len(days):,} station rain-days judged.",
        "",
        "| Verdict | Days |",
        "|---|---|",
    ]
    lines += [f"| {v} | {n:,} |" for v, n in verdicts.items()]
    imp = days[days["verdict"] == "implausible"]["reason"].value_counts()
    lines += [
        "",
        "Implausible by reason: " + ", ".join(f"{r} ({n})" for r, n in imp.items()),
        "",
        (
            f"Hourly rows produced: {len(hourly):,} across {hourly['station_id'].nunique()} stations; "
            f"max hourly value {hourly['rainfall_mm'].max():.1f} mm."
        ),
        "",
        "Tip increment per station: "
        + ", ".join(
            f"{v} mm at {n}" for v, n in inc.value_counts().sort_index().items()
        ),
        "",
        "![daily totals](daily_totals.png)",
        "",
        f"Stations with a single record above {cfg['plausibility']['max_record_mm']} mm (not tip-resolution on those days):",
        "",
    ]
    lines += [
        f"- {sid} {names.get(sid, '')}: {int(r['distinct'])} distinct values, largest record {r['max_record']:.1f} mm"
        for sid, r in non_tip.sort_values("max_record", ascending=False).iterrows()
    ]
    lines += [
        "",
        "![value profile](value_profile.png)",
        "",
        "![heatmap](heatmap.png)",
        "",
    ]
    after = fig_stations_after(Path(cfg["output_dir"]), bounds, out)
    if after is not None:
        n_use = int(after["use_in_study"].sum())
        lines += [
            "## After the station rules",
            "",
            f"{n_use} of {len(after)} stations have use_in_study = true (from `stations.csv`).",
            "",
            "![after](stations_after.png)",
            "",
        ]
    (out / "profile.md").write_text("\n".join(lines))
    print(f"wrote {out}/profile.md and 6 figures")


if __name__ == "__main__":
    main()
