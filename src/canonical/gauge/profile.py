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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill): categorical slots 1-3, text and surface tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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
    (out / "profile.md").write_text("\n".join(lines))
    print(f"wrote {out}/profile.md and 3 figures")


if __name__ == "__main__":
    main()
