"""Draw every BoM rainfall gauge in Australia so a bounding box can be chosen by eye.

Helper for Component 01, not a prescribed output. Pulls the national station list
(one request) and, when a config is given, the national list of tip series (one more
request) so gauges can be coloured by whether they cover the study period. Overlays
the config's bounding box and counts what falls inside. Writes to database/survey/,
which is gitignored.

    python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml            # with the box
    python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml --no-box   # Australia only
    python -m src.ingestion.gauge.survey config/ingestion/sydney.yaml --show
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from src.ingestion.config import load_config
from src.ingestion.gauge.kiwis import KiwisClient

# Australia's real extent. WDO lists a few stations with corrupt coordinates
# (e.g. lat -85); they are kept in the CSV but flagged and not drawn.
AUS = {"min_lat": -44.0, "max_lat": -9.0, "min_lon": 112.0, "max_lon": 154.0}


def national_stations(client: KiwisClient, cfg: dict | None) -> pd.DataFrame:
    """All rainfall stations with coordinates, plus coverage of the tip series if a
    config names one. Coverage is BoM's advertised from/to, not verified data."""
    _, rows = client.station_list()
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["station_latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["station_longitude"], errors="coerce")
    df["coords_ok"] = df["lat"].between(AUS["min_lat"], AUS["max_lat"]) & df[
        "lon"
    ].between(AUS["min_lon"], AUS["max_lon"])

    if cfg:
        _, series = client.series_list(ts_name=cfg["gauge"]["series"]["tips"])
        s = pd.DataFrame(series)[["station_id", "ts_id", "from", "to"]]
        s["from"] = pd.to_datetime(s["from"], utc=True, errors="coerce")
        s["to"] = pd.to_datetime(s["to"], utc=True, errors="coerce")
        df = df.merge(s, on="station_id", how="left")
        start = pd.Timestamp(cfg["start"], tz="UTC")
        end = pd.Timestamp(cfg["end"], tz="UTC")
        df["covers_period"] = (df["from"] <= start) & (df["to"] >= end)
    return df


def in_box(df: pd.DataFrame, b: dict) -> pd.Series:
    return df["lat"].between(b["min_lat"], b["max_lat"]) & df["lon"].between(
        b["min_lon"], b["max_lon"]
    )


def draw(df: pd.DataFrame, cfg: dict | None, out_png: Path, show: bool) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[df["coords_ok"]]
    fig, ax = plt.subplots(figsize=(11, 9))
    if cfg:
        cov = ok["covers_period"].fillna(False)
        ax.scatter(
            ok.lon[~cov],
            ok.lat[~cov],
            s=4,
            c="lightgrey",
            label="no full coverage of period",
        )
        ax.scatter(
            ok.lon[cov], ok.lat[cov], s=5, c="black", label="covers study period"
        )
        b = cfg["gauge"]["bounds"]
        inside = in_box(ok, b)
        n, n_cov = int(inside.sum()), int((inside & cov).sum())
        ax.add_patch(
            plt.Rectangle(
                (b["min_lon"], b["min_lat"]),
                b["max_lon"] - b["min_lon"],
                b["max_lat"] - b["min_lat"],
                fill=False,
                ec="tab:red",
                lw=2,
                label=f"{cfg['region']} box: {n} gauges, {n_cov} cover period",
            )
        )
        ax.set_title(
            f"BoM rainfall gauges, {cfg['start']} to {cfg['end']} coverage, box from {cfg['_path']}"
        )
    else:
        ax.scatter(ok.lon, ok.lat, s=4, c="black", label="rainfall gauge")
        ax.set_title("BoM rainfall gauges in Water Data Online")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config", help="ingestion YAML (api settings always come from here)"
    )
    ap.add_argument("--out-dir", default="database/survey")
    ap.add_argument(
        "--show", action="store_true", help="also open an interactive window"
    )
    ap.add_argument(
        "--no-box", action="store_true", help="draw Australia without the config's box"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    client = KiwisClient(**cfg["gauge"]["api"])
    box_cfg = None if args.no_box else cfg
    df = national_stations(client, box_cfg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "australia_survey" if args.no_box else f"{cfg['region']}_survey"
    df.to_csv(out / f"{stem}.csv", index=False)
    draw(df, box_cfg, out / f"{stem}.png", args.show)

    print(
        f"stations listed: {len(df)}, with plausible coordinates: {int(df['coords_ok'].sum())}"
    )
    if box_cfg:
        inside = in_box(df[df["coords_ok"]], cfg["gauge"]["bounds"])
        print(
            f"inside {cfg['region']} box: {int(inside.sum())}, "
            f"of which {int((inside & df['covers_period'].fillna(False)).sum())} cover the study period"
        )
    print(f"wrote {out / (stem + '.csv')} and {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
