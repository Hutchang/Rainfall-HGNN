"""Load and validate a Component 01 ingestion config (config/ingestion/<region>.yaml).

One file per region, one section per source. Shared by every source's fetcher and
by the gauge survey.
"""

from datetime import date
from pathlib import Path

import yaml

REQUIRED = ("region", "start", "end", "output_dir", "gauge")
GAUGE_REQUIRED = ("bounds", "api", "series")


def load_config(path: str | Path) -> dict:
    """Read the YAML and fail with a concrete message if a required key is missing."""
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)

    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")

    if not (isinstance(cfg["start"], date) and isinstance(cfg["end"], date)):
        raise TypeError(f"{path}: start and end must be YYYY-MM-DD dates")
    if cfg["start"] > cfg["end"]:
        raise ValueError(f"{path}: start {cfg['start']} is after end {cfg['end']}")

    g = cfg["gauge"]
    missing = [k for k in GAUGE_REQUIRED if k not in g]
    if missing:
        raise ValueError(f"{path}: gauge section is missing {missing}")

    b = g["bounds"]
    for k in ("min_lat", "max_lat", "min_lon", "max_lon"):
        if k not in b:
            raise ValueError(f"{path}: gauge.bounds is missing {k}")
    if not (b["min_lat"] < b["max_lat"] and b["min_lon"] < b["max_lon"]):
        raise ValueError(f"{path}: gauge.bounds min must be less than max: {b}")

    for k in ("tips", "daily"):
        if k not in g["series"]:
            raise ValueError(f"{path}: gauge.series is missing {k}")

    r = cfg.get("radar") or {}
    if r:
        for k in ("site_id", "product", "api"):
            if k not in r:
                raise ValueError(f"{path}: radar section is missing {k}")
        if "base_url" not in r["api"]:
            raise ValueError(f"{path}: radar.api is missing base_url")

    cfg["_path"] = str(path)
    return cfg
