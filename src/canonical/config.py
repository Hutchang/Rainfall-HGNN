"""Load and validate a Component 02 config (config/canonical/standard.yaml).

Every key the build reads is checked here, so a missing or nonsensical setting fails
at startup with the key named rather than as a KeyError halfway through a run.
"""

from pathlib import Path

import yaml

REQUIRED = (
    "raw_dir",
    "output_dir",
    "min_coverage",
    "dead_gauge_days",
    "dead_gauge_radius_km",
    "dead_gauge_neighbours",
    "colocated_m",
    "default_tip_increment_mm",
    "quality",
    "plausibility",
)


def _check_gauge_bounds(path: Path, ingestion_config: str | None) -> None:
    """restrict_to_bounds reads the box from the ingestion config's gauge.bounds."""
    keys = ("min_lat", "max_lat", "min_lon", "max_lon")
    if not ingestion_config:
        raise ValueError(f"{path}: restrict_to_bounds needs ingestion_config")
    with open(ingestion_config) as fh:
        bounds = ((yaml.safe_load(fh) or {}).get("gauge") or {}).get("bounds") or {}
    if any(k not in bounds for k in keys):
        raise ValueError(
            f"{path}: restrict_to_bounds needs gauge.bounds with {keys} in "
            f"{ingestion_config}; set restrict_to_bounds: false to build every station"
        )


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if not 0 < cfg["min_coverage"] <= 1:
        raise ValueError(f"{path}: min_coverage must be in (0, 1]")
    if not cfg["default_tip_increment_mm"] > 0:
        raise ValueError(
            f"{path}: default_tip_increment_mm must be positive; it is the comparison "
            "tolerance for a station with no positive tip to infer an increment from"
        )
    q = cfg["quality"]
    for section, keys in (
        ("tips", ("count", "ignore", "gap")),
        ("daily", ("trusted", "dry", "unverifiable")),
    ):
        if section not in q or any(k not in q[section] for k in keys):
            raise ValueError(f"{path}: quality.{section} needs keys {keys}")
    for k in ("max_record_mm", "max_hourly_mm", "max_daily_mm"):
        if k not in cfg["plausibility"]:
            raise ValueError(f"{path}: plausibility is missing {k}")
    if cfg.get("restrict_to_bounds"):
        _check_gauge_bounds(path, cfg.get("ingestion_config"))
    cfg["_path"] = str(path)
    return cfg
