"""Load and validate a Component 02 config (config/canonical/standard.yaml)."""

from pathlib import Path

import yaml

REQUIRED = ("raw_dir", "output_dir", "min_coverage", "dead_gauge_days", "quality")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if not 0 < cfg["min_coverage"] <= 1:
        raise ValueError(f"{path}: min_coverage must be in (0, 1]")
    q = cfg["quality"]
    for section, keys in (
        ("tips", ("count", "ignore", "gap")),
        ("daily", ("trusted", "dry", "unverifiable")),
    ):
        if section not in q or any(k not in q[section] for k in keys):
            raise ValueError(f"{path}: quality.{section} needs keys {keys}")
    cfg["_path"] = str(path)
    return cfg
