"""Read-only config helpers for the Gadi pipeline jobs (docs/nci/README.md, section 6).

    python scripts/nci/pipeline_config.py get         CONFIG KEY
    python scripts/nci/pipeline_config.py check-grids CONFIG
    python scripts/nci/pipeline_config.py trial       CONFIG START:END DIRECTORY

`check-grids` exists because the grid build reads the whole archive, hours of work,
before it looks at its own outputs; the job runs it first so a clash or a missing
archive mount stops the job at once. `trial` writes a copy of a grid config with a
shorter period and its own output folder, to time the readers and check the paths.
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml


def read(config_path):
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def grid_problems(config_path):
    """Everything that would stop a grid build, found without reading the archive."""
    config = read(config_path)
    output = Path(config["output_dir"])
    targets = [output / "grid_report.md"] + [
        output / f"{layer}_{suffix}.csv"
        for layer in config["layers"]
        for suffix in ("hourly", "nodes")
    ]
    problems = [f"{path} already exists" for path in targets if path.exists()]
    for layer, settings in config["layers"].items():
        root = settings.get("root")
        if root and not Path(root).is_dir():
            problems.append(
                f"{layer} archive root {root} is not readable here; check project "
                "membership (rq0, rv74) and the job's storage directive"
            )
    return problems


def write_trial(config_path, period, directory):
    """A copy of the grid config covering `period` (START:END, inclusive), writing
    into `directory`. Returns the copy's path."""
    try:
        start, end = (datetime.date.fromisoformat(day) for day in period.split(":"))
    except ValueError:
        raise ValueError(
            f"trial period must be START:END dates, got {period!r}"
        ) from None
    if end < start:
        raise ValueError(f"trial period ends before it starts: {period}")
    config = read(config_path)
    config.update(start=start, end=end, output_dir=str(directory))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "grids.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("get")
    get.add_argument("config")
    get.add_argument("key")
    check = commands.add_parser("check-grids")
    check.add_argument("config")
    trial = commands.add_parser("trial")
    trial.add_argument("config")
    trial.add_argument("period")
    trial.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "get":
        config = read(args.config)
        if args.key not in config:
            print(f"BLOCKED: {args.config} has no {args.key}", file=sys.stderr)
            return 1
        print(config[args.key])
    elif args.command == "check-grids":
        problems = grid_problems(args.config)
        for problem in problems:
            print(f"BLOCKED: {problem}", file=sys.stderr)
        return 1 if problems else 0
    else:
        try:
            print(write_trial(args.config, args.period, args.directory))
        except (ValueError, FileExistsError) as error:
            print(f"BLOCKED: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
