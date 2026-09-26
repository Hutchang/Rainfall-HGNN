"""D13.1: confirm the program knows where the NCI radar/satellite archives are.

This does not read a single data file -- that is D13.2 (radar) and D13.3
(satellite). It only resolves each configured `layers.<name>.root` against the
filesystem and reports which calendar days, within the config's study period,
have an archive file whose name embeds that date. A layer with no `root`
configured is not checked: satellite's location is not confirmed yet (ISS-03),
so `config/canonical/grids.yaml` ships it as `null` rather than a guessed path.

A root under `/g/data/<project>/...` only resolves on Gadi, with membership of
that NCI project. A missing root there gets that pointed out explicitly, since
"file not found" alone reads as a typo rather than "wrong machine, or missing
access" -- see `_gadi_hint`.

The date is read from the filename, not the file's contents, so this works
without knowing a source's internal layout -- only that each archive file's
name contains its date as eight contiguous digits (`YYYYMMDD`), which holds for
the NCI radar archive's `<site>_<YYYYMMDD>.<product>.zip` and is assumed, not
yet confirmed, for satellite.
"""

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import yaml

_DATE_DIGITS = 8
_GDATA_PROJECT = re.compile(r"^/g/data/([^/]+)/")


def _gadi_hint(root: Path) -> str:
    """For a root under /g/data/<project>/, the reason it's likely missing and
    how to fix it; empty for any other path, where nothing this specific can
    be said."""
    match = _GDATA_PROJECT.match(str(root))
    if not match:
        return ""
    project = match.group(1)
    return (
        f" -- this path only exists on Gadi (NCI's supercomputer). This program "
        f"needs to be run on Gadi, and you need to be a member of NCI project "
        f"{project!r}; join at https://my.nci.org.au."
    )


def _dates_in_name(name: str) -> list[date]:
    """Every valid YYYYMMDD found among the runs of 8 digits in `name`."""
    found = []
    run = ""
    for char in name:
        run = run + char if char.isdigit() else ""
        if len(run) == _DATE_DIGITS:
            try:
                found.append(datetime.strptime(run, "%Y%m%d").date())
            except ValueError:
                pass
            run = ""
    return found


def find_archive_dates(root: Path) -> list[date]:
    """Every date named by a file under `root`, deduplicated and sorted.

    Recurses through whatever directory layout the archive uses; a file with no
    8-digit run in its name (a coordinate or manifest file, say) contributes
    nothing rather than raising.
    """
    dates = set()
    for path in root.rglob("*"):
        if path.is_file():
            dates.update(_dates_in_name(path.name))
    return sorted(dates)


def check_inputs(config: dict) -> str:
    """Confirm every configured archive root, and report the days found in the
    configured period for each. Raises FileNotFoundError naming every root that
    does not exist; checks all layers first so one report names every problem
    at once rather than stopping at the first.
    """
    start, end = config["start"], config["end"]
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    layers = config["layers"]
    configured = {
        name: Path(settings["root"])
        for name, settings in layers.items()
        if settings.get("root")
    }
    if not configured:
        return "No archive roots configured; nothing to check (see grids.yaml)."

    missing = [
        f"{name}: {root}{_gadi_hint(root)}"
        for name, root in configured.items()
        if not root.is_dir()
    ]
    if missing:
        raise FileNotFoundError("archive root not found for " + "; ".join(missing))

    lines = [f"# Archive input check: {start} to {end}", ""]
    for name, root in configured.items():
        found = [d for d in find_archive_dates(root) if start <= d <= end]
        by_year = sorted({d.year for d in found})
        total_days = (end - start).days + 1
        lines.append(f"## {name}")
        lines.append(f"root: {root}")
        lines.append(f"days found in range: {len(found)} of {total_days}")
        for year in by_year:
            count = sum(1 for d in found if d.year == year)
            lines.append(f"  {year}: {count}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="the only mode today; kept explicit for consistency with grids.py",
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    try:
        print(check_inputs(config))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
