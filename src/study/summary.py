"""Human-readable summary and fold map for a Component 03 split (ISS-18).

Written beside the split ``.json`` as ``<name>.md`` and ``<name>.png`` so a reviewer
can check a split by looking at it. Both are derived from the split and the station
table alone and can be regenerated at any time.
"""

from collections.abc import Iterable
from math import cos, radians
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schema import SplitConfig, StationRecord, StudySplitOutput

# Okabe-Ito colours: distinguishable with the common colour-vision deficiencies.
FOLD_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
ROLE_STYLES = {
    "train": {"color": "#BBBBBB", "marker": "o", "label": "train"},
    "validation": {"color": "#E69F00", "marker": "s", "label": "validation"},
    "test": {"color": "#0072B2", "marker": "^", "label": "hidden (test)"},
}


def write_split_summary(
    output: StudySplitOutput,
    stations: Iterable[StationRecord],
    config: SplitConfig,
    split_path: Path,
) -> tuple[Path, Path]:
    """Write ``<name>.md`` and ``<name>.png`` beside the split and return both paths."""
    positions = {
        station.station_id: (station.longitude, station.latitude)
        for station in stations
        if station.use_in_study
    }
    figure_path = split_path.with_suffix(".png")
    summary_path = split_path.with_suffix(".md")
    _draw_fold_map(output, positions, config, figure_path)
    summary_path.write_text(
        _summary_text(output, config, figure_path.name), encoding="utf-8"
    )
    return summary_path, figure_path


def _summary_text(output: StudySplitOutput, config: SplitConfig, figure: str) -> str:
    lines = [
        f"# Split summary: {output.name}",
        "",
        f"- stations file: `{output.stations_file}`",
        f"- eligible stations: {output.n_eligible}; excluded: {len(output.excluded)}",
        f"- folds: {config.n_folds}; clusters: {config.n_clusters}; "
        f"fold assignment: `{config.fold_assignment}`",
        f"- validation fraction: {config.validation_fraction}; seed: {output.seed}",
        f"- created: {output.created}",
        "",
        "## Stations per fold",
        "",
        "| Fold | Train | Validation | Hidden (test) |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {fold.fold_index} | {len(fold.ml.train)} | {len(fold.ml.validation)} "
        f"| {len(fold.ml.test)} |"
        for fold in output.folds
    )
    lines.extend(
        [
            "",
            "## Fold map",
            "",
            f"![Fold map for {output.name}]({figure})",
            "",
            "Left: every eligible station, coloured by the fold that hides it. "
            "Right: fold 0's roles.",
            "",
        ]
    )
    return "\n".join(lines)


def _draw_fold_map(
    output: StudySplitOutput,
    positions: dict[str, tuple[float, float]],
    config: SplitConfig,
    path: Path,
) -> None:
    fig, (by_fold, roles) = plt.subplots(
        1, 2, figsize=(13, 6), sharex=True, sharey=True
    )

    for fold in output.folds:
        points = [positions[station_id] for station_id in fold.ml.test]
        by_fold.scatter(
            [x for x, _ in points],
            [y for _, y in points],
            s=28,
            color=FOLD_COLOURS[fold.fold_index % len(FOLD_COLOURS)],
            label=f"fold {fold.fold_index} ({len(points)})",
        )
    by_fold.set_title("Which fold hides each station")
    by_fold.legend(title="hidden in", fontsize=8)

    first = output.folds[0]
    for role, style in ROLE_STYLES.items():
        points = [positions[station_id] for station_id in getattr(first.ml, role)]
        roles.scatter(
            [x for x, _ in points],
            [y for _, y in points],
            s=28,
            color=style["color"],
            marker=style["marker"],
            label=f"{style['label']} ({len(points)})",
        )
    roles.set_title(f"Fold {first.fold_index} roles")
    roles.legend(fontsize=8)

    mean_latitude = sum(y for _, y in positions.values()) / len(positions)
    for axis in (by_fold, roles):
        # One degree of longitude is shorter than one of latitude away from the
        # equator; this keeps distances on the page proportional to kilometres.
        axis.set_aspect(1 / cos(radians(mean_latitude)))
        axis.set_xlabel("longitude")
        axis.grid(alpha=0.3)
    by_fold.set_ylabel("latitude")
    fig.suptitle(
        f"{output.name}: {output.n_eligible} stations, {config.n_folds} folds, "
        f"{config.fold_assignment}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
