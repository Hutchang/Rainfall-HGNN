"""Result summaries and portable plot exports for experiment workspaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .workspace import Experiment, load_history, read_status

HistoryRow = dict[str, float | int | str]


def _overall_metric_rows(report: Path) -> list[str]:
    """Return completed ``overall`` metric rows from a model Markdown report."""
    if not report.is_file():
        return []
    rows = []
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1].lower() == "overall":
            rows.append("| " + " | ".join(cells) + " |")
    return rows


def results_summary(
    experiment: Experiment,
    rows: Sequence[HistoryRow] | None = None,
    *,
    selected_fold: int | None = None,
    state: str | None = None,
) -> str:
    """Build the compact Markdown summary displayed below the live chart."""
    history = list(load_history(experiment) if rows is None else rows)
    run_state = state or str(read_status(experiment).get("state", "UNKNOWN"))
    lines = ["### Results summary", "", f"**Run state:** `{run_state}`"]

    grouped: dict[int, list[HistoryRow]] = defaultdict(list)
    for row in history:
        grouped[int(row["fold"])].append(row)
    for fold_rows in grouped.values():
        fold_rows.sort(key=lambda row: int(row["epoch"]))

    if grouped:
        lines += [
            "",
            "| Fold | Epochs | Best val loss | Latest train | Latest val | "
            "Early stopping |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        for fold, fold_rows in sorted(grouped.items()):
            latest = fold_rows[-1]
            best = min(fold_rows, key=lambda row: float(row["val_loss"]))
            early_stop = str(latest.get("early_stop_status", "—"))
            deterioration = latest.get("deteriorations")
            if deterioration is not None:
                early_stop += f" ({deterioration} deteriorations)"
            marker = " ← selected" if fold == selected_fold else ""
            lines.append(
                f"| {fold}{marker} | {len(fold_rows)} | "
                f"{float(best['val_loss']):.6g} (epoch {int(best['epoch'])}) | "
                f"{float(latest['train_loss']):.6g} | "
                f"{float(latest['val_loss']):.6g} | {early_stop} |"
            )
    else:
        lines += ["", "No training history has been recorded yet."]

    metric_rows = _overall_metric_rows(experiment.root / "model" / "report.md")
    if metric_rows:
        lines += [
            "",
            "#### Final prediction metrics",
            "",
            "| Method | Fold | Rows | RMSE (mm) | MAE (mm) | Correlation |",
            "|---|---|---:|---:|---:|---:|",
            *metric_rows,
        ]
    elif history and run_state != "DONE":
        lines += [
            "",
            "Final RMSE, MAE and correlation are available only after a completed run.",
        ]
    return "\n".join(lines)


def _available_export_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 2
    while (directory / f"{stem}-{index}{suffix}").exists():
        index += 1
    return directory / f"{stem}-{index}{suffix}"


def save_loss_plot(experiment: Experiment, *, fold: int | None = None) -> Path:
    """Save one fold's training history as a non-overwriting Matplotlib PNG."""
    rows = load_history(experiment)
    folds = sorted({int(row["fold"]) for row in rows})
    if not folds:
        raise ValueError("No training history is available to plot")
    selected_fold = folds[-1] if fold is None else fold
    if selected_fold not in folds:
        raise ValueError(f"Fold {selected_fold} has no training history")
    points = sorted(
        (row for row in rows if int(row["fold"]) == selected_fold),
        key=lambda row: int(row["epoch"]),
    )

    # Use Matplotlib's object API and an off-screen canvas. This avoids pyplot state and
    # works in terminal-only sessions such as Gadi compute nodes.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    epochs = [int(row["epoch"]) for row in points]
    axes.plot(
        epochs,
        [float(row["train_loss"]) for row in points],
        color="#0891b2",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Train",
    )
    axes.plot(
        epochs,
        [float(row["val_loss"]) for row in points],
        color="#c026d3",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Validation",
    )
    axes.set_title(f"{experiment.name} — fold {selected_fold}")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.grid(alpha=0.25)
    axes.legend()

    directory = experiment.root / "results"
    directory.mkdir(parents=True, exist_ok=True)
    last_epoch = epochs[-1]
    target = _available_export_path(
        directory,
        f"train-validation-loss-fold-{selected_fold}-through-epoch-{last_epoch}.png",
    )
    figure.savefig(target, dpi=160)
    figure.clear()
    return target
