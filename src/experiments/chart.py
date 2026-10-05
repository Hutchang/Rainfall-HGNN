"""Responsive terminal chart for experiment training history."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from rich.text import Text

HistoryRow = dict[str, float | int | str]

TRAIN_STYLE = "bold cyan"
VALIDATION_STYLE = "bold magenta"
OVERLAP_STYLE = "bold yellow"
GRID_STYLE = "dim"


def _line_character(previous: tuple[int, int], current: tuple[int, int]) -> str:
    if previous[1] == current[1]:
        return "─"
    return "/" if current[1] < previous[1] else "\\"


def _segment(start: tuple[int, int], end: tuple[int, int]):
    """Yield integer cells along a line using Bresenham's algorithm."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def loss_chart(
    rows: Sequence[HistoryRow],
    *,
    fold: int | None = None,
    width: int = 80,
    height: int = 20,
) -> Text:
    """Render a scaled, labelled train/validation line chart for one fold."""
    if not rows:
        return Text("Training history will appear here after the first epoch.")
    folds = sorted({int(row["fold"]) for row in rows})
    selected_fold = folds[-1] if fold not in folds else int(fold)
    points = sorted(
        (row for row in rows if int(row["fold"]) == selected_fold),
        key=lambda row: int(row["epoch"]),
    )
    width = max(width, 42)
    height = max(height, 12)
    label_width = 11
    plot_width = max(20, width - label_width - 1)
    plot_height = max(6, height - 5)

    values = [float(row[key]) for row in points for key in ("train_loss", "val_loss")]
    minimum, maximum = min(values), max(values)
    span = maximum - minimum
    padding = span * 0.08 if span else max(abs(maximum) * 0.08, 1e-9)
    y_min = max(0.0, minimum - padding)
    y_max = maximum + padding
    if y_max == y_min:
        y_max = y_min + 1.0

    first_epoch = int(points[0]["epoch"])
    last_epoch = int(points[-1]["epoch"])
    epoch_span = max(1, last_epoch - first_epoch)

    def coordinate(row: HistoryRow, key: str) -> tuple[int, int]:
        x = round((int(row["epoch"]) - first_epoch) / epoch_span * (plot_width - 1))
        ratio = (float(row[key]) - y_min) / (y_max - y_min)
        y = round((1.0 - ratio) * (plot_height - 1))
        return max(0, min(plot_width - 1, x)), max(0, min(plot_height - 1, y))

    grid: list[list[tuple[str, str]]] = [
        [(" ", "") for _ in range(plot_width)] for _ in range(plot_height)
    ]
    for row_index in (0, plot_height // 2, plot_height - 1):
        grid[row_index] = [("·", GRID_STYLE) for _ in range(plot_width)]

    def draw(key: str, style: str) -> None:
        coordinates = [coordinate(row, key) for row in points]
        for start, end in pairwise(coordinates):
            character = _line_character(start, end)
            for x, y in _segment(start, end):
                _existing, existing_style = grid[y][x]
                if existing_style and existing_style not in {GRID_STYLE, style}:
                    grid[y][x] = ("◆", OVERLAP_STYLE)
                else:
                    grid[y][x] = (character, style)
        for x, y in coordinates:
            _existing, existing_style = grid[y][x]
            if existing_style and existing_style not in {GRID_STYLE, style}:
                grid[y][x] = ("◆", OVERLAP_STYLE)
            else:
                grid[y][x] = ("●", style)

    draw("train_loss", TRAIN_STYLE)
    draw("val_loss", VALIDATION_STYLE)

    latest = points[-1]
    output = Text("Loss   ", style="bold")
    output.append("● Train", style=TRAIN_STYLE)
    output.append("   ")
    output.append("● Validation", style=VALIDATION_STYLE)
    output.append(
        f"   Fold {selected_fold}   Epoch {last_epoch}   "
        f"Train {float(latest['train_loss']):.6g}   "
        f"Validation {float(latest['val_loss']):.6g}\n"
    )

    tick_rows = {
        0: y_max,
        plot_height // 2: (y_max + y_min) / 2,
        plot_height - 1: y_min,
    }
    for row_index, row in enumerate(grid):
        label = (
            f"{tick_rows[row_index]:>9.4g} ┤"
            if row_index in tick_rows
            else " " * (label_width - 1) + "│"
        )
        output.append(label, style=GRID_STYLE)
        for character, style in row:
            output.append(character, style=style)
        output.append("\n")

    output.append(" " * (label_width - 1) + "└" + "─" * plot_width, style=GRID_STYLE)
    output.append("\n" + " " * label_width)
    if first_epoch == last_epoch:
        output.append(f"{first_epoch:^{plot_width}}")
    else:
        midpoint = (first_epoch + last_epoch) // 2
        left = str(first_epoch)
        middle = str(midpoint)
        right = str(last_epoch)
        spacing = max(1, (plot_width - len(left) - len(middle) - len(right)) // 2)
        labels = left + " " * spacing + middle
        labels += " " * max(1, plot_width - len(labels) - len(right)) + right
        output.append(labels[:plot_width])
    output.append("\n" + " " * label_width + f"{'Epoch':^{plot_width}}", style="bold")
    return output
