"""Recompute all reported scores from predictions.csv alone."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import KEY_COLUMNS, validate_prediction_table


def score_predictions(df):
    frame = validate_prediction_table(df)
    reference = None
    records = []
    for method, rows in frame.groupby("method", sort=True):
        if set(rows.fold) != set(range(5)):
            raise ValueError(f"{method}: completed scoring requires all five folds")
        truth = rows.set_index(KEY_COLUMNS).actual_mm.sort_index()
        if reference is None:
            reference = truth
        elif not truth.equals(reference):
            raise ValueError(
                "methods must be scored on identical target rows and actual_mm values"
            )
        groups = [
            ("overall", rows),
            *[(str(fold), values) for fold, values in rows.groupby("fold")],
        ]
        for fold, values in groups:
            error = values.predicted_mm.to_numpy() - values.actual_mm.to_numpy()
            actual = values.actual_mm.to_numpy()
            predicted = values.predicted_mm.to_numpy()
            correlation = float("nan")
            if len(values) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
                correlation = float(np.corrcoef(actual, predicted)[0, 1])
            records.append(
                {
                    "method": method,
                    "fold": fold,
                    "n": len(values),
                    "rmse_mm": float(np.sqrt(np.mean(error**2))),
                    "mae_mm": float(np.mean(np.abs(error))),
                    "correlation": correlation,
                }
            )
    return pd.DataFrame(records)


def write_report(path, df, settings, notes=()):
    scores = score_predictions(df)
    lines = ["# Model run", ""]
    if settings.get("fixture_label"):
        lines += [
            "**SYNTHETIC SOFTWARE TEST — not a scientific rainfall result.**",
            "",
            str(settings["fixture_label"]),
            "",
        ]
    lines += [
        "Target validation must pass before this run is complete.",
        "",
        "| Method | Fold | Rows | RMSE (mm) | MAE (mm) | Correlation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in scores.itertuples(index=False):
        correlation = (
            f"{row.correlation:.6g}"
            if np.isfinite(row.correlation)
            else "undefined (constant or too few values)"
        )
        lines.append(
            f"| {row.method} | {row.fold} | {row.n} | {row.rmse_mm:.6g} | {row.mae_mm:.6g} | {correlation} |"
        )
    lines += [
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(settings, indent=2, default=str),
        "```",
    ]
    if notes:
        lines += ["", "## Notes", "", *[f"- {note}" for note in notes]]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.models.evaluate experiments/<run>/predictions.csv"
        )
    frame = pd.read_csv(sys.argv[1], dtype={"station_id": str})
    print(score_predictions(frame).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
