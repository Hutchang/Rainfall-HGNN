"""Join fold-subset training runs (`run.py --folds`) into one five-fold best.pt.

    python -m src.models.combine_folds OUT_DIR RUN_DIR [RUN_DIR ...]

Writes OUT_DIR/best.pt and OUT_DIR/history.csv. Score the result with
`python -m src.models.run CONFIG --predict-from OUT_DIR/best.pt`, which checks the
inputs and settings again and predicts every method on the full target set.

Each fold must come from exactly one run, every run must share the same input
hashes and model settings, and a fold counts only if its training finished: early
stopping triggered or the last epoch was reached. A fold cut off by a walltime limit
is refused rather than scored from a part-trained checkpoint.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

SHARED_SETTINGS = (
    "model_seed",
    "hidden_channels",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay",
    "loss",
    "weighted_loss_alpha",
    "batch_size",
    "epochs",
    "early_stop_patience",
    "input_rainfall_scale",
    "graphs_dir",
    "split_file",
)


def finished_folds(history, epochs):
    done = set()
    for fold, rows in history.groupby("fold"):
        if (rows.early_stop_status == "TRIGGERED").any() or rows.epoch.max() >= epochs:
            done.add(int(fold))
    return done


def combine(out_dir, run_dirs):
    folds, histories, scaling = {}, [], {}
    reference = None
    for run_dir in map(Path, run_dirs):
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        history = pd.read_csv(run_dir / "history.csv")
        settings = best["settings"]
        if reference is None:
            reference = best
        if best["inputs"] != reference["inputs"]:
            raise ValueError(f"{run_dir}: input hashes differ from {run_dirs[0]}")
        for key in SHARED_SETTINGS:
            if settings.get(key) != reference["settings"].get(key):
                raise ValueError(f"{run_dir}: setting {key} differs from {run_dirs[0]}")
        if set(settings.get("input_layers", ["raingauge"])) != set(
            reference["settings"].get("input_layers", ["raingauge"])
        ):
            raise ValueError(f"{run_dir}: input_layers differ from {run_dirs[0]}")
        done = finished_folds(history, settings["epochs"])
        for index, checkpoint in best["folds"].items():
            if index not in done:
                raise ValueError(
                    f"{run_dir}: fold {index} did not finish training (no early stop "
                    "and fewer than the configured epochs); rerun it"
                )
            if index in folds:
                raise ValueError(f"fold {index} appears in more than one run")
            folds[index] = checkpoint
            scaling[index] = settings.get("input_scaling", {}).get(index)
        histories.append(history[history.fold.isin(best["folds"])])
    if set(folds) != set(range(5)):
        raise ValueError(f"folds {sorted(set(range(5)) - set(folds))} are missing")
    settings = dict(reference["settings"])
    settings.pop("fold_subset", None)
    if settings.get("input_rainfall_scale") == "zscore_train":
        settings["input_scaling"] = scaling
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "format": "component06-best-v1",
            "folds": dict(sorted(folds.items())),
            "settings": settings,
            "inputs": reference["inputs"],
            "combined_from": [str(Path(d).resolve()) for d in run_dirs],
        },
        out_dir / "best.pt",
    )
    pd.concat(histories).sort_values(["fold", "epoch"]).to_csv(
        out_dir / "history.csv", index=False
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        print(combine(args.out_dir, args.run_dirs))
    except (ValueError, OSError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
