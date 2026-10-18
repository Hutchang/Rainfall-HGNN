"""Run Component 06 against supplied canonical tables, splits and graph samples."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import torch_geometric
import yaml

from src.models.baselines import predict_baseline
from src.models.contracts import (
    load_inputs,
    target_rows,
    utc_hours,
    validate_predictions,
)
from src.models.evaluate import write_report
from src.models.graphs import load_graph, validate_graph
from src.models.modalities import validate_layer_selection
from src.models.scaling import apply_rainfall_scaling, rainfall_scaling
from src.models.scope import (
    measure_sensor_layers,
    measure_tables,
    record_graph_nodes,
    scope_lines,
)
from src.models.trainer import model_spec, predict_fold, restore_model, train_fold

INPUT_KEYS = ("gauge_file", "stations_file", "split_file", "seeds_file")
ROLES = ("train", "validation", "test")
HISTORY_COLUMNS = (
    "fold",
    "epoch",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "deteriorations",
    "patience_remaining",
    "early_stop_status",
)
# physical_mm feeds the model raw millimetres (inherited). zscore_train scales the
# rainfall input per node type on each fold's training graph (paper sec 4.3; D08.2).
RAINFALL_SCALES = ("physical_mm", "zscore_train")


def read_settings(path):
    with Path(path).open() as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise TypeError("model config must contain a YAML mapping")
    return settings


def validate_settings(settings):
    method = settings.get("method")
    if method not in ("hgnn", "idw", "network_mean"):
        raise ValueError("method must be hgnn, idw or network_mean")
    if settings.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be explicitly cpu or cuda")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but this PyTorch environment has no usable CUDA device"
        )
    for key in INPUT_KEYS:
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f"missing input path config.{key}")
    if method == "hgnn":
        validate_layer_selection(settings.get("input_layers", ["raingauge"]))
        for key in (
            "hidden_channels",
            "num_layers",
            "batch_size",
            "epochs",
            "early_stop_patience",
            "cpu_threads",
        ):
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("dropout", "learning_rate", "weight_decay", "weighted_loss_alpha"):
            value = settings.get(key)
            if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                raise ValueError(f"{key} must be finite and nonnegative")
        if settings["learning_rate"] == 0 or settings["dropout"] >= 1:
            raise ValueError(
                "learning_rate must be positive and dropout must be less than 1"
            )
        if settings.get("loss") not in ("weighted_mse", "mse"):
            raise ValueError("loss must be weighted_mse or mse")
        if type(settings.get("deterministic")) is not bool:
            raise ValueError("deterministic must be explicitly true or false")
        if settings.get("input_rainfall_scale") not in RAINFALL_SCALES:
            raise ValueError(
                "input_rainfall_scale must be physical_mm or zscore_train; graph files "
                "stay in physical millimetres either way"
            )
        if not isinstance(settings.get("graphs_dir"), str):
            raise ValueError("missing input path config.graphs_dir")
        baselines = settings.get("baselines")
        if (
            not isinstance(baselines, list)
            or set(baselines) != {"idw", "network_mean"}
            or len(baselines) != 2
        ):
            raise ValueError(
                "hgnn runs must include baselines: [idw, network_mean] for comparison"
            )
    if method in ("hgnn", "idw"):
        power = settings.get("idw_power")
        if not isinstance(power, (int, float)) or not 0 < power < float("inf"):
            raise ValueError("idw_power must be explicitly finite and positive")


def dropped_hours_file(settings):
    """Component 04's list of hours it left out of the graphs (DEC-01), if any.

    Graph folders built before DEC-01 have no such file and drop no hours.
    """
    if not isinstance(settings.get("graphs_dir"), str):
        return None
    path = Path(settings["graphs_dir"]) / "dropped_hours.csv"
    return path if path.is_file() else None


def drop_hours(gauge, settings):
    """Remove the hours Component 04 dropped, so every method in the run is
    scored on the same hours as the graphs (DEC-01)."""
    path = dropped_hours_file(settings)
    if path is None:
        return gauge
    listed = pd.read_csv(path, dtype=str, keep_default_na=False)
    if (
        list(listed.columns) != ["timestamp", "layer", "cause"]
        or not listed.layer.isin(["radar", "satellite"]).all()
        or listed.cause.eq("").any()
    ):
        raise ValueError(
            f"{path}: every dropped hour needs a timestamp, a radar or satellite "
            "layer and a cause"
        )
    hours = set(utc_hours(listed.timestamp)) if len(listed) else set()
    kept = gauge[~gauge.timestamp.isin(hours)].reset_index(drop=True)
    settings["dropped_hours"] = {
        "file": str(path),
        "hours": len(hours),
        "station_hours_removed": len(gauge) - len(kept),
    }
    return kept


def input_paths(settings):
    paths = {key: Path(settings[key]) for key in INPUT_KEYS}
    if dropped_hours_file(settings) is not None:
        paths["dropped_hours"] = dropped_hours_file(settings)
    if settings["method"] == "hgnn":
        for fold in range(5):
            for role in ROLES:
                paths[f"fold-{fold}/{role}"] = (
                    Path(settings["graphs_dir"]) / f"fold-{fold}" / f"{role}.pt"
                )
    return paths


def read_inputs(settings):
    validate_settings(settings)
    paths = input_paths(settings)
    missing = [
        f"{label}: {path}" for label, path in paths.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "BLOCKED: missing upstream inputs:\n" + "\n".join(missing)
        )
    gauge, split, stations = load_inputs(
        settings["gauge_file"], settings["split_file"], settings["stations_file"]
    )
    gauge = drop_hours(gauge, settings)
    with Path(settings["seeds_file"]).open() as stream:
        seeds = yaml.safe_load(stream)
    seed = seeds.get("model_seed") if isinstance(seeds, dict) else None
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(
            "seeds_file must supply model_seed as an integer in [0, 2**32)"
        )
    override = settings.pop("model_seed_override", None)
    if override is not None:
        # Repeat runs over model seeds; the split seed still comes from seeds_file.
        if type(override) is not int or not 0 <= override < 2**32:
            raise ValueError("--model-seed must be an integer in [0, 2**32)")
        seed = override
    elif "model_seed" in settings and settings["model_seed"] != seed:
        raise ValueError(
            "model_seed may not override the shared Component 03 seed file"
        )
    settings["model_seed"] = seed
    return gauge, split, stations


def read_fold(settings, fold, gauge):
    graphs = {
        role: validate_graph(
            load_graph(
                Path(settings["graphs_dir"])
                / f"fold-{fold['fold_index']}"
                / f"{role}.pt"
            ),
            fold,
            role,
            gauge,
            settings.get("input_layers", ["raingauge"]),
        )
        for role in ROLES
    }
    training_spec = model_spec(graphs["train"], settings)
    for role in ("validation", "test"):
        if model_spec(graphs[role], settings) != training_spec:
            raise ValueError(
                f"fold {fold['fold_index']}: {role} feature widths or edge relations differ from train"
            )
    if settings["input_rainfall_scale"] == "zscore_train":
        # Statistics come from the training graph alone and are applied to all three,
        # so nothing about validation or test rainfall reaches the scaling.
        statistics = rainfall_scaling(
            graphs["train"], f"fold {fold['fold_index']} train"
        )
        for graph in graphs.values():
            apply_rainfall_scaling(graph, statistics)
        settings.setdefault("input_scaling", {})[fold["fold_index"]] = statistics
    return graphs


def write_config(directory, settings):
    (directory / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))


def check_inputs(settings):
    """Read-only preflight; constrained PyG deserialization, no training or outputs."""
    gauge, split, _ = read_inputs(settings)
    if settings["method"] == "hgnn":
        for fold in split["folds"]:
            read_fold(settings, fold, gauge)
    return {"folds": len(split["folds"]), "target_rows": len(target_rows(gauge, split))}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(settings):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    source_files = [*Path("src/models").glob("*.py"), Path("models/gnn.py")]
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_geometric": str(torch_geometric.__version__),
        "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name()
        if settings["device"] == "cuda"
        else platform.processor(),
        "code_revision": revision.stdout.strip() or "UNKNOWN",
        "worktree_dirty": bool(status.stdout.strip()),
        "source_sha256": {
            str(path): file_digest(path) for path in source_files if path.is_file()
        },
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": file_digest(path)}
            for label, path in input_paths(settings).items()
        },
    }


def copy_upstream_reports(directory, settings):
    """Copy the reports that state the configured period, region, split and graph
    settings into the run's inputs/ folder, so the folder still describes its data
    once it is copied away from the upstream tree (ISS-39)."""
    standard = Path(settings["gauge_file"]).parent
    sources = {
        "standard_report.md": standard / "report.md",
        "split_summary.md": Path(settings["split_file"]).with_suffix(".md"),
    }
    if settings["method"] == "hgnn":
        sources["graph_report.md"] = Path(settings["graphs_dir"]) / "report.md"
        if set(settings.get("input_layers", ["raingauge"])) != {"raingauge"}:
            sources["grid_report.md"] = standard / "grid_report.md"
    copied = {}
    for name, source in sources.items():
        if not source.is_file():
            copied[name] = {"source": str(source)}
            continue
        (directory / "inputs").mkdir(exist_ok=True)
        shutil.copyfile(source, directory / "inputs" / name)
        copied[name] = {"source": str(source.resolve()), "sha256": file_digest(source)}
    settings["data_scope"]["upstream_reports"] = copied


def run(settings, output_dir=None, predict_from=None):
    settings = dict(settings)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    directory = Path(
        output_dir
        or settings.get("output_dir")
        or f"experiments/{settings.get('method', 'unknown')}_{stamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "report.md"
    report_path.write_text("# Model run\n\nStatus: RUNNING\n")
    history, checkpoints = [], {}
    try:
        gauge, split, stations = read_inputs(settings)
        settings["data_scope"] = measure_tables(gauge, stations, split)
        copy_upstream_reports(directory, settings)
        if settings["method"] == "hgnn":
            torch.set_num_threads(settings["cpu_threads"])
            if settings["device"] == "cuda" and settings["deterministic"]:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        settings["provenance"] = provenance(settings)
        write_config(directory, settings)
        tables = []
        checkpoint_input = None
        subset = settings.get("fold_subset")
        if subset is not None and (predict_from or not set(subset) <= set(range(5))):
            raise ValueError(
                "--folds takes indices 0-4 and cannot be combined with --predict-from"
            )
        if predict_from:
            checkpoint_input = torch.load(
                predict_from, map_location="cpu", weights_only=True
            )
            if checkpoint_input.get("format") != "component06-best-v1" or set(
                checkpoint_input["folds"]
            ) != set(range(5)):
                raise ValueError(
                    "--predict-from needs a complete Component 06 five-fold best.pt"
                )
            for index, checkpoint in checkpoint_input["folds"].items():
                if (
                    type(index) is not int
                    or type(checkpoint.get("fold")) is not int
                    or checkpoint["fold"] != index
                ):
                    raise ValueError(
                        f"checkpoint fold identity mismatch at slot {index}; "
                        "using another fold's weights could leak test-station training data"
                    )
                if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1:
                    raise ValueError(f"fold {index}: checkpoint has an invalid epoch")
                if not math.isfinite(checkpoint.get("val_loss", float("nan"))):
                    raise ValueError(
                        f"fold {index}: checkpoint has nonfinite validation loss"
                    )
                state = checkpoint.get("state_dict")
                if (
                    not isinstance(state, dict)
                    or not state
                    or any(
                        not isinstance(value, torch.Tensor)
                        or not torch.isfinite(value).all()
                        for value in state.values()
                    )
                ):
                    raise ValueError(
                        f"fold {index}: checkpoint parameters must be finite tensors"
                    )
            # A checkpoint is tied to the exact canonical targets, graph tensors and settings.
            previous_hashes = {
                name: value["sha256"]
                for name, value in checkpoint_input["inputs"].items()
            }
            current_hashes = {
                name: value["sha256"]
                for name, value in settings["provenance"]["inputs"].items()
            }
            if previous_hashes != current_hashes:
                raise ValueError("checkpoint input hashes differ from this run")
            saved = checkpoint_input["settings"]
            if set(saved.get("input_layers", ["raingauge"])) != set(
                settings.get("input_layers", ["raingauge"])
            ):
                raise ValueError("checkpoint setting differs: input_layers")
            for key in (
                "model_seed",
                "hidden_channels",
                "num_layers",
                "dropout",
                "input_rainfall_scale",
            ):
                if saved[key] != settings[key]:
                    raise ValueError(f"checkpoint setting differs: {key}")
        if settings["method"] == "hgnn":
            targets = target_rows(gauge, split)
            for fold in split["folds"]:
                index = fold["fold_index"]
                if subset is not None and index not in subset:
                    continue
                graphs = read_fold(settings, fold, gauge)
                record_graph_nodes(settings["data_scope"], index, graphs)
                # Records this fold's input scaling and node counts beside the
                # settings it ran with.
                write_config(directory, settings)

                def save_progress(row, best, index=index):
                    history.append(row)
                    checkpoints[index] = best
                    pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(
                        directory / "history.csv", index=False
                    )
                    torch.save(
                        {
                            "format": "component06-best-v1",
                            "folds": checkpoints,
                            "settings": settings,
                            "inputs": settings["provenance"]["inputs"],
                        },
                        directory / "best.pt",
                    )
                    print(
                        f"fold={index} epoch={row['epoch']}/{settings['epochs']} "
                        f"train_loss={row['train_loss']:.8g} "
                        f"val_loss={row['val_loss']:.8g} "
                        f"best_val_loss={row['best_val_loss']:.8g} "
                        f"deteriorations={row['deteriorations']}/"
                        f"{settings['early_stop_patience']} "
                        f"patience_remaining={row['patience_remaining']} "
                        f"early_stop={row['early_stop_status']}",
                        flush=True,
                    )

                if checkpoint_input:
                    checkpoint = checkpoint_input["folds"][index]
                    if checkpoint["model_spec"] != model_spec(
                        graphs["train"], settings
                    ):
                        raise ValueError(
                            f"fold {index}: checkpoint model graph schema differs"
                        )
                    if saved.get("input_scaling", {}).get(index) != settings.get(
                        "input_scaling", {}
                    ).get(index):
                        raise ValueError(
                            f"fold {index}: checkpoint input scaling differs from "
                            "this run's training graph"
                        )
                    model = restore_model(checkpoint, settings)
                    checkpoints[index] = checkpoint
                else:
                    model, _, _ = train_fold(graphs, index, settings, save_progress)
                if subset is not None:
                    continue
                predicted = predict_fold(model, graphs["test"], index, settings)
                predicted = predicted.merge(
                    targets,
                    on=["fold", "station_id", "timestamp"],
                    how="left",
                    validate="one_to_one",
                )
                tables.append(predicted)
            measure_sensor_layers(
                settings["data_scope"], settings["gauge_file"], file_digest
            )
            write_config(directory, settings)
            if subset is not None:
                # Training only: src.models.combine_folds joins the fold runs and
                # --predict-from scores them, so every method still meets the full
                # target check.
                report_path.write_text(
                    "# Model run\n\nStatus: PARTIAL\n\n"
                    f"Trained folds {sorted(subset)} only; no predictions or scores. "
                    "Join the fold runs with src.models.combine_folds, then score the "
                    "result with --predict-from.\n\n"
                    + "\n".join(scope_lines(settings["data_scope"]))
                    + "\n"
                )
                return directory
            hgnn = (
                pd.concat(tables, ignore_index=True)
                .sort_values(["method", "fold", "station_id", "timestamp"])
                .reset_index(drop=True)
            )
            validate_predictions(hgnn, gauge, split, method="hgnn")
            tables = [hgnn]
            methods = settings["baselines"]
        else:
            if predict_from:
                raise ValueError("--predict-from only applies to method hgnn")
            methods = [settings["method"]]
        for method in methods:
            table = predict_baseline(
                method, gauge, stations, split, idw_power=settings.get("idw_power", 2.0)
            )
            validate_predictions(table, gauge, split, method=method)
            tables.append(table)
        predictions = (
            pd.concat(tables, ignore_index=True)
            .sort_values(["method", "fold", "station_id", "timestamp"])
            .reset_index(drop=True)
        )
        validate_predictions(predictions, gauge, split)
        predictions.to_csv(directory / "predictions.csv", index=False)
        pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(
            directory / "history.csv", index=False
        )
        if checkpoint_input:
            torch.save(checkpoint_input, directory / "best.pt")
        missing_counts = []
        if settings["method"] == "hgnn":
            for fold in split["folds"]:
                for role, graph in read_fold(settings, fold, gauge).items():
                    missing_counts.append(
                        f"fold {fold['fold_index']} {role}: {int((~graph['raingauge'].observed).sum())} unobserved station-hours excluded from loss/prediction, using canonical presence and unchanged validity."
                    )
        notes = [
            "Target checker passed: every method predicts exactly the canonical observed test rows across all five folds.",
            *(
                [
                    "Hours dropped by Component 04 (DEC-01): "
                    f"{settings['dropped_hours']['hours']}, removing "
                    f"{settings['dropped_hours']['station_hours_removed']} observed "
                    "station-hours from every method's targets. Each hour and its "
                    f"cause is listed in {settings['dropped_hours']['file']}."
                ]
                if settings.get("dropped_hours")
                else []
            ),
            (
                "Rainfall inputs are z-scored per node type with each fold's training-graph mean and spread, recorded under input_scaling in config.yaml (paper sec 4.3; D08.2). Graph files, targets and predictions stay in physical millimetres; validity and LPE are unscaled. Hidden rainfall and validity are zeroed while LPE is preserved."
                if settings.get("input_rainfall_scale") == "zscore_train"
                else "Gauge features use physical millimetres with no Component 06 normalization; hidden rainfall and validity are zeroed while LPE is preserved. Normalized upstream graphs are rejected until Component 04 supplies an agreed transform contract."
            ),
            "Training retains inherited leave-one-gauge-out masking, weighted MSE and gradient clipping at 1. Missing targets are excluded, and epoch losses are weighted by observed row count; these correct the inherited zero-label/missingness behaviour.",
            "Every method reads the same gauges when predicting the test gauges: the fold's training and validation gauges (paper sec 4.2; D08.2).",
            f"IDW uses haversine distance with explicit power {settings.get('idw_power', 'not used')}; exponent requires Component 03 reviewer signoff.",
            "Checkpoint prediction uses each fold's lowest validation-loss state.",
            "Validation retains the inherited clamp-before-loss rule: negative outputs become zero before scoring. A tied all-zero plateau keeps the first best epoch and can stop before raw outputs cross zero; training loss remains unclamped. This policy needs reviewer agreement before scientific runs.",
            *missing_counts,
        ]
        if settings.get("fixture_label"):
            notes.insert(
                0,
                f"SYNTHETIC FIXTURE ONLY: {settings['fixture_label']}. These scores are software checks, not scientific rainfall results.",
            )
        layers = settings.get("input_layers", ["raingauge"])
        if settings["method"] == "hgnn" and set(layers) != {"raingauge"}:
            notes.extend(
                [
                    f"Selected input layers: {', '.join(layers)}. Feature widths are taken from the validated Component 04 graphs; graph topology and target rows are unchanged.",
                    "Unavailable satellite payloads are computationally zeroed with the source validity flag kept at 0. Radar has no validity channel, so Component 04 drops any hour with a radar cell that has no value, and this run scores every method without those hours (DEC-01).",
                    "TO VALIDATE: tensor checks do not certify radar/satellite product, units, timestamp meaning, coordinate alignment or gauge correction. Components 01/02/04 must supply that evidence before a fusion score supports scientific claims. Gauge-corrected inputs can contain held-out gauge information even when graph masking is correct.",
                ]
            )
        if (
            settings["method"] == "hgnn"
            and predictions.loc[predictions.method.eq("hgnn"), "predicted_mm"]
            .eq(0)
            .all()
        ):
            notes.append(
                "TO VALIDATE: every HGNN prediction in this run is zero after clamping; passing the target checker does not establish useful learning."
            )
        if predict_from:
            notes.append(
                f"Prediction restored from {Path(predict_from).resolve()}; no additional training performed."
            )
        write_report(report_path, predictions, settings, notes=notes)
        return directory
    except BaseException as error:
        write_config(directory, settings)
        report_path.write_text(
            f"# Model run\n\nStatus: FAILED\n\n{type(error).__name__}: {error}\n\nPartial artifacts are retained. No completed result is claimed.\n"
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--predict-from", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument(
        "--folds",
        help="train only these folds, e.g. 0,2 (no scoring; see combine_folds)",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        help="replace seeds_file's model_seed for this run (recorded in config.yaml)",
    )
    args = parser.parse_args(argv)
    try:
        settings = read_settings(args.config)
        for key in ("epochs", "device"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        if args.model_seed is not None:
            settings["model_seed_override"] = args.model_seed
        if args.folds is not None:
            settings["fold_subset"] = sorted({int(f) for f in args.folds.split(",")})
        if args.check_inputs:
            print(json.dumps(check_inputs(settings)))
        else:
            print(run(settings, args.output_dir, args.predict_from))
    except Exception as error:  # noqa: BLE001 -- CLI reports errors already preserved in report.md
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
