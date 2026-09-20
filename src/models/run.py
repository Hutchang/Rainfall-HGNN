"""Run Component 06 against supplied canonical tables, splits and graph samples."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import torch_geometric
import yaml

from src.models.baselines import predict_baseline
from src.models.contracts import load_inputs, target_rows, validate_predictions
from src.models.evaluate import write_report
from src.models.graphs import load_graph, validate_graph
from src.models.modalities import validate_layer_selection
from src.models.trainer import model_spec, predict_fold, restore_model, train_fold

INPUT_KEYS = ("gauge_file", "stations_file", "split_file", "seeds_file")
ROLES = ("train", "validation", "test")


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
        if settings.get("input_rainfall_scale") != "physical_mm":
            raise ValueError(
                "input_rainfall_scale must be physical_mm; normalized graph inputs are TO VALIDATE with Component 04"
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


def input_paths(settings):
    paths = {key: Path(settings[key]) for key in INPUT_KEYS}
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
    with Path(settings["seeds_file"]).open() as stream:
        seeds = yaml.safe_load(stream)
    seed = seeds.get("model_seed") if isinstance(seeds, dict) else None
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(
            "seeds_file must supply model_seed as an integer in [0, 2**32)"
        )
    if "model_seed" in settings and settings["model_seed"] != seed:
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
    return graphs


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
        if settings["method"] == "hgnn":
            torch.set_num_threads(settings["cpu_threads"])
            if settings["device"] == "cuda" and settings["deterministic"]:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        settings["provenance"] = provenance(settings)
        (directory / "config.yaml").write_text(
            yaml.safe_dump(settings, sort_keys=False)
        )
        tables = []
        checkpoint_input = None
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
                graphs = read_fold(settings, fold, gauge)

                def save_progress(row, best, index=index):
                    history.append(row)
                    checkpoints[index] = best
                    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                    torch.save(
                        {
                            "format": "component06-best-v1",
                            "folds": checkpoints,
                            "settings": settings,
                            "inputs": settings["provenance"]["inputs"],
                        },
                        directory / "best.pt",
                    )

                if checkpoint_input:
                    checkpoint = checkpoint_input["folds"][index]
                    if checkpoint["model_spec"] != model_spec(
                        graphs["train"], settings
                    ):
                        raise ValueError(
                            f"fold {index}: checkpoint model graph schema differs"
                        )
                    model = restore_model(checkpoint, settings)
                    checkpoints[index] = checkpoint
                else:
                    model, _, _ = train_fold(graphs, index, settings, save_progress)
                predicted = predict_fold(model, graphs["test"], index, settings)
                predicted = predicted.merge(
                    targets,
                    on=["fold", "station_id", "timestamp"],
                    how="left",
                    validate="one_to_one",
                )
                tables.append(predicted)
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
        pd.DataFrame(
            history, columns=["fold", "epoch", "train_loss", "val_loss"]
        ).to_csv(directory / "history.csv", index=False)
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
            "Gauge features use physical millimetres with no Component 06 normalization; hidden rainfall and validity are zeroed while LPE is preserved. Normalized upstream graphs are rejected until Component 04 supplies an agreed transform contract.",
            "Training retains inherited leave-one-gauge-out masking, weighted MSE and gradient clipping at 1. Missing targets are excluded, and epoch losses are weighted by observed row count; these correct the inherited zero-label/missingness behaviour.",
            "TO VALIDATE: Component 04 permits train + validation context at GNN test time; the prescribed baselines read training gauges only. The two methods therefore read different gauges, so the comparison is unbalanced pending reviewer agreement.",
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
                    "Unavailable satellite payloads are computationally zeroed with the source validity flag kept at 0. Radar missingness is rejected because its prescribed single feature has no validity channel. No rainfall rescaling or automatic timeline intersection occurs.",
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
        (directory / "config.yaml").write_text(
            yaml.safe_dump(settings, sort_keys=False)
        )
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
    args = parser.parse_args(argv)
    try:
        settings = read_settings(args.config)
        for key in ("epochs", "device"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
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
