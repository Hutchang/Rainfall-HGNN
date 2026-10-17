"""Ensure the deployment entrypoints cannot accidentally start local/long training."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def environment_without_pbs():
    return {key: value for key, value in os.environ.items() if not key.startswith("PBS_")}


def test_job_refuses_execution_without_pbs():
    result = subprocess.run(
        ["bash", "scripts/nci/train_smoke.pbs"],
        cwd=ROOT,
        env=environment_without_pbs(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "submit this script through PBS" in result.stderr
    assert "PBS job:" not in result.stdout


def test_gpu_preflight_refuses_login_node_probe():
    result = subprocess.run(
        ["bash", "scripts/nci/preflight.sh", "--cuda"],
        cwd=ROOT,
        env=environment_without_pbs(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "requires a PBS allocation" in result.stderr
    assert "Python:" not in result.stdout


@pytest.mark.parametrize("epochs", ["0", "3", "100", "-1", "two", "18446744073709551617"])
def test_smoke_rejects_unbounded_or_invalid_epochs_before_gpu_use(epochs):
    environment = environment_without_pbs()
    environment.update(
        PBS_JOBID="guard-test",
        PBS_O_WORKDIR=str(ROOT),
        PBS_NGPUS="1",
        NCI_SMOKE_EPOCHS=epochs,
    )
    result = subprocess.run(
        ["bash", "scripts/nci/train_smoke.pbs"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "must be 1 or 2" in result.stderr
    assert "PBS job:" not in result.stdout


@pytest.mark.parametrize("script", ["build_grids.pbs", "build_graphs.pbs", "train.pbs"])
def test_pipeline_jobs_refuse_execution_without_pbs(script):
    result = subprocess.run(
        ["bash", f"scripts/nci/{script}"],
        cwd=ROOT,
        env=environment_without_pbs(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "submit this script through PBS" in result.stderr
    assert "PBS job:" not in result.stdout


def training_environment(**settings):
    environment = environment_without_pbs()
    environment.update(PBS_JOBID="guard-test", PBS_O_WORKDIR=str(ROOT), PBS_NGPUS="1")
    environment.update(settings)
    return environment


def test_training_needs_a_config_and_an_output_folder():
    result = subprocess.run(
        ["bash", "scripts/nci/train.pbs"],
        cwd=ROOT,
        env=training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "must be passed with qsub -v" in result.stderr


@pytest.mark.parametrize("epochs", ["0", "1000", "-1", "two", "1.5"])
def test_training_rejects_a_bad_epoch_count_before_gpu_use(epochs):
    result = subprocess.run(
        ["bash", "scripts/nci/train.pbs"],
        cwd=ROOT,
        env=training_environment(
            NCI_MODEL_CONFIG="config/models/d09_round_robin_grs.yaml",
            NCI_OUTPUT_DIR="experiments/nci/guard-test",
            NCI_EPOCHS=epochs,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "NCI_EPOCHS must be a whole number" in result.stderr
    assert "PBS job:" not in result.stdout


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"NCI_FOLDS": "5"}, "NCI_FOLDS must be fold numbers"),
        ({"NCI_FOLDS": "0,1"}, "NCI_FOLDS must be fold numbers"),
        ({"NCI_MODEL_SEED": "-1"}, "NCI_MODEL_SEED must be a whole number"),
        (
            {"NCI_FOLDS": "0", "NCI_PREDICT_FROM": "joined/best.pt"},
            "do not set NCI_FOLDS",
        ),
        ({"NCI_PREDICT_FROM": "no/such/best.pt"}, "is not a file"),
    ],
    ids=[
        "fold-5",
        "comma-list",
        "negative-seed",
        "folds-with-scoring",
        "no-checkpoint",
    ],
)
def test_training_rejects_bad_fold_seed_or_scoring_settings(settings, message):
    result = subprocess.run(
        ["bash", "scripts/nci/train.pbs"],
        cwd=ROOT,
        env=training_environment(
            NCI_MODEL_CONFIG="config/models/d09_round_robin_grs.yaml",
            NCI_OUTPUT_DIR="experiments/nci/guard-test",
            **settings,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert message in result.stderr
    assert "PBS job:" not in result.stdout


def test_the_chain_is_submitted_from_a_login_node_only():
    result = subprocess.run(
        ["bash", "scripts/nci/submit_pipeline.sh"],
        cwd=ROOT,
        env=training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not inside a job" in result.stderr
