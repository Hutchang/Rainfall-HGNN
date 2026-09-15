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
