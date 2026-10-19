"""A fold stopped by its walltime carries on from last.pt exactly where it stopped."""

import runpy
from pathlib import Path

import pandas as pd
import pytest
import torch

import src.models.run as run_module
from src.models.run import read_settings, run

MAKE_FIXTURE = runpy.run_path(
    str(Path(__file__).parent / "data/models/make_fixture.py")
)["make_fixture"]


class StoppedError(Exception):
    """Stands in for PBS ending a job at its walltime."""


@pytest.fixture
def settings(tmp_path):
    settings = read_settings(MAKE_FIXTURE(tmp_path / "sample"))
    # Enough epochs to stop part-way; patience high enough that none stop early.
    return dict(settings, epochs=5, early_stop_patience=10, fold_subset=[1])


def stop_after(epochs, monkeypatch):
    train_fold = run_module.train_fold

    def stopping(graphs, index, settings, on_epoch, resume=None):
        def counted(row, best, last):
            on_epoch(row, best, last)
            if row["epoch"] == epochs:
                raise StoppedError

        return train_fold(graphs, index, settings, counted, resume)

    monkeypatch.setattr(run_module, "train_fold", stopping)


def test_a_resumed_fold_matches_one_that_never_stopped(settings, tmp_path, monkeypatch):
    whole = run(settings, tmp_path / "whole")

    with monkeypatch.context() as patch:
        stop_after(2, patch)
        with pytest.raises(StoppedError):
            run(settings, tmp_path / "stopped")
    stopped = tmp_path / "stopped"
    assert len(pd.read_csv(stopped / "history.csv")) == 2
    run(settings, stopped, resume=True)

    pd.testing.assert_frame_equal(
        pd.read_csv(whole / "history.csv"), pd.read_csv(stopped / "history.csv")
    )
    expected = torch.load(whole / "best.pt", weights_only=True)["folds"][1]
    resumed = torch.load(stopped / "best.pt", weights_only=True)["folds"][1]
    assert resumed["epoch"] == expected["epoch"]
    for name, value in expected["state_dict"].items():
        assert torch.equal(resumed["state_dict"][name], value), name
    assert "Status: PARTIAL" in (stopped / "report.md").read_text()


def test_resuming_a_finished_fold_trains_nothing_more(settings, tmp_path):
    first = run(settings, tmp_path / "run")
    before = pd.read_csv(first / "history.csv")
    run(settings, first, resume=True)
    pd.testing.assert_frame_equal(before, pd.read_csv(first / "history.csv"))


def test_a_resume_with_other_settings_is_refused(settings, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        stop_after(1, patch)
        with pytest.raises(StoppedError):
            run(settings, tmp_path / "stopped")
    with pytest.raises(
        ValueError, match="setting differs from the stopped run: dropout"
    ):
        run(dict(settings, dropout=0.5), tmp_path / "stopped", resume=True)


def test_a_resume_needs_the_stopped_runs_checkpoint(settings, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=r"last\.pt"):
        run(settings, tmp_path / "empty", resume=True)
    with pytest.raises(ValueError, match="cannot --predict-from"):
        run(settings, tmp_path / "empty", tmp_path / "best.pt", resume=True)
