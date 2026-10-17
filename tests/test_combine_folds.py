"""src/models/combine_folds.py joins fold-subset runs (run.py --folds) for scoring."""

import runpy
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.combine_folds import combine, main
from src.models.run import read_settings, run


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    """The synthetic fixture trained as two partial runs: folds 0-1 and folds 2-4."""
    root = tmp_path_factory.mktemp("folds")
    make = runpy.run_path(str(Path(__file__).parent / "data/models/make_fixture.py"))[
        "make_fixture"
    ]
    inputs = read_settings(make(root / "sample"))
    part_a = run(dict(inputs, fold_subset=[0, 1]), root / "part-a")
    part_b = run(dict(inputs, fold_subset=[2, 3, 4]), root / "part-b")
    return inputs, part_a, part_b


def read_predictions(directory):
    result = pd.read_csv(directory / "predictions.csv", dtype={"station_id": str})
    keys = ["method", "fold", "station_id", "timestamp"]
    return result.sort_values(keys).reset_index(drop=True)


def edited_copy(run_dir, destination, edit_best=None, edit_history=None):
    shutil.copytree(run_dir, destination)
    if edit_best:
        best = torch.load(destination / "best.pt", weights_only=True)
        edit_best(best)
        torch.save(best, destination / "best.pt")
    if edit_history:
        history = pd.read_csv(destination / "history.csv")
        edit_history(history).to_csv(destination / "history.csv", index=False)
    return destination


def test_folds_trained_in_parts_score_the_same_as_one_full_run(parts, tmp_path):
    inputs, part_a, part_b = parts
    assert "Status: PARTIAL" in (part_a / "report.md").read_text()
    assert not (part_a / "predictions.csv").exists()
    joined = combine(tmp_path / "joined", [part_a, part_b])
    scored = run(dict(inputs), tmp_path / "scored", joined / "best.pt")
    full = run(dict(inputs), tmp_path / "full")
    # Each fold is seeded on its own, so training it alone gives the same model.
    pd.testing.assert_frame_equal(
        read_predictions(full), read_predictions(scored), check_exact=True
    )
    history = pd.read_csv(joined / "history.csv")
    assert sorted(history.fold.unique()) == [0, 1, 2, 3, 4]


def test_a_fold_cut_off_before_it_finished_is_refused(parts, tmp_path):
    _, part_a, part_b = parts

    def cut_fold_3(history):
        kept = history[(history.fold != 3) | (history.epoch == 1)].copy()
        kept.loc[kept.fold == 3, "early_stop_status"] = "IMPROVED"
        return kept

    cut = edited_copy(part_b, tmp_path / "cut", edit_history=cut_fold_3)
    with pytest.raises(ValueError, match="fold 3 did not finish training"):
        combine(tmp_path / "joined", [part_a, cut])


def test_runs_with_different_settings_or_inputs_are_refused(parts, tmp_path):
    _, part_a, part_b = parts

    def wider(best):
        best["settings"]["hidden_channels"] += 1

    def other_inputs(best):
        best["inputs"] = dict(best["inputs"], extra="another file")

    changed = edited_copy(part_b, tmp_path / "wider", edit_best=wider)
    with pytest.raises(ValueError, match="setting hidden_channels differs"):
        combine(tmp_path / "joined-a", [part_a, changed])
    moved = edited_copy(part_b, tmp_path / "moved", edit_best=other_inputs)
    with pytest.raises(ValueError, match="input hashes differ"):
        combine(tmp_path / "joined-b", [part_a, moved])


def test_a_fold_twice_or_a_missing_fold_is_refused(parts, tmp_path):
    _, part_a, part_b = parts
    with pytest.raises(ValueError, match="fold 0 appears in more than one run"):
        combine(tmp_path / "twice", [part_a, part_a, part_b])
    with pytest.raises(ValueError, match=r"folds \[2, 3, 4\] are missing"):
        combine(tmp_path / "missing", [part_a])
    assert not (tmp_path / "twice").exists() and not (tmp_path / "missing").exists()


def test_an_existing_output_folder_is_never_overwritten(parts, tmp_path, capsys):
    _, part_a, part_b = parts
    existing = tmp_path / "joined"
    existing.mkdir()
    (existing / "note.txt").write_text("keep me\n")
    assert main([str(existing), str(part_a), str(part_b)]) == 1
    assert "FileExistsError" in capsys.readouterr().err
    assert sorted(path.name for path in existing.iterdir()) == ["note.txt"]
