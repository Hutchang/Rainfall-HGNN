"""Offline tests for scripts/nci/remote.py; a fake stands in for ssh and Gadi."""

import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from scripts.nci import remote
from scripts.nci.remote import BlockedError

ROOT = Path(__file__).resolve().parents[1]

# Real output from Gadi on 2026-09-27, as pasted by the project lead.
LQUOTA = """\
--------------------------------------------------------------------------
           fs       Usage      Quota      Limit   iUsage   iQuota   iLimit
--------------------------------------------------------------------------
   rq0 scratch 136.00 KiB     0.00 B     0.00 B       27        0        0
  rv74 scratch 148.00 KiB     0.00 B     0.00 B       30        0        0
  no50 scratch 192.37 GiB   1.00 TiB   1.05 TiB   106239   202000   212100
   rq0   gdata 276.55 TiB 283.00 TiB 297.15 TiB 10175130 16740000 17577000
  rv74   gdata  74.80 TiB 120.00 TiB 126.00 TiB  4296016  7954000  8351700
  no50   gdata  17.63 GiB  50.00 GiB  52.50 GiB   137533   202000   212100
"""
ACCOUNT = """\

Usage Report: Project=no50 Period=2026.q3
=============================================================
    Grant:    34.00 KSU
     Used:    28.33 KSU
 Reserved:     0.00 SU
    Avail:     5.67 KSU
"""
# The end of a real Gadi job log (job 181460515, 2026-10-17), as pasted by the
# project lead.
LOG_END = """\
======================================================================================
                  Resource Usage on 2026-10-17 14:57:55:
   Job Id:             181460515.gadi-pbs
   Project:            no50
   Exit Status:        0
   Service Units:      0.04
   NCPUs Requested:    16                  CPU Time Used: 00:00:00
   Memory Requested:   16.0GB                Memory Used: 18.95MB
   NGPUs Requested:    1                 GPU Utilisation: 0%
                                         GPU Memory Used: 0B
   Walltime Requested: 00:05:00            Walltime Used: 00:00:02
   JobFS Requested:    100.0MB                JobFS Used: 0B
======================================================================================
"""


LOCK = b"version = 1\n"  # the test repository's uv.lock
LOCK_SHA256 = hashlib.sha256(LOCK).hexdigest()
RUNNER = "/scratch/no50/rh0513/rainflow"  # scratch_root for user rh0513


def gadi_jobs(*jobs):
    """`qstat -f -F json` for (job id, state, folder it was submitted from)."""
    return json.dumps(
        {
            "Jobs": {
                job_id: {"job_state": state, "Variable_List": {"PBS_O_WORKDIR": folder}}
                for job_id, state, folder in jobs
            }
        }
    )


def probe_output(
    venv="present", inputs="absent", account=ACCOUNT, env_lock=LOCK_SHA256, jobs=None
):
    return (
        f"repo=absent\nrepo_head=\nhas_revision=no\ninputs={inputs}\nvenv={venv}\n"
        f"env_lock={env_lock}\n"
        f"--- lquota\n{LQUOTA}--- nci_account\n{account}"
        f"--- jobs\n{jobs or gadi_jobs()}\n"
    )


def qstat(job_id, state, exit_status=None):
    job = {"job_state": state}
    if exit_status is not None:
        job["Exit_status"] = exit_status
    return json.dumps({"Jobs": {job_id: job}})


class FakeSsh:
    """Answers commands from (pattern, reply) pairs and records what was sent."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.files = []
        self.tars = []
        self.tar_to_return = None

    def _reply(self, command):
        self.calls.append(command)
        for pattern, reply in self.replies:
            if re.search(pattern, command):
                return reply(command) if callable(reply) else reply
        return ""

    def run(self, host, command, data=None):
        return self._reply(command)

    def send_file(self, host, command, path):
        self._reply(command)
        self.files.append(Path(path).read_bytes())

    def send_tar(self, host, command, files):
        self._reply(command)
        buffer = io.BytesIO()
        remote.write_tar(buffer, files)
        self.tars.append(buffer.getvalue())

    def receive_tar(self, host, command, destination):
        self._reply(command)
        remote.extract(io.BytesIO(self.tar_to_return), destination)


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "config" / "nci").mkdir(parents=True)
    (root / "data").mkdir()
    shutil.copy(ROOT / "config/nci/remote.yaml", root / "config/nci/remote.yaml")
    (root / "config/seeds.yaml").write_text("global: 1\n")
    settings = {
        "method": "idw",
        "gauge_file": "data/gauge.csv",
        "stations_file": "data/stations.csv",
        "split_file": "data/split.json",
        "seeds_file": "config/seeds.yaml",
        "device": "cpu",
        "idw_power": 2.0,
    }
    (root / "config/model.yaml").write_text(yaml.safe_dump(settings))
    for name in ("gauge.csv", "stations.csv", "split.json"):
        (root / "data" / name).write_text(f"contents of {name}\n")
    (root / ".gitignore").write_text("data/\nexperiments/\n")
    (root / "uv.lock").write_bytes(LOCK)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "fixture")
    return root


def gadi(**overrides):
    replies = {
        "probe": probe_output(),
        "check": "ready\n",
        "prepare_qsub": "100.gadi-pbs\n",
        "qsub": "123.gadi-pbs\n",
        **overrides,
    }
    return FakeSsh(
        [
            (r"^id -un$", "rh0513\n"),
            (r"^bash -s", replies["probe"]),
            (r"check_environment", replies["check"]),
            (r"prepare_environment\.pbs", replies["prepare_qsub"]),
            (r"qsub", replies["qsub"]),
        ]
    )


def test_lquota_rows_are_read_for_one_project():
    rows = remote.parse_lquota(LQUOTA, "no50")
    assert set(rows) == {"scratch", "gdata"}
    assert rows["gdata"]["files"] == 137533
    assert rows["gdata"]["file_quota"] == 202000
    assert rows["gdata"]["quota"] == 50 * 2**30
    assert rows["scratch"]["used"] == pytest.approx(192.37 * 2**30)


def test_unpacked_raw_cache_does_not_fit_gdata_file_quota():
    rows = remote.parse_lquota(LQUOTA, "no50")
    problem = remote.headroom_problem(rows, "gdata", 339 * 2**20, 81930)
    assert "file-count quota" in problem
    assert remote.headroom_problem(rows, "scratch", 339 * 2**20, 81930) is None


def test_size_quota_is_checked():
    rows = remote.parse_lquota(LQUOTA, "no50")
    assert "size quota" in remote.headroom_problem(rows, "gdata", 40 * 2**30, 1)


def test_available_su_is_read_from_nci_account():
    assert remote.parse_available_su(ACCOUNT) == pytest.approx(5670)


def test_su_estimate_matches_the_documented_smoke_job():
    job = {"ncpus": 16, "walltime": "00:10:00"}
    assert remote.su_estimate(job, 4.5) == pytest.approx(12)


def test_walltime_must_be_hours_minutes_seconds():
    with pytest.raises(BlockedError, match="HH:MM:SS"):
        remote.walltime_hours("1:00")


def test_remote_config_loads_and_rejects_smuggled_resources(tmp_path):
    cfg = remote.load_remote_config(ROOT / "config/nci/remote.yaml")
    assert cfg["project"] == "no50"
    cfg["job"]["mem"] = "48GB,ngpus=4"
    path = tmp_path / "remote.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(BlockedError, match=r"job\.mem"):
        remote.load_remote_config(path)


def test_qstat_json_gives_state_and_exit_status():
    assert remote.parse_qstat(qstat("1.gadi-pbs", "F", 0), "1.gadi-pbs") == ("F", 0)
    assert remote.parse_qstat(qstat("1.gadi-pbs", "R"), "1.gadi-pbs") == ("R", None)
    with pytest.raises(BlockedError, match="no record"):
        remote.parse_qstat('{"Jobs": {}}', "1.gadi-pbs")


def test_storage_directive_mounts_each_project_once():
    directive = remote.storage_directive(
        "/scratch/no50/u/run", "/g/data/no50/u/in", "/scratch/no50/u/x"
    )
    assert directive == "scratch/no50+gdata/no50"
    with pytest.raises(BlockedError):
        remote.storage_directive("/home/123/u/run")


def test_qsub_command_carries_resources_storage_and_dependency():
    cfg = remote.load_remote_config(ROOT / "config/nci/remote.yaml")
    command = remote.qsub_command(
        cfg, "train.pbs", {"A": "/x"}, "scratch/no50", "/x/log", "100.gadi-pbs"
    )
    assert command[:4] == ["qsub", "-P", "no50", "-q"]
    assert any(part.endswith(",storage=scratch/no50") for part in command)
    assert command[-3:] == ["-W", "depend=afterok:100.gadi-pbs", "train.pbs"]
    with pytest.raises(BlockedError, match="qsub -v"):
        remote.qsub_command(cfg, "train.pbs", {"A": "/x,y"}, "s", "/log")


def test_run_folders_never_reuse_a_name(tmp_path):
    now = datetime(2026, 9, 27, 14, 32, 5)
    first = remote.new_run_dir(tmp_path, "baseline", now)
    second = remote.new_run_dir(tmp_path, "baseline", now)
    assert first.name == "2026-09-27_1432_baseline"
    assert second.name == "2026-09-27_143205_baseline"
    with pytest.raises(BlockedError, match="label"):
        remote.new_run_dir(tmp_path, "../escape", now)


def test_downloads_cannot_write_outside_the_results_folder(tmp_path):
    buffer = io.BytesIO()
    remote.write_tar(buffer, [(b"x", "../escaped.txt")])
    buffer.seek(0)
    with pytest.raises(tarfile.FilterError):
        remote.extract(buffer, tmp_path / "results")
    assert not (tmp_path / "escaped.txt").exists()


def test_uncommitted_tracked_edits_are_refused_but_untracked_files_are_not(repo):
    (repo / "notes.md").write_text("untracked\n")
    assert remote.committed_revision(repo) == git(repo, "rev-parse", "HEAD")
    (repo / "config/seeds.yaml").write_text("global: 2\n")
    with pytest.raises(BlockedError, match="commit or stash"):
        remote.committed_revision(repo)


def test_tracked_inputs_travel_with_code_and_the_rest_are_uploaded(repo):
    settings = remote.read_settings(repo / "config/model.yaml")
    inputs = {item.label: item for item in remote.plan_inputs(settings, repo)}
    assert inputs["seeds_file"].tracked
    assert inputs["seeds_file"].remote == "config/seeds.yaml"
    assert not inputs["gauge_file"].tracked
    assert inputs["gauge_file"].remote == "gauge_file/gauge.csv"
    assert inputs["gauge_file"].sha256 == remote.file_digest(repo / "data/gauge.csv")


def test_tracked_inputs_are_hashed_as_committed_not_as_checked_out(repo):
    # Windows checks text out with CRLF and commits LF; Gadi reads the LF bytes.
    git(repo, "config", "core.autocrlf", "true")
    (repo / "config/seeds.yaml").unlink()
    git(repo, "checkout", "--", "config/seeds.yaml")
    assert (repo / "config/seeds.yaml").read_bytes() == b"global: 1\r\n"
    remote.committed_revision(repo)  # git does not count the CRLF as an edit
    settings = remote.read_settings(repo / "config/model.yaml")
    inputs = {item.label: item for item in remote.plan_inputs(settings, repo)}
    assert inputs["seeds_file"].sha256 == hashlib.sha256(b"global: 1\n").hexdigest()
    assert inputs["seeds_file"].size == len(b"global: 1\n")


def test_missing_inputs_are_all_named(repo):
    (repo / "data/split.json").unlink()
    (repo / "data/stations.csv").unlink()
    settings = remote.read_settings(repo / "config/model.yaml")
    with pytest.raises(BlockedError) as error:
        remote.plan_inputs(settings, repo)
    assert "split_file" in str(error.value)
    assert "stations_file" in str(error.value)


def test_graphs_are_uploaded_under_one_folder(repo):
    for fold in range(5):
        for role in ("train", "validation", "test"):
            path = repo / "data/graphs" / f"fold-{fold}" / f"{role}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"graph")
    settings = dict(
        remote.read_settings(repo / "config/model.yaml"),
        method="hgnn",
        graphs_dir="data/graphs",
    )
    inputs = remote.plan_inputs(settings, repo)
    by_label = {item.label: item for item in inputs}
    assert by_label["fold-2/test"].remote == "graphs/fold-2/test.pt"
    result = remote.remote_settings(settings, inputs, "/g/in", "/s/out")
    assert result["graphs_dir"] == "/g/in/graphs"
    assert result["seeds_file"] == "config/seeds.yaml"
    assert result["gauge_file"] == "/g/in/gauge_file/gauge.csv"
    assert result["output_dir"] == "/s/out"
    assert result["device"] == "cuda"


def test_inputs_fingerprint_follows_file_contents(repo):
    settings = remote.read_settings(repo / "config/model.yaml")
    before = remote.inputs_fingerprint(remote.plan_inputs(settings, repo))
    (repo / "data/gauge.csv").write_text("changed\n")
    after = remote.inputs_fingerprint(remote.plan_inputs(settings, repo))
    assert before != after


def test_plan_refuses_a_job_the_su_balance_cannot_cover(repo):
    cfg = remote.load_remote_config(repo / "config/nci/remote.yaml")
    ssh = gadi(probe=probe_output(account=ACCOUNT.replace("5.67 KSU", "30.00 SU")))
    with pytest.raises(BlockedError, match="72 SU; 30 SU left"):
        remote.make_plan(Path("config/model.yaml"), cfg, ssh, repo)


def test_declining_the_plan_uploads_and_submits_nothing(repo):
    ssh = gadi()
    code = remote.main(
        ["submit", "config/model.yaml"],
        ssh_factory=lambda cfg: ssh,
        confirm=lambda prompt: "no",
        root=repo,
    )
    assert code == 1
    assert not ssh.files and not ssh.tars
    assert not any("qsub" in call or "mkdir" in call for call in ssh.calls)
    assert not (repo / "experiments").exists()


def submitted_run(repo, ssh):
    code = remote.main(
        ["submit", "config/model.yaml", "--label", "t", "--yes"],
        ssh_factory=lambda cfg: ssh,
        root=repo,
    )
    assert code == 0
    (run_dir,) = (repo / "experiments/nci").iterdir()
    return run_dir, json.loads((run_dir / "state.json").read_text())


def test_submit_ships_the_commit_and_checked_inputs_then_records_the_job(
    repo, tmp_path
):
    ssh = gadi()
    run_dir, state = submitted_run(repo, ssh)
    revision = git(repo, "rev-parse", "HEAD")
    assert state["status"] == "SUBMITTED"
    assert state["revision"] == revision
    assert state["jobs"] == {"train": "123.gadi-pbs"}

    (bundle,) = ssh.files
    (tmp_path / "code.bundle").write_bytes(bundle)
    git(tmp_path, "clone", "-q", "code.bundle", "clone")
    assert git(tmp_path / "clone", "rev-parse", "HEAD") == revision

    (tar,) = ssh.tars
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        sums = archive.extractfile("SHA256SUMS").read().decode().splitlines()
        names = set(archive.getnames())
    assert names == {
        "SHA256SUMS",
        "gauge_file/gauge.csv",
        "stations_file/stations.csv",
        "split_file/split.json",
    }
    assert (
        f"{remote.file_digest(repo / 'data/gauge.csv')}  gauge_file/gauge.csv" in sums
    )

    settings = yaml.safe_load((run_dir / "run.yaml").read_text())
    inputs = f"/g/data/no50/rh0513/rainflow-inputs/{state['inputs_fingerprint'][:16]}"
    assert settings["gauge_file"] == f"{inputs}/gauge_file/gauge.csv"
    assert settings["seeds_file"] == "config/seeds.yaml"
    (qsub_call,) = [call for call in ssh.calls if "train.pbs" in call]
    assert "storage=scratch/no50+gdata/no50" in qsub_call
    assert "depend" not in qsub_call
    assert any("--check-inputs" in call for call in ssh.calls)


def test_missing_environment_is_prepared_before_training(repo):
    ssh = gadi(check="prepare\n")
    _, state = submitted_run(repo, ssh)
    assert state["jobs"] == {"prepare": "100.gadi-pbs", "train": "123.gadi-pbs"}
    (qsub_call,) = [call for call in ssh.calls if "train.pbs" in call]
    assert "depend=afterok:100.gadi-pbs" in qsub_call
    assert not any("--check-inputs" in call for call in ssh.calls)


def test_a_refused_training_job_leaves_the_environment_job_on_record(repo, capsys):
    def refuse(command):
        raise BlockedError("qsub: Job violates queue and/or server resource limits")

    ssh = gadi(check="prepare\n", qsub=refuse)
    code = remote.main(
        ["submit", "config/model.yaml", "--label", "t", "--yes"],
        ssh_factory=lambda cfg: ssh,
        root=repo,
    )
    assert code == 1
    (run_dir,) = (repo / "experiments/nci").iterdir()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "SUBMIT_FAILED"
    assert state["jobs"] == {"prepare": "100.gadi-pbs"}
    assert "resource limits" in state["error"]
    assert "qdel 100.gadi-pbs" in capsys.readouterr().err


@pytest.mark.parametrize("job_state", ["Q", "R"])
def test_a_new_environment_is_refused_before_upload_while_a_job_uses_the_old(
    repo, capsys, job_state
):
    busy = gadi_jobs(("200.gadi-pbs", job_state, f"{RUNNER}/runs/a/code"))
    ssh = gadi(probe=probe_output(env_lock="0" * 64, jobs=busy))
    code = remote.main(
        ["submit", "config/model.yaml", "--yes"], ssh_factory=lambda cfg: ssh, root=repo
    )
    assert code == 1
    assert "200.gadi-pbs" in capsys.readouterr().err
    assert not ssh.files and not ssh.tars
    assert not any("qsub" in call or "mkdir" in call for call in ssh.calls)
    assert not (repo / "experiments").exists()


def test_the_same_environment_runs_alongside_unfinished_jobs(repo):
    busy = gadi_jobs(("200.gadi-pbs", "R", f"{RUNNER}/runs/a/code"))
    _, state = submitted_run(repo, gadi(probe=probe_output(jobs=busy)))
    assert state["jobs"] == {"train": "123.gadi-pbs"}


def test_a_changed_uv_lock_rebuilds_when_only_other_jobs_are_unfinished(repo):
    other = gadi_jobs(
        ("200.gadi-pbs", "R", "/scratch/no50/rh0513/COMP3888-m17-01-p23"),
        ("201.gadi-pbs", "F", f"{RUNNER}/runs/a/code"),
    )
    # The environment check passes (say, only a dev tool changed), but the job runs
    # anyway, so the shared checkout's uv.lock keeps naming the environment's.
    ssh = gadi(probe=probe_output(env_lock="0" * 64, jobs=other), check="ready\n")
    _, state = submitted_run(repo, ssh)
    assert state["jobs"] == {"prepare": "100.gadi-pbs", "train": "123.gadi-pbs"}


def test_an_unfinished_environment_job_blocks_a_second_one(repo):
    pending = gadi_jobs(("100.gadi-pbs", "Q", f"{RUNNER}/repo"))
    ssh = gadi(probe=probe_output(jobs=pending), check="prepare\n")
    code = remote.main(
        ["submit", "config/model.yaml", "--yes"], ssh_factory=lambda cfg: ssh, root=repo
    )
    assert code == 1
    assert not any("prepare_environment" in call for call in ssh.calls)
    (run_dir,) = (repo / "experiments/nci").iterdir()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "SUBMIT_FAILED"
    assert "100.gadi-pbs" in state["error"]


def test_a_job_list_that_cannot_be_read_blocks_the_plan(repo):
    cfg = remote.load_remote_config(repo / "config/nci/remote.yaml")
    ssh = gadi(probe=probe_output(jobs="qselect failed: command not found"))
    with pytest.raises(BlockedError, match="could not list your jobs"):
        remote.make_plan(Path("config/model.yaml"), cfg, ssh, repo)


def results_tar(report="# Model run\n\n| fold | rmse |\n"):
    buffer = io.BytesIO()
    remote.write_tar(
        buffer,
        [
            (b"station_id,value\n", "output/predictions.csv"),
            (report.encode(), "output/report.md"),
            (b"PBS job: 123\n", "pbs.log"),
        ],
    )
    return buffer.getvalue()


def write_run(tmp_path, jobs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "login_host": "gadi",
        "transfer_host": "gadi-dm",
        "remote": {
            "run": "/scratch/no50/u/rainflow/runs/run",
            "log": "/scratch/no50/u/rainflow/runs/run/pbs.log",
            "prepare_log": "/scratch/no50/u/rainflow/runs/run/prepare.log",
        },
        "jobs": jobs,
    }
    (run_dir / "state.json").write_text(json.dumps(state))
    return run_dir


def test_fetch_waits_for_the_job_then_downloads_once(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    states = iter([qstat("123.gadi-pbs", "R"), qstat("123.gadi-pbs", "F", 0)])
    ssh = FakeSsh([(r"^qstat", lambda command: next(states))])
    ssh.tar_to_return = results_tar()
    said = []
    state = remote.fetch(run_dir, ssh, 300, sleep=lambda seconds: None, say=said.append)
    assert state["status"] == "DONE"
    assert said == ["train 123.gadi-pbs: running", "train 123.gadi-pbs: finished"]
    assert (run_dir / "results/output/predictions.csv").is_file()
    with pytest.raises(BlockedError, match="never overwritten"):
        remote.fetch(run_dir, ssh, 300, sleep=lambda seconds: None)


def test_a_failed_report_is_not_called_done(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", qstat("123.gadi-pbs", "F", 0))])
    ssh.tar_to_return = results_tar("# Model run\n\nStatus: FAILED\n")
    state = remote.fetch(run_dir, ssh, 300, say=lambda text: None)
    assert state["status"] == "FAILED: report.md does not show a completed run"


def test_training_dropped_after_failed_preparation_is_reported(tmp_path):
    run_dir = write_run(tmp_path, {"prepare": "100.gadi-pbs", "train": "123.gadi-pbs"})
    ssh = FakeSsh(
        [
            (r"100\.gadi-pbs", qstat("100.gadi-pbs", "F", 1)),
            (r"123\.gadi-pbs", '{"Jobs": {}}'),
        ]
    )
    ssh.tar_to_return = results_tar()
    state = remote.fetch(run_dir, ssh, 300, say=lambda text: None)
    assert state["exits"] == {"prepare": 1, "train": None}
    assert state["status"] == "FAILED: environment preparation exited 1"


def test_exit_status_is_read_from_the_end_of_a_gadi_log():
    assert remote.parse_exit_status(LOG_END) == 0
    killed = LOG_END.replace("Exit Status:        0", "Exit Status:        271")
    assert remote.parse_exit_status(killed) == 271
    assert remote.parse_exit_status("Epoch 3 of 100\n") is None


def test_a_job_qstat_has_forgotten_is_fetched_using_its_log(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    ssh = FakeSsh([(r"^tail .*pbs\.log", LOG_END), (r"^qstat", '{"Jobs": {}}')])
    ssh.tar_to_return = results_tar()
    state = remote.fetch(run_dir, ssh, 300, say=lambda text: None)
    assert state["status"] == "DONE"
    assert state["exits"] == {"train": 0}
    assert not any(call.startswith("qstat") for call in ssh.calls)


def test_a_job_neither_log_nor_qstat_knows_is_reported(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", '{"Jobs": {}}')])
    with pytest.raises(BlockedError, match="may never have started"):
        remote.fetch(run_dir, ssh, 300, say=lambda text: None)
    assert not (run_dir / "results").exists()


def flaky_qstat(failures, then):
    replies = iter([*([None] * failures), then])

    def reply(command):
        answer = next(replies)
        if answer is None:
            raise remote.ConnectionLostError("ssh to gadi failed: Connection reset")
        return answer

    return reply


def test_a_dropped_connection_is_tried_again(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", flaky_qstat(2, qstat("123.gadi-pbs", "F", 0)))])
    ssh.tar_to_return = results_tar()
    said = []
    state = remote.fetch(run_dir, ssh, 300, sleep=lambda s: None, say=said.append)
    assert state["status"] == "DONE"
    assert said[:2] == [
        "lost the connection to Gadi (1 of 3); trying again in 300 s",
        "lost the connection to Gadi (2 of 3); trying again in 300 s",
    ]


def test_waiting_stops_after_repeated_dropped_connections(tmp_path):
    run_dir = write_run(tmp_path, {"train": "123.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", flaky_qstat(4, qstat("123.gadi-pbs", "F", 0)))])
    with pytest.raises(remote.ConnectionLostError):
        remote.fetch(run_dir, ssh, 300, sleep=lambda s: None, say=lambda text: None)
    assert sum(call.startswith("qstat") for call in ssh.calls) == 4


def test_ssh_failures_are_told_apart_from_failed_commands():
    ssh = remote.Ssh({"ssh": ["ssh"]})
    with pytest.raises(remote.ConnectionLostError):
        ssh._check("gadi", "qstat", 255, "Connection reset by peer")
    with pytest.raises(BlockedError) as error:
        ssh._check("gadi", "qstat", 153, "qstat: Unknown Job Id")
    assert not isinstance(error.value, remote.ConnectionLostError)
