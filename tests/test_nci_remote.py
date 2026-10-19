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
# `qstat -Q gpuvolta-exec dgxa100-exec normal-exec` on Gadi, 2026-10-19.
QUEUES = """\
Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
gpuvolta-exec        0  1123 yes yes   397   441   282     0     0     0 Exe*
dgxa100-exec         0    51 yes yes    33     8    10     0     0     0 Exe*
normal-exec          0  2292 yes yes   104  1622   552     0     0     4 Exe*
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
ENV = f"/g/data/no50/rainflow/envs/{hashlib.sha256(LOCK).hexdigest()[:16]}"
TABLES = "/g/data/no50/rainflow/data/sydney-0123456789abcdef"
SUMS = "a" * 64
RUN = "/scratch/no50/rh0513/rainflow/runs"


def probe_output(
    env="ready",
    graph="ready",
    dataset=f"{SUMS} gauge_hourly.csv stations.csv report.md",
    account=ACCOUNT,
):
    return (
        "repo=absent\nrepo_head=\nhas_revision=no\n"
        f"env={env}\ngraph={graph}\ndataset {TABLES} {dataset}\n"
        f"--- lquota\n{LQUOTA}--- nci_account\n{account}--- queues\n{QUEUES}"
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

    def receive_tar(self, host, command, destination):
        self._reply(command)
        remote.extract(io.BytesIO(self.tar_to_return), destination)


def numbered_qsub():
    ids = iter(range(100, 200))
    return lambda command: f"{next(ids)}.gadi-pbs\n"


def gadi(probe=None, qsub=None):
    return FakeSsh(
        [
            (r"^id -un$", "rh0513\n"),
            (r"^bash -s", probe or probe_output()),
            (r"qsub", qsub or numbered_qsub()),
        ]
    )


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def model_settings(**changes):
    settings = yaml.safe_load((ROOT / "config/models/hgnn.yaml").read_text())
    settings.update(
        gauge_file="database/standard/gauge_hourly.csv",
        stations_file="database/standard/stations.csv",
        split_file="config/splits/split.json",
        graphs_dir="database/graphs/g",
        epochs=100,
    )
    settings.update(changes)
    return settings


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture
def repo(tmp_path):
    """A committed checkout whose tables are published and graphs not yet built.
    No data exists on this machine: runs read published tables on Gadi."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "config/nci", root / "config/nci")
    write_yaml(
        root / "config/nci/datasets.yaml",
        {
            "datasets": {
                "database/standard": {"path": TABLES, "sha256sums": SUMS, "hours": 8760}
            }
        },
    )
    (root / "config/seeds.yaml").write_text("split_seed: 1\nmodel_seed: 1\n")
    (root / "config/splits").mkdir()
    (root / "config/splits/split.json").write_text("{}\n")
    write_yaml(root / "config/models/model.yaml", model_settings())
    write_yaml(
        root / "config/graphs/g.yaml",
        {
            "data_dir": "database/standard",
            "split_file": "config/splits/split.json",
            "output_dir": "database/graphs/g",
        },
    )
    for folder in ("src/graph", "src/canonical", "src/study"):
        (root / folder).mkdir(parents=True)
        (root / folder / "code.py").write_text(f"# {folder}\n")
    (root / "uv.lock").write_bytes(LOCK)
    (root / ".gitignore").write_text("database/\nexperiments/\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "fixture")
    return root


def config(repo):
    return remote.load_remote_config(repo / "config/nci/remote.yaml")


def plan(repo, ssh=None, queue="auto", model="config/models/model.yaml"):
    return remote.make_plan(Path(model), config(repo), ssh or gadi(), repo, queue)


def commit(repo, path, data):
    write_yaml(repo / path, data)
    git(repo, "add", path)
    git(repo, "commit", "-qm", "change")


# --- reading Gadi's reports ------------------------------------------------------


def test_lquota_rows_are_read_for_one_project():
    rows = remote.parse_lquota(LQUOTA, "no50")
    assert set(rows) == {"scratch", "gdata"}
    assert rows["gdata"]["files"] == 137533
    assert rows["gdata"]["file_quota"] == 202000
    assert rows["gdata"]["quota"] == 50 * 2**30
    assert rows["scratch"]["used"] == pytest.approx(192.37 * 2**30)


def test_quotas_are_checked_by_size_and_by_files():
    rows = remote.parse_lquota(LQUOTA, "no50")
    assert "file-count quota" in remote.headroom_problem(rows, "gdata", 1, 81930)
    assert "size quota" in remote.headroom_problem(rows, "gdata", 40 * 2**30, 1)
    assert remote.headroom_problem(rows, "scratch", 2**30, 81930) is None


def test_available_su_is_read_from_nci_account():
    assert remote.parse_available_su(ACCOUNT) == pytest.approx(5670)


def test_queue_load_is_read_from_qstat():
    assert remote.parse_queue_load(QUEUES) == {
        "gpuvolta": (397, 441),
        "dgxa100": (33, 8),
        "normal": (104, 1622),
    }


def test_qstat_json_gives_state_and_exit_status():
    assert remote.parse_qstat(qstat("1.gadi-pbs", "F", 0), "1.gadi-pbs") == ("F", 0)
    assert remote.parse_qstat(qstat("1.gadi-pbs", "R"), "1.gadi-pbs") == ("R", None)
    with pytest.raises(BlockedError, match="no record"):
        remote.parse_qstat('{"Jobs": {}}', "1.gadi-pbs")


def test_exit_status_is_read_from_the_end_of_a_gadi_log():
    assert remote.parse_exit_status(LOG_END) == 0
    killed = LOG_END.replace("Exit Status:        0", "Exit Status:        271")
    assert remote.parse_exit_status(killed) == 271
    assert remote.parse_exit_status("Epoch 3 of 100\n") is None


# --- configuration and job commands --------------------------------------------


def test_remote_config_loads_and_rejects_smuggled_resources(tmp_path):
    cfg = remote.load_remote_config(ROOT / "config/nci/remote.yaml")
    assert set(cfg["queues"]) == {"gpuvolta", "dgxa100", "normal"}
    cfg["queues"]["gpuvolta"]["mem"] = "90GB,ngpus=4"
    path = tmp_path / "remote.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(BlockedError, match=r"queues\.gpuvolta\.mem"):
        remote.load_remote_config(path)


def test_su_estimate_counts_every_cpu_for_the_whole_walltime():
    assert remote.su_estimate({"ncpus": 16, "walltime": "00:10:00"}, 4.5) == 12
    with pytest.raises(BlockedError, match="HH:MM:SS"):
        remote.walltime_hours("1:00")


def test_storage_directive_mounts_each_project_once():
    directive = remote.storage_directive(
        "/scratch/no50/u/run", "/g/data/no50/rainflow/", "/scratch/no50/x/"
    )
    assert directive == "scratch/no50+gdata/no50"
    with pytest.raises(BlockedError):
        remote.storage_directive("/home/123/u/run")


def test_qsub_command_carries_resources_dependencies_name_and_mail():
    job = {
        "queue": "gpuvolta",
        "ncpus": 12,
        "ngpus": 1,
        "mem": "90GB",
        "jobfs": "10GB",
        "walltime": "08:00:00",
    }
    command = remote.qsub_command(
        "no50",
        job,
        "train.pbs",
        {"A": "/x"},
        "scratch/no50",
        "/x/log",
        ["1.gadi-pbs", "2.gadi-pbs"],
        name="t-fold-0",
        mail="a",
    )
    assert command[:5] == ["qsub", "-P", "no50", "-q", "gpuvolta"]
    assert (
        "ncpus=12,ngpus=1,mem=90GB,jobfs=10GB,walltime=08:00:00,storage=scratch/no50"
        in command
    )
    assert command[-3:] == ["-W", "depend=afterok:1.gadi-pbs:2.gadi-pbs", "train.pbs"]
    assert ["-N", "t-fold-0"] == command[command.index("-N") : command.index("-N") + 2]
    assert ["-m", "a"] == command[command.index("-m") : command.index("-m") + 2]
    cpu = remote.qsub_command(
        "no50", dict(job, queue="normal", ngpus=0), "s", {}, "s", "/l"
    )
    assert not any("ngpus" in part for part in cpu)
    with pytest.raises(BlockedError, match="qsub -v"):
        remote.qsub_command("no50", job, "train.pbs", {"A": "/x,y"}, "s", "/log")


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


# --- what a run reads -------------------------------------------------------------


def test_uncommitted_tracked_edits_are_refused_but_untracked_files_are_not(repo):
    (repo / "notes.md").write_text("untracked\n")
    assert remote.committed_revision(repo) == git(repo, "rev-parse", "HEAD")
    (repo / "config/seeds.yaml").write_text("split_seed: 2\n")
    with pytest.raises(BlockedError, match="commit or stash"):
        remote.committed_revision(repo)


def test_tables_are_read_where_they_are_published(repo):
    revision = git(repo, "rev-parse", "HEAD")
    inputs = remote.plan_inputs(model_settings(), revision, repo)
    assert inputs.tables == {
        "gauge_file": f"{TABLES}/gauge_hourly.csv",
        "stations_file": f"{TABLES}/stations.csv",
    }
    assert inputs.graph_config == "config/graphs/g.yaml"
    assert inputs.hours == 8760


def test_an_unpublished_table_folder_is_refused(repo):
    revision = git(repo, "rev-parse", "HEAD")
    settings = model_settings(gauge_file="database/standard_2022/gauge_hourly.csv")
    with pytest.raises(BlockedError, match="standard_2022 is not published"):
        remote.plan_inputs(settings, revision, repo)


def test_an_uncommitted_split_is_refused(repo):
    revision = git(repo, "rev-parse", "HEAD")
    with pytest.raises(BlockedError, match="not committed"):
        remote.plan_inputs(
            model_settings(split_file="local/split.json"), revision, repo
        )


def test_graphs_dir_must_come_from_exactly_one_graph_config(repo):
    revision = git(repo, "rev-parse", "HEAD")
    with pytest.raises(BlockedError, match="found 0"):
        remote.plan_inputs(
            model_settings(graphs_dir="database/graphs/x"), revision, repo
        )
    commit(
        repo,
        "config/graphs/copy.yaml",
        yaml.safe_load((repo / "config/graphs/g.yaml").read_text()),
    )
    with pytest.raises(BlockedError, match="found 2"):
        remote.plan_inputs(model_settings(), git(repo, "rev-parse", "HEAD"), repo)


def test_graphs_built_for_another_split_are_refused(repo):
    commit(
        repo,
        "config/graphs/g.yaml",
        {
            "data_dir": "database/standard",
            "split_file": "config/splits/other.json",
            "output_dir": "database/graphs/g",
        },
    )
    with pytest.raises(BlockedError, match=r"scores config/splits/split\.json"):
        remote.plan_inputs(model_settings(), git(repo, "rev-parse", "HEAD"), repo)


def test_the_graph_fingerprint_follows_what_builds_graphs_and_nothing_else(repo):
    def fingerprint():
        revision = git(repo, "rev-parse", "HEAD")
        return remote.plan_inputs(model_settings(), revision, repo).graph_fingerprint

    first = fingerprint()
    commit(repo, "config/models/model.yaml", model_settings(hidden_channels=64))
    assert fingerprint() == first  # a model setting does not change the graphs
    (repo / "src/graph/code.py").write_text("# changed\n")
    git(repo, "commit", "-qam", "graph code")
    second = fingerprint()
    assert second != first
    entry = {"path": TABLES, "sha256sums": "b" * 64, "hours": 8760}
    commit(repo, "config/nci/datasets.yaml", {"datasets": {"database/standard": entry}})
    assert fingerprint() not in (first, second)


# --- queue and walltime -----------------------------------------------------------


def test_auto_picks_the_gpu_queue_with_fewer_jobs_waiting_per_job_running(repo):
    cfg = config(repo)
    load = remote.parse_queue_load(QUEUES)
    assert remote.choose_queue("auto", load, cfg) == "gpuvolta"
    load["gpuvolta"] = (900, 100)
    assert remote.choose_queue("auto", load, cfg) == "dgxa100"
    assert remote.choose_queue("normal", load, cfg) == "normal"
    with pytest.raises(BlockedError, match=r"not in remote\.yaml"):
        remote.choose_queue("gpuhopper", load, cfg)


def test_fold_walltime_comes_from_the_measured_epoch_time(repo):
    cfg = config(repo)
    # 100 epochs x 24.2 s per 1,000 hours x 8.76 x 1.3 margin + 15 min = 7.9 h -> 8 h
    assert remote.fold_walltime(cfg, "gpuvolta", "raingauge+radar", 8760, 100) == (
        8.0,
        False,
    )
    assert remote.epoch_seconds(cfg, "gpuvolta", "raingauge+radar") == (24.2, True)
    # An unmeasured layer set is budgeted at 1.5 times the slowest measured one.
    assert remote.epoch_seconds(cfg, "gpuvolta", "raingauge+satellite") == (
        pytest.approx(1.5 * 27.1),
        False,
    )
    # More than the queue allows is capped and flagged.
    assert remote.fold_walltime(cfg, "normal", "raingauge", 200000, 100) == (48.0, True)
    assert remote.format_walltime(8.0) == "08:00:00"
    assert remote.format_walltime(0.75) == "00:45:00"


# --- planning ------------------------------------------------------------------------


def test_a_ready_environment_and_graph_set_mean_five_folds_then_scoring(repo):
    result = plan(repo)
    assert result.queue == "gpuvolta"
    assert result.env_dir == ENV
    assert result.graph_dir.startswith("/scratch/no50/rainflow/graphs/")
    assert [job.name for job in result.jobs] == [
        "fold-0",
        "fold-1",
        "fold-2",
        "fold-3",
        "fold-4",
        "score",
    ]
    assert all(job.after == [] for job in result.jobs[:5])
    assert result.jobs[-1].after == [f"fold-{fold}" for fold in range(5)]
    # 100 epochs of G at 7.5 s per 1,000 hours over 8,760 hours: 2.62 h -> 2.75 h
    assert result.walltime == "02:45:00"
    assert result.su_estimate == pytest.approx(5 * 12 * 2.75 * 3.0 + 8 * 1 * 2.0)
    text = remote.describe(result, config(repo))
    assert "5 jobs on gpuvolta" in text
    assert "gpuvolta 397 queued / 441 running" in text


def test_a_missing_environment_and_graph_set_are_built_first(repo):
    result = plan(repo, gadi(probe_output(env="absent", graph="absent")))
    by_name = {job.name: job for job in result.jobs}
    assert list(by_name)[:2] == ["prepare", "graph"]
    assert by_name["graph"].after == ["prepare"]
    assert by_name["fold-3"].after == ["prepare", "graph"]
    assert by_name["graph"].variables == {
        "NCI_GRAPH_CONFIG": "config/graphs/g.yaml",
        "NCI_DATA_DIR": TABLES,
        "NCI_SPLIT_FILE": "config/splits/split.json",
        "NCI_SEEDS": "config/seeds.yaml",
        "NCI_GRAPH_DIR": result.graph_dir,
    }


def test_folds_on_the_cpu_queue_train_on_cpu(repo):
    result = plan(repo, queue="normal")
    fold = result.jobs[0]
    assert fold.variables["NCI_DEVICE"] == "cpu"
    assert fold.job["ngpus"] == 0


@pytest.mark.parametrize(
    ("dataset", "problem"),
    [
        ("missing", "is not published on Gadi"),
        (f"{'c' * 64} gauge_hourly.csv stations.csv", "different tables"),
        (f"{SUMS} gauge_hourly.csv", "stations.csv is not in the published folder"),
    ],
)
def test_tables_that_are_not_on_gadi_as_listed_are_refused(repo, dataset, problem):
    with pytest.raises(BlockedError, match=problem):
        plan(repo, gadi(probe_output(dataset=dataset)))


def test_a_run_the_su_balance_cannot_cover_is_refused(repo):
    account = ACCOUNT.replace("5.67 KSU", "30.00 SU")
    with pytest.raises(BlockedError, match="30 SU left"):
        plan(repo, gadi(probe_output(account=account)))


def test_only_hgnn_runs_are_sent(repo):
    commit(repo, "config/models/idw.yaml", model_settings(method="idw"))
    with pytest.raises(BlockedError, match="only hgnn"):
        plan(repo, model="config/models/idw.yaml")


def test_declining_the_plan_submits_nothing(repo):
    ssh = gadi()
    code = remote.main(
        ["submit", "config/models/model.yaml"],
        ssh_factory=lambda cfg: ssh,
        confirm=lambda prompt: "no",
        root=repo,
    )
    assert code == 1
    assert not ssh.files
    assert not any("qsub" in call or "mkdir" in call for call in ssh.calls)
    assert not (repo / "experiments").exists()


# --- submitting ----------------------------------------------------------------------


def submitted(repo, ssh, *extra):
    code = remote.main(
        ["submit", "config/models/model.yaml", "--label", "t", "--yes", *extra],
        ssh_factory=lambda cfg: ssh,
        root=repo,
    )
    (run_dir,) = (repo / "experiments/nci").iterdir()
    return code, run_dir, json.loads((run_dir / "state.json").read_text())


def qsub_calls(ssh):
    return {
        re.search(r"-N (\S+)", call)[1].split("-", 1)[1]: call
        for call in ssh.calls
        if call.startswith("cd ") and " qsub " in call
    }


def test_submit_ships_the_commit_and_queues_every_fold_then_scoring(repo, tmp_path):
    ssh = gadi(probe_output(env="absent", graph="absent"))
    code, run_dir, state = submitted(repo, ssh)
    assert code == 0
    assert state["status"] == "SUBMITTED"
    assert state["jobs"] == {
        "prepare": "100.gadi-pbs",
        "graph": "101.gadi-pbs",
        **{f"fold-{fold}": f"{102 + fold}.gadi-pbs" for fold in range(5)},
        "score": "107.gadi-pbs",
    }

    (bundle,) = ssh.files
    (tmp_path / "code.bundle").write_bytes(bundle)
    git(tmp_path, "clone", "-q", "code.bundle", "clone")
    assert git(tmp_path / "clone", "rev-parse", "HEAD") == state["revision"]

    run = f"{RUN}/{run_dir.name}"
    settings = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert settings["gauge_file"] == f"{TABLES}/gauge_hourly.csv"
    assert settings["graphs_dir"] == state["graph_dir"]
    assert settings["split_file"] == "config/splits/split.json"
    assert settings["device"] == "cuda"
    assert any(f"ln -s {ENV} {run}/code/.venv" in call for call in ssh.calls)
    (prepare,) = [call for call in ssh.calls if "prepare_environment.pbs" in call]
    assert "-m a" in prepare

    calls = qsub_calls(ssh)
    assert "depend=afterok:100.gadi-pbs" in calls["graph"]
    for fold in range(5):
        call = calls[f"fold-{fold}"]
        assert "depend=afterok:100.gadi-pbs:101.gadi-pbs" in call
        assert f"NCI_FOLDS={fold}" in call
        assert f"NCI_OUTPUT_DIR={run}/output/fold-{fold}" in call
        assert "-q gpuvolta" in call and "-m a " in call
        assert "storage=scratch/no50+gdata/no50" in call
    assert (
        "depend=afterok:102.gadi-pbs:103.gadi-pbs:104.gadi-pbs:105.gadi-pbs"
        in (calls["score"])
    )
    assert "-m ae" in calls["score"]


def test_a_refused_job_leaves_the_submitted_ones_on_record(repo, capsys):
    ids = numbered_qsub()

    def refuse_scoring(command):
        if "score.pbs" in command:
            raise BlockedError("qsub: Job violates queue and/or server resource limits")
        return ids(command)

    code, _, state = submitted(repo, gadi(qsub=refuse_scoring))
    assert code == 1
    assert state["status"] == "SUBMIT_FAILED"
    assert list(state["jobs"]) == [f"fold-{fold}" for fold in range(5)]
    assert "resource limits" in state["error"]
    assert "qdel 100.gadi-pbs 101.gadi-pbs" in capsys.readouterr().err


# --- status and fetching -----------------------------------------------------------


def results_tar(report="# Model run\n\n| fold | rmse |\n"):
    buffer = io.BytesIO()
    remote.write_tar(
        buffer,
        [
            (report.encode(), "output/scored/report.md"),
            (b"history\n", "output/fold-0/history.csv"),
            (b"PBS job: 123\n", "logs/score.log"),
        ],
    )
    return buffer.getvalue()


def write_run(tmp_path, jobs, after=None, status="SUBMITTED"):
    run_dir = tmp_path / "experiments/nci/run"
    run_dir.mkdir(parents=True)
    state = {
        "status": status,
        "login_host": "gadi",
        "transfer_host": "gadi-dm",
        "remote": {"run": f"{RUN}/run"},
        "logs": {name: f"{RUN}/run/logs/{name}.log" for name in jobs},
        "after": after or {},
        "jobs": jobs,
    }
    (run_dir / "state.json").write_text(json.dumps(state))
    return run_dir


def by_job(answers):
    """qstat replies by job id: {id: reply}."""
    return [(rf"^qstat .* {re.escape(job)}$", reply) for job, reply in answers.items()]


def test_an_unfinished_run_is_not_downloaded(tmp_path):
    run_dir = write_run(tmp_path, {"fold-0": "1.gadi-pbs", "score": "2.gadi-pbs"})
    ssh = FakeSsh(
        by_job(
            {
                "1.gadi-pbs": qstat("1.gadi-pbs", "F", 0),
                "2.gadi-pbs": qstat("2.gadi-pbs", "Q"),
            }
        )
    )
    state = remote.fetch(run_dir, ssh, 300)
    assert state["status"] == "WAITING: score not finished"
    assert not (run_dir / "results").exists()


def test_a_finished_run_downloads_once_without_predictions(tmp_path):
    run_dir = write_run(tmp_path, {"fold-0": "1.gadi-pbs", "score": "2.gadi-pbs"})
    ssh = FakeSsh(
        by_job(
            {
                "1.gadi-pbs": qstat("1.gadi-pbs", "F", 0),
                "2.gadi-pbs": qstat("2.gadi-pbs", "F", 0),
            }
        )
    )
    ssh.tar_to_return = results_tar()
    state = remote.fetch(run_dir, ssh, 300)
    assert state["status"] == "DONE"
    assert (run_dir / "results/output/scored/report.md").is_file()
    (download,) = [call for call in ssh.calls if "tar -cf" in call]
    assert "--exclude=predictions.csv" in download
    with pytest.raises(BlockedError, match="never overwritten"):
        remote.fetch(run_dir, ssh, 300)


def test_fetching_with_predictions_downloads_them(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    ssh = FakeSsh(by_job({"2.gadi-pbs": qstat("2.gadi-pbs", "F", 0)}))
    ssh.tar_to_return = results_tar()
    remote.fetch(run_dir, ssh, 300, with_predictions=True)
    (download,) = [call for call in ssh.calls if "tar -cf" in call]
    assert "--exclude" not in download


def test_a_failed_fold_is_named_and_the_dropped_scoring_job_is_explained(tmp_path):
    run_dir = write_run(
        tmp_path,
        {"fold-2": "1.gadi-pbs", "score": "2.gadi-pbs"},
        after={"score": ["fold-2"]},
    )
    ssh = FakeSsh(
        by_job(
            {"1.gadi-pbs": qstat("1.gadi-pbs", "F", 271), "2.gadi-pbs": '{"Jobs": {}}'}
        )
    )
    ssh.tar_to_return = results_tar()
    state = remote.fetch(run_dir, ssh, 300)
    assert state["exits"] == {"fold-2": 271, "score": None}
    assert state["status"] == "FAILED: fold-2 job exit status 271"


def test_a_failed_report_is_not_called_done(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    ssh = FakeSsh(by_job({"2.gadi-pbs": qstat("2.gadi-pbs", "F", 0)}))
    ssh.tar_to_return = results_tar("# Model run\n\nStatus: FAILED\n")
    state = remote.fetch(run_dir, ssh, 300)
    assert state["status"] == "FAILED: report.md does not show a completed run"


def test_a_job_qstat_has_forgotten_is_read_from_its_log(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    ssh = FakeSsh([(r"^tail .*score\.log", LOG_END), (r"^qstat", '{"Jobs": {}}')])
    ssh.tar_to_return = results_tar()
    assert remote.fetch(run_dir, ssh, 300)["status"] == "DONE"
    assert not any(call.startswith("qstat") for call in ssh.calls)


def test_a_job_neither_log_nor_qstat_knows_is_reported(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", '{"Jobs": {}}')])
    with pytest.raises(BlockedError, match="may never have started"):
        remote.fetch(run_dir, ssh, 300)
    assert not (run_dir / "results").exists()


def flaky_qstat(failures, then):
    replies = iter([*([None] * failures), then])

    def reply(command):
        answer = next(replies)
        if answer is None:
            raise remote.ConnectionLostError("ssh to gadi failed: Connection reset")
        return answer

    return reply


def test_waiting_tries_a_dropped_connection_again_then_gives_up(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    ssh = FakeSsh([(r"^qstat", flaky_qstat(2, qstat("2.gadi-pbs", "F", 0)))])
    ssh.tar_to_return = results_tar()
    said = []
    state = remote.fetch(
        run_dir, ssh, 300, block=True, sleep=lambda s: None, say=said.append
    )
    assert state["status"] == "DONE"
    assert said[:2] == [
        "lost the connection to Gadi (1 of 3); trying again in 300 s",
        "lost the connection to Gadi (2 of 3); trying again in 300 s",
    ]
    shutil.rmtree(run_dir / "results")
    ssh = FakeSsh([(r"^qstat", flaky_qstat(4, qstat("2.gadi-pbs", "F", 0)))])
    with pytest.raises(remote.ConnectionLostError):
        remote.fetch(
            run_dir, ssh, 300, block=True, sleep=lambda s: None, say=lambda text: None
        )


def test_fetch_with_no_run_named_takes_every_submitted_unfetched_run(tmp_path):
    run_dir = write_run(tmp_path, {"score": "2.gadi-pbs"})
    other = tmp_path / "experiments/nci/failed"
    other.mkdir()
    (other / "state.json").write_text(json.dumps({"status": "SUBMIT_FAILED"}))
    assert remote.unfetched_runs(tmp_path) == [run_dir]


def test_ssh_failures_are_told_apart_from_failed_commands():
    ssh = remote.Ssh({"ssh": ["ssh"]})
    with pytest.raises(remote.ConnectionLostError):
        ssh._check("gadi", "qstat", 255, "Connection reset by peer")
    with pytest.raises(BlockedError) as error:
        ssh._check("gadi", "qstat", 153, "qstat: Unknown Job Id")
    assert not isinstance(error.value, remote.ConnectionLostError)


# --- a teammate's first check ------------------------------------------------------


def doctor_output(member="yes", python="yes", lquota=LQUOTA):
    return (
        f"user=ab1234\nmember={member}\npython={python}\n"
        f"--- lquota\n{lquota}--- nci_account\n{ACCOUNT}"
    )


def test_doctor_passes_a_ready_account():
    ssh = FakeSsh([(r"^bash -s", doctor_output())])
    said = []
    assert (
        remote.doctor(
            remote.load_remote_config(ROOT / "config/nci/remote.yaml"), ssh, said.append
        )
        == []
    )
    assert "Gadi user: ab1234" in said


def test_doctor_names_each_problem():
    full = LQUOTA.replace("106239   202000", "199000   202000")
    ssh = FakeSsh([(r"^bash -s", doctor_output("no", "no", full))])
    problems = remote.doctor(
        remote.load_remote_config(ROOT / "config/nci/remote.yaml"), ssh, lambda t: None
    )
    assert any("not in project no50" in problem for problem in problems)
    assert any("shared Python" in problem for problem in problems)
    assert any("scratch is nearly out of files" in problem for problem in problems)


def test_doctor_explains_a_failed_login():
    def refuse(command):
        raise remote.ConnectionLostError("ssh to gadi failed: Permission denied")

    ssh = FakeSsh([(r"^bash -s", refuse)])
    (problem,) = remote.doctor(
        remote.load_remote_config(ROOT / "config/nci/remote.yaml"), ssh, lambda t: None
    )
    assert "SSH key" in problem


# --- resuming a run whose folds did not all finish ---------------------------------


def stopped_run(tmp_path, fold_exits, graph_exit=0):
    jobs = {
        "graph": "1.gadi-pbs",
        **{f"fold-{f}": f"{10 + f}.gadi-pbs" for f in range(5)},
        "score": "20.gadi-pbs",
    }
    after = {"score": [f"fold-{f}" for f in range(5)]}
    after.update({f"fold-{f}": ["graph"] for f in range(5)})
    run_dir = write_run(tmp_path, jobs, after=after)
    state = json.loads((run_dir / "state.json").read_text())
    state["run"] = "2026-10-19_1300_t"
    state["queue"] = "gpuvolta"
    state["storage"] = "scratch/no50+gdata/no50"
    state.update(layers="raingauge+radar", hours=8760, epochs=100)
    state["remote"].update(
        code=f"{RUN}/run/code",
        config=f"{RUN}/run/run.yaml",
        output=f"{RUN}/run/output",
        logs=f"{RUN}/run/logs",
    )
    (run_dir / "state.json").write_text(json.dumps(state))
    answers = {"1.gadi-pbs": qstat("1.gadi-pbs", "F", graph_exit)}
    for fold, code in enumerate(fold_exits):
        job = f"{10 + fold}.gadi-pbs"
        answers[job] = qstat(job, "F", code) if code is not None else '{"Jobs": {}}'
    answers["20.gadi-pbs"] = '{"Jobs": {}}'  # PBS dropped it when a fold failed
    return run_dir, answers


def test_resume_reruns_only_the_folds_that_failed_then_scores_again(tmp_path):
    run_dir, answers = stopped_run(tmp_path, [0, 271, 0, 271, 0])
    ssh = FakeSsh(
        [
            *by_job(answers),
            (r"last\.pt", "fold-1\n"),  # fold 1 saved a checkpoint, fold 3 did not
            (r"qsub", numbered_qsub()),
        ]
    )
    state = remote.resume(
        run_dir, remote.load_remote_config(ROOT / "config/nci/remote.yaml"), ssh
    )
    assert state["status"] == "RESUMED"
    assert state["jobs"]["fold-0"] == "10.gadi-pbs"  # kept
    assert state["jobs"]["fold-1"] == "100.gadi-pbs"
    assert state["jobs"]["fold-3"] == "101.gadi-pbs"
    assert state["jobs"]["score"] == "102.gadi-pbs"
    calls = [call for call in ssh.calls if " qsub " in call]
    assert "NCI_RESUME=1" in calls[0] and "NCI_FOLDS=1" in calls[0]
    assert "NCI_RESUME" not in calls[1] and "NCI_FOLDS=3" in calls[1]
    assert "depend=afterok:100.gadi-pbs:101.gadi-pbs" in calls[2]
    assert state["logs"]["fold-1"].endswith("fold-1.2.log")
    assert state["previous"][0]["fold-1"] == "11.gadi-pbs"
    assert remote.unfetched_runs(tmp_path) == [run_dir]


def test_resume_waits_for_unfinished_jobs_and_refuses_a_failed_graph_build(tmp_path):
    cfg = remote.load_remote_config(ROOT / "config/nci/remote.yaml")
    run_dir, answers = stopped_run(tmp_path, [0, 0, 0, 0, 0])
    answers["20.gadi-pbs"] = qstat("20.gadi-pbs", "R")
    with pytest.raises(BlockedError, match="wait for score"):
        remote.resume(run_dir, cfg, FakeSsh(by_job(answers)))
    shutil.rmtree(run_dir.parent)
    run_dir, answers = stopped_run(tmp_path, [None] * 5, graph_exit=1)
    with pytest.raises(BlockedError, match="the graph job failed"):
        remote.resume(run_dir, cfg, FakeSsh(by_job(answers)))


# --- warnings: Gadi can take the run, but something deserves a second look --------


def test_a_comfortable_run_has_no_warnings(repo):
    assert plan(repo).warnings == []


def test_a_run_that_would_use_much_of_the_su_left_is_flagged(repo):
    account = ACCOUNT.replace("5.67 KSU", "1.20 KSU")
    (warning,) = plan(repo, gadi(probe_output(account=account))).warnings
    assert "of the 1200 SU the project has left" in warning


def test_a_nearly_full_filesystem_is_flagged(repo):
    full = LQUOTA.replace("106239   202000", "195000   202000")
    (warning,) = plan(repo, gadi(probe_output().replace(LQUOTA, full))).warnings
    assert warning.startswith("scratch holds 195,000 of the 202,000 files")


def test_a_crowded_queue_is_flagged(repo):
    result = plan(repo, queue="dgxa100")
    assert any(
        "dgxa100 has 33 jobs waiting for 8 running" in w for w in result.warnings
    )
    assert "WARNING: dgxa100 has 33" in remote.describe(result, config(repo))


def test_space_for_a_graph_build_is_checked(repo):
    tight = LQUOTA.replace("192.37 GiB   1.00 TiB", "1015.00 GiB   1.00 TiB")
    with pytest.raises(BlockedError, match=r"scratch: .* size quota"):
        plan(repo, gadi(probe_output(graph="absent").replace(LQUOTA, tight)))
    plan(repo, gadi(probe_output(graph="ready").replace(LQUOTA, tight)))
