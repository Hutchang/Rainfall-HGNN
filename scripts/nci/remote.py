"""Run Component 06 experiments on NCI Gadi from a clean local checkout.

    uv run python -m scripts.nci.remote plan   config/models/<run>.yaml
    uv run python -m scripts.nci.remote submit config/models/<run>.yaml --label NAME
    uv run python -m scripts.nci.remote status [experiments/nci/<run> ...]
    uv run python -m scripts.nci.remote fetch  [experiments/nci/<run> ...]
    uv run python -m scripts.nci.remote doctor

Code travels as a git bundle of the committed revision, so Gadi runs exactly that
commit. Tables are never uploaded: they are published once on Gadi
(scripts/nci/publish_tables.pbs) and listed in config/nci/datasets.yaml. Graphs are
built on Gadi into a cache the project shares, named by everything that decides them,
and reused by later runs. Every fold trains as its own job, on the GPU queue with the
shorter wait unless you choose one, with a walltime estimated from measured epoch
times. A scoring job joins the folds and scores every method once all five have
finished, and PBS emails you when it ends. Nothing on Gadi is overwritten. SSH must
already log in without a password prompt; see docs/nci/ONBOARDING.md.
"""

import argparse
import hashlib
import io
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.models.run import read_settings, validate_settings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("config/nci/remote.yaml")
DATASETS = Path("config/nci/datasets.yaml")
RUNS_DIR = Path("experiments/nci")
CONFIG_KEYS = (
    "project",
    "login_host",
    "transfer_host",
    "ssh",
    "scratch_root",
    "shared_root",
    "graph_cache",
    "queues",
    "graph_job",
    "score_job",
    "epoch_seconds_per_1000_hours",
    "walltime_margin",
    "poll_seconds",
)
QUEUE_KEYS = ("ncpus", "ngpus", "mem", "jobfs", "su_per_cpu_hour", "device")
JOB_KEYS = ("queue", "ncpus", "mem", "jobfs", "walltime", "su_per_cpu_hour")
GPU_QUEUES = ("gpuvolta", "dgxa100")
FOLDS = range(5)
ROLES = ("train", "validation", "test")
LAYERS = ("raingauge", "radar", "satellite")
TABLE_KEYS = ("gauge_file", "stations_file")
COMMITTED_KEYS = ("split_file", "seeds_file")
# Code a graph set depends on, besides its config, split, seeds and tables.
GRAPH_CODE = ("src/graph", "src/canonical", "src/study", "uv.lock")
SIZE_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}
SU_UNITS = {"SU": 1, "KSU": 1e3, "MSU": 1e6}
ENVIRONMENT_FILES = 32000  # files a new shared environment adds to gdata
GRAPH_BYTES = 15 * 2**30  # the largest graph set so far (G+R+S, 2021-2022) is 12 GB
# When the plan warns: a run that could use more than this share of the SU left, a
# filesystem fuller than this, a queue with more than this many jobs waiting for
# each one running.
SU_SHARE_WARNING = 0.25
QUOTA_WARNING = 0.9
QUEUE_WARNING = 2
JOB_STATES = {
    "Q": "queued",
    "H": "held",
    "W": "waiting",
    "R": "running",
    "E": "exiting",
    "F": "finished",
}

# Read-only facts about the remote side: key=value lines, one `dataset` line per table
# folder, then the raw lquota, nci_account and queue reports.
PROBE = r"""
repo=$1 revision=$2 env=$3 graph=$4 project=$5
shift 5
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 \
    && echo repo=present || echo repo=absent
echo "repo_head=$(git -C "$repo" rev-parse HEAD 2>/dev/null)"
git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null \
    && echo has_revision=yes || echo has_revision=no
[[ -f $env/READY ]] && echo env=ready || echo env=absent
complete=yes
[[ -f $graph/READY ]] || complete=no
for fold in 0 1 2 3 4; do
    for role in train validation test; do
        [[ -f $graph/fold-$fold/$role.pt ]] || complete=no
    done
done
[[ $complete == yes ]] && echo graph=ready || echo graph=absent
for folder in "$@"; do
    if [[ -f $folder/READY && -f $folder/SHA256SUMS ]]; then
        echo "dataset $folder $(sha256sum < "$folder/SHA256SUMS" | cut -d' ' -f1)" \
            "$(awk '{print $2}' "$folder/SHA256SUMS" | tr '\n' ' ')"
    else
        echo "dataset $folder missing"
    fi
done
echo '--- lquota'
lquota
echo '--- nci_account'
nci_account -P "$project"
echo '--- queues'
qstat -Q gpuvolta-exec dgxa100-exec normal-exec
"""


class BlockedError(Exception):
    """A check failed; the message says what to fix."""


class ConnectionLostError(BlockedError):
    """ssh itself failed (exit 255), as opposed to the command it ran."""


def quote(*parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def single_value(name, value):
    if re.search(r"[\s,=+]", str(value)):
        raise BlockedError(f"{name} must be a single qsub value, got {value!r}")


def load_remote_config(path):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise BlockedError(f"{path} must contain a YAML mapping")
    missing = [key for key in CONFIG_KEYS if key not in cfg]
    if missing:
        raise BlockedError(f"{path} is missing {', '.join(missing)}")
    for queue, profile in cfg["queues"].items():
        single_value("queue name", queue)
        for key in QUEUE_KEYS:
            if key not in profile:
                raise BlockedError(f"{path}: queues.{queue} is missing {key}")
        for key in ("ncpus", "ngpus", "mem", "jobfs"):
            single_value(f"queues.{queue}.{key}", profile[key])
        if profile["device"] not in ("cpu", "cuda"):
            raise BlockedError(f"queues.{queue}.device must be cpu or cuda")
        walltime_hours(profile.get("max_walltime", "48:00:00"))
    for job in ("graph_job", "score_job"):
        for key in JOB_KEYS:
            if key not in cfg[job]:
                raise BlockedError(f"{path}: {job} is missing {key}")
            if key != "su_per_cpu_hour":
                single_value(f"{job}.{key}", cfg[job][key])
        walltime_hours(cfg[job]["walltime"])
    return cfg


def walltime_hours(text):
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", str(text))
    if not match:
        raise BlockedError(f"walltime must be HH:MM:SS, got {text!r}")
    hours, minutes, seconds = map(int, match.groups())
    return hours + minutes / 60 + seconds / 3600


def format_walltime(hours):
    minutes = round(hours * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def su_estimate(job, rate):
    """The most the job can be charged: every CPU for the whole walltime."""
    return int(job["ncpus"]) * walltime_hours(job["walltime"]) * float(rate)


def git(*args, root=ROOT):
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise BlockedError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def committed_revision(root=ROOT):
    """HEAD, refusing edits to tracked files: Gadi must run exactly this revision."""
    changed = git("status", "--porcelain", "--untracked-files=no", root=root)
    if changed:
        raise BlockedError(
            "commit or stash these changes first; Gadi runs committed code only:\n"
            + changed
        )
    return git("rev-parse", "HEAD", root=root)


def committed_bytes(path, revision, root=ROOT):
    """A file exactly as committed, which is what Gadi checks out."""
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise BlockedError(f"{path} is not committed at {revision[:12]}")
    return result.stdout


def committed_text(path, revision, root=ROOT):
    return committed_bytes(path, revision, root).decode()


def object_id(path, revision, root=ROOT):
    """git's id for a committed file or folder; it changes with any content."""
    try:
        return git("rev-parse", f"{revision}:{path}", root=root)
    except BlockedError:
        raise BlockedError(f"{path} is not committed at {revision[:12]}") from None


def load_datasets(revision, root=ROOT):
    """Published table folders, keyed by the folder configs read on your machine."""
    entries = yaml.safe_load(committed_text(DATASETS.as_posix(), revision, root))
    datasets = (entries or {}).get("datasets") or {}
    for key, entry in datasets.items():
        if (
            not isinstance(entry, dict)
            or not str(entry.get("path", "")).startswith("/g/data/")
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256sums", "")))
            or type(entry.get("hours")) is not int
            or entry["hours"] < 1
        ):
            raise BlockedError(
                f"{DATASETS}: {key} needs path (under /g/data), sha256sums and hours, "
                "as printed by scripts/nci/publish_tables.pbs"
            )
    return datasets


def dataset_for(folder, datasets):
    key = Path(folder).as_posix()
    if key not in datasets:
        raise BlockedError(
            f"{key} is not published on Gadi. Publish it once with "
            "scripts/nci/publish_tables.pbs and add the printed entry to "
            f"{DATASETS} under {key}; published: {', '.join(datasets) or 'none'}"
        )
    return key, datasets[key]


def find_graph_config(graphs_dir, revision, root=ROOT):
    """The committed config/graphs/*.yaml whose output_dir is graphs_dir."""
    names = git("ls-tree", "--name-only", revision, "config/graphs/", root=root)
    found = []
    for name in names.splitlines():
        if name.endswith(".yaml"):
            config = yaml.safe_load(committed_text(name, revision, root)) or {}
            if Path(str(config.get("output_dir", ""))) == Path(graphs_dir):
                found.append((name, config))
    if len(found) != 1:
        raise BlockedError(
            f"graphs_dir {graphs_dir} must be the output_dir of exactly one committed "
            f"config/graphs/*.yaml; found {len(found)}"
        )
    return found[0]


def layer_key(settings):
    chosen = settings.get("input_layers", ["raingauge"])
    return "+".join(layer for layer in LAYERS if layer in chosen)


@dataclass
class Inputs:
    """Where a run's inputs are on Gadi and what decides its graphs."""

    tables: dict  # model config key -> path on Gadi
    datasets: dict  # local folder -> datasets.yaml entry
    graph_config: str
    graph_settings: dict
    graph_fingerprint: str
    hours: int


def plan_inputs(settings, revision, root=ROOT):
    """Resolve the model config's inputs without reading any data on this machine."""
    datasets = load_datasets(revision, root)
    used, tables = {}, {}
    for key in TABLE_KEYS:
        folder, entry = dataset_for(Path(settings[key]).parent, datasets)
        used[folder] = entry
        tables[key] = f"{entry['path']}/{Path(settings[key]).name}"
    for key in COMMITTED_KEYS:
        object_id(settings[key], revision, root)
    graph_config, graph_settings = find_graph_config(
        settings["graphs_dir"], revision, root
    )
    if Path(str(graph_settings.get("split_file"))) != Path(settings["split_file"]):
        raise BlockedError(
            f"{graph_config} builds graphs for {graph_settings.get('split_file')} but "
            f"the model config scores {settings['split_file']}"
        )
    data_dir, data = dataset_for(graph_settings.get("data_dir", ""), datasets)
    used[data_dir] = data
    parts = [
        object_id(graph_config, revision, root),
        object_id(settings["split_file"], revision, root),
        object_id(settings["seeds_file"], revision, root),
        data["sha256sums"],
        *(object_id(path, revision, root) for path in GRAPH_CODE),
    ]
    fingerprint = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return Inputs(
        tables, used, graph_config, graph_settings, fingerprint, data["hours"]
    )


def epoch_seconds(cfg, queue, layers):
    """(seconds per epoch per 1,000 hours, whether it was measured for these layers)."""
    known = cfg["epoch_seconds_per_1000_hours"].get(queue, {})
    if layers in known:
        return float(known[layers]), True
    if not known:
        raise BlockedError(f"no epoch timings for queue {queue} in remote.yaml")
    return 1.5 * max(map(float, known.values())), False


def fold_walltime(cfg, queue, layers, hours, epochs):
    """Hours for one fold to run every epoch, capped at the queue's limit."""
    seconds, _ = epoch_seconds(cfg, queue, layers)
    estimate = epochs * seconds * hours / 1000 / 3600 * cfg["walltime_margin"] + 0.25
    estimate = max(0.5, math.ceil(estimate * 4) / 4)
    limit = walltime_hours(cfg["queues"][queue].get("max_walltime", "48:00:00"))
    return min(estimate, limit), estimate > limit


def parse_queue_load(text):
    """{queue: (queued, running)} from `qstat -Q <queue>-exec ...`."""
    load = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].endswith("-exec") and parts[5].isdigit():
            load[parts[0].removesuffix("-exec")] = (int(parts[5]), int(parts[6]))
    return load


def choose_queue(requested, load, cfg):
    """The asked-for queue, or for `auto` the GPU queue with the fewest jobs queued
    for each one running."""
    if requested != "auto":
        if requested not in cfg["queues"]:
            raise BlockedError(
                f"queue {requested} is not in remote.yaml; "
                f"choose from {', '.join(cfg['queues'])} or auto"
            )
        return requested
    candidates = [queue for queue in GPU_QUEUES if queue in cfg["queues"]]
    if not candidates:
        raise BlockedError("remote.yaml lists no GPU queue for --queue auto")
    return min(
        candidates,
        key=lambda queue: (
            (load.get(queue, (0, 0))[0] + 1) / (load.get(queue, (0, 0))[1] + 1)
        ),
    )


def remote_paths(cfg, user, run_id):
    """Decides where everything lives on Gadi."""
    fill = {"project": cfg["project"], "user": user}
    scratch = cfg["scratch_root"].format(**fill)
    run = f"{scratch}/runs/{run_id}"
    return {
        "root": scratch,
        "repo": f"{scratch}/repo",
        "bundles": f"{scratch}/bundles",
        "runs": f"{scratch}/runs",
        "run": run,
        "code": f"{run}/code",
        "config": f"{run}/run.yaml",
        "output": f"{run}/output",
        "logs": f"{run}/logs",
        "shared": cfg["shared_root"].format(**fill),
        "graph_cache": cfg["graph_cache"].format(**fill),
    }


def storage_directive(*paths):
    """The PBS storage list that mounts every project the paths live in."""
    found = []
    for path in paths:
        match = re.match(r"/(scratch|g/data)/(\w+)/", path)
        if not match:
            raise BlockedError(
                f"{path} is not under /scratch/<project> or /g/data/<project>"
            )
        item = f"{'gdata' if match[1] == 'g/data' else 'scratch'}/{match[2]}"
        if item not in found:
            found.append(item)
    return "+".join(found)


def qsub_command(project, job, script, variables, storage, log, after=(), **flags):
    """flags: name (-N) and mail (-m), both optional."""
    resources = [
        f"{key}={job[key]}"
        for key in ("ncpus", "ngpus", "mem", "jobfs", "walltime")
        if key in job and not (key == "ngpus" and int(job[key]) == 0)
    ]
    for name, value in variables.items():
        if re.search(r"[,\s'\"]", str(value)):
            raise BlockedError(f"{name}={value!r} cannot be passed through qsub -v")
    command = [
        "qsub",
        "-P",
        project,
        "-q",
        str(job["queue"]),
        "-l",
        ",".join([*resources, f"storage={storage}"]),
        "-o",
        log,
    ]
    if flags.get("name"):
        command += ["-N", flags["name"]]
    if flags.get("mail"):
        command += ["-m", flags["mail"]]
    if variables:
        command += [
            "-v",
            ",".join(f"{name}={value}" for name, value in variables.items()),
        ]
    if after:
        command += ["-W", "depend=afterok:" + ":".join(after)]
    return [*command, script]


def parse_lquota(text, project):
    """One project's rows from `lquota`, keyed by filesystem, sizes in bytes."""
    amount = r"([\d.]+)\s+(\S+)"  # a number and its unit
    pattern = re.compile(
        rf"^\s*(\S+)\s+(scratch|gdata)\s+{amount}\s+{amount}\s+[\d.]+\s+\S+"
        r"\s+(\d+)\s+(\d+)\s+\d+\s*$"
    )
    rows = {}
    for line in text.splitlines():
        match = pattern.match(line)
        if match and match[1] == project:
            rows[match[2]] = {
                "used": float(match[3]) * SIZE_UNITS[match[4]],
                "quota": float(match[5]) * SIZE_UNITS[match[6]],
                "files": int(match[7]),
                "file_quota": int(match[8]),
            }
    return rows


def headroom_problem(rows, fs, add_bytes, add_files):
    row = rows.get(fs)
    if row is None:
        return f"lquota shows no {fs} quota for this project"
    if row["used"] + add_bytes > row["quota"]:
        return (
            f"{fs}: {add_bytes / 2**30:.2f} GiB more would pass the size quota "
            f"({row['used'] / 2**30:.2f} of {row['quota'] / 2**30:.2f} GiB used)"
        )
    if row["files"] + add_files > row["file_quota"]:
        return (
            f"{fs}: {add_files} more files would pass the file-count quota "
            f"({row['files']} of {row['file_quota']} used)"
        )
    return None


def parse_available_su(text):
    match = re.search(r"^\s*Avail:\s+([\d.]+)\s+(SU|KSU|MSU)\s*$", text, re.MULTILINE)
    if not match:
        raise BlockedError("could not read the SU balance from nci_account:\n" + text)
    return float(match[1]) * SU_UNITS[match[2]]


def parse_qstat(text, job_id):
    """(state letter, exit status or None) from `qstat -xf -F json`."""
    job = json.loads(text).get("Jobs", {}).get(job_id)
    if job is None:
        raise BlockedError(f"qstat has no record of {job_id}")
    exit_status = job.get("Exit_status")
    return job["job_state"], None if exit_status is None else int(exit_status)


def parse_exit_status(text):
    """The exit status from the Resource Usage block Gadi adds to the end of a
    job's log when the job ends, or None while that block is not there."""
    match = re.search(r"^\s*Exit Status:\s+(-?\d+)\s*$", text, re.MULTILINE)
    return None if match is None else int(match[1])


def parse_probe(text):
    head, _, rest = text.partition("--- lquota\n")
    quota, _, rest = rest.partition("--- nci_account\n")
    account, _, queues = rest.partition("--- queues\n")
    facts, datasets = {}, {}
    for line in head.splitlines():
        if line.startswith("dataset "):
            _, folder, digest, *files = line.split()
            datasets[folder] = None if digest == "missing" else (digest, set(files))
        elif "=" in line:
            key, value = line.split("=", 1)
            facts[key] = value
    return facts, datasets, quota, account, queues


def dataset_problems(inputs, found):
    """What is wrong with the published table folders this run reads, if anything."""
    problems = []
    for folder, entry in inputs.datasets.items():
        state = found.get(entry["path"])
        if state is None:
            problems.append(
                f"{entry['path']} ({folder}) is not published on Gadi, or is "
                "incomplete; run scripts/nci/publish_tables.pbs"
            )
        elif state[0] != entry["sha256sums"]:
            problems.append(
                f"{entry['path']} holds different tables from those {DATASETS} names"
            )
    for key, path in inputs.tables.items():
        folder, name = path.rsplit("/", 1)
        state = found.get(folder)
        if state and name not in state[1]:
            problems.append(f"{key}: {path} is not in the published folder")
    return problems


def code_footprint(revision, root=ROOT):
    """Bytes and files (plus folders) a checkout of the revision adds on Gadi."""
    size, names = 0, git("ls-tree", "-r", "-l", revision, root=root).splitlines()
    folders = set()
    for line in names:
        meta, name = line.split("\t", 1)
        size += int(meta.split()[3]) if meta.split()[3] != "-" else 0
        folders.update(str(parent) for parent in Path(name).parents)
    return size, len(names) + len(folders)


def new_run_dir(base, label, now):
    """experiments/nci/<date>_<time>[_label]; seconds are added on a clash."""
    if label and not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise BlockedError("label may use letters, digits, '.', '_' and '-' only")
    suffix = f"_{label}" if label else ""
    path = base / f"{now:%Y-%m-%d_%H%M}{suffix}"
    if path.exists():
        path = base / f"{now:%Y-%m-%d_%H%M%S}{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def extract(stream, destination):
    """Unpack a tar stream, refusing links or paths that leave the destination."""
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        archive.extractall(destination, filter="data")


def write_tar(stream, files):
    """files: (local path or bytes, name in the archive)."""
    with tarfile.open(fileobj=stream, mode="w|") as archive:
        for source, name in files:
            if isinstance(source, bytes):
                info = tarfile.TarInfo(name)
                info.size = len(source)
                archive.addfile(info, io.BytesIO(source))
            else:
                archive.add(source, arcname=name)


class Ssh:
    """Commands on Gadi through the local ssh client. It never asks for passwords."""

    def __init__(self, cfg):
        self.base = [str(part) for part in cfg["ssh"]]

    def _argv(self, host, command):
        # bash explicitly: the remote login shell may not be bash
        return [*self.base, host, "bash -c " + shlex.quote(command)]

    def _check(self, host, command, code, errors):
        if code == 255:  # ssh's own failure: the command may never have run
            raise ConnectionLostError(f"ssh to {host} failed: {errors.strip()}")
        if code:
            raise BlockedError(f"on {host}: {command}\n{errors.strip()}")

    def run(self, host, command, data=None):
        result = subprocess.run(
            self._argv(host, command), input=data, capture_output=True, check=False
        )
        self._check(
            host, command, result.returncode, result.stderr.decode(errors="replace")
        )
        return result.stdout.decode(errors="replace")

    def send_file(self, host, command, path):
        with Path(path).open("rb") as source:
            result = subprocess.run(
                self._argv(host, command),
                stdin=source,
                capture_output=True,
                check=False,
            )
        self._check(
            host, command, result.returncode, result.stderr.decode(errors="replace")
        )

    def receive_tar(self, host, command, destination):
        with tempfile.TemporaryFile() as errors:
            with subprocess.Popen(
                self._argv(host, command), stdout=subprocess.PIPE, stderr=errors
            ) as process:
                extract(process.stdout, destination)
            errors.seek(0)
            self._check(
                host,
                command,
                process.returncode,
                errors.read().decode(errors="replace"),
            )


@dataclass
class Job:
    name: str  # prepare, graph, fold-0 ... fold-4, score
    script: str
    job: dict  # queue and resources; empty for prepare, which uses its own
    variables: dict
    after: list  # names of jobs that must succeed first
    su: float


@dataclass
class Plan:
    model_config: str
    settings: dict
    revision: str
    user: str
    facts: dict
    inputs: Inputs
    env_dir: str
    graph_dir: str
    queue: str
    queue_load: dict
    walltime: str
    capped: bool
    measured: bool
    jobs: list
    su_estimate: float
    su_available: float
    warnings: list


def plan_warnings(rows, estimate, available, queue, load):
    """What could go wrong even though Gadi can take the run."""
    warnings = []
    if available and estimate > SU_SHARE_WARNING * available:
        warnings.append(
            f"this run can use {estimate:.0f} of the {available:.0f} SU the project "
            f"has left ({estimate / available:.0%}); everyone shares that balance"
        )
    for fs, row in sorted(rows.items()):
        if row["files"] > QUOTA_WARNING * row["file_quota"]:
            warnings.append(
                f"{fs} holds {row['files']:,} of the {row['file_quota']:,} files it "
                "may; any job that writes fails once it is full"
            )
        if row["used"] > QUOTA_WARNING * row["quota"]:
            warnings.append(f"{fs} is {row['used'] / row['quota']:.0%} full by size")
    queued, running = load.get(queue, (0, 0))
    if queued > QUEUE_WARNING * max(running, 1):
        warnings.append(
            f"{queue} has {queued} jobs waiting for {running} running, so the folds "
            "may wait a long time; --queue chooses another queue"
        )
    return warnings


def fold_dir(output, fold):
    return f"{output}/fold-{fold}"


def plan_jobs(cfg, settings, inputs, paths, env_dir, graph_dir, facts, queue, walltime):
    """Every job the run needs, in submission order. Paths are this run's on Gadi."""
    jobs, before = [], []
    if facts["env"] != "ready":
        # prepare_environment.pbs asks for 1 copyq CPU for 30 minutes at 2 SU/hour.
        jobs.append(
            Job("prepare", "scripts/nci/prepare_environment.pbs", {}, {}, [], 1)
        )
        before = ["prepare"]
    if facts["graph"] != "ready":
        data = next(
            entry["path"]
            for folder, entry in inputs.datasets.items()
            if folder == Path(inputs.graph_settings["data_dir"]).as_posix()
        )
        job = cfg["graph_job"]
        jobs.append(
            Job(
                "graph",
                "scripts/nci/build_graph.pbs",
                job,
                {
                    "NCI_GRAPH_CONFIG": inputs.graph_config,
                    "NCI_DATA_DIR": data,
                    "NCI_SPLIT_FILE": settings["split_file"],
                    "NCI_SEEDS": settings["seeds_file"],
                    "NCI_GRAPH_DIR": graph_dir,
                },
                list(before),
                su_estimate(job, job["su_per_cpu_hour"]),
            )
        )
        before = [*before, "graph"]
    profile = dict(cfg["queues"][queue], queue=queue, walltime=walltime)
    for fold in FOLDS:
        variables = {
            "NCI_MODEL_CONFIG": paths["config"],
            "NCI_OUTPUT_DIR": fold_dir(paths["output"], fold),
            "NCI_FOLDS": str(fold),
        }
        if profile["device"] == "cpu":
            variables["NCI_DEVICE"] = "cpu"
        jobs.append(
            Job(
                f"fold-{fold}",
                "scripts/nci/train.pbs",
                profile,
                variables,
                list(before),
                su_estimate(profile, profile["su_per_cpu_hour"]),
            )
        )
    job = cfg["score_job"]
    jobs.append(
        Job(
            "score",
            "scripts/nci/score.pbs",
            job,
            {
                "NCI_MODEL_CONFIG": paths["config"],
                "NCI_FOLD_RUNS": ":".join(fold_dir(paths["output"], f) for f in FOLDS),
                "NCI_OUTPUT_DIR": paths["output"],
            },
            [f"fold-{fold}" for fold in FOLDS],
            su_estimate(job, job["su_per_cpu_hour"]),
        )
    )
    return jobs


def make_plan(model_config, cfg, ssh, root=ROOT, queue="auto"):
    """Run every check before actually making changes. If everything passes,
    return a Plan which `describe` prints and `submit` carries out; otherwise
    raise BlockedError saying what to fix."""
    settings = read_settings(root / model_config)
    if settings.get("method") != "hgnn":
        raise BlockedError(
            "only hgnn configs are sent to Gadi; IDW and network mean are scored "
            "with every HGNN run"
        )
    validate_settings(dict(settings, device="cpu"))  # CUDA is checked on the GPU node
    revision = committed_revision(root)
    inputs = plan_inputs(settings, revision, root)
    user = ssh.run(cfg["login_host"], "id -un").strip()
    paths = remote_paths(cfg, user, "probe")
    # Named as scripts/nci/prepare_environment.pbs names it: sha256sum < uv.lock.
    lock = hashlib.sha256(committed_bytes("uv.lock", revision, root)).hexdigest()
    env_dir = f"{paths['shared']}/envs/{lock[:16]}"
    graph_dir = f"{paths['graph_cache']}/{inputs.graph_fingerprint[:16]}"
    folders = sorted({entry["path"] for entry in inputs.datasets.values()})
    facts, found, quota, account, queues = parse_probe(
        ssh.run(
            cfg["login_host"],
            "bash -s -- "
            + quote(
                paths["repo"], revision, env_dir, graph_dir, cfg["project"], *folders
            ),
            data=PROBE.encode(),
        )
    )
    load = parse_queue_load(queues)
    chosen = choose_queue(queue, load, cfg)
    layers = layer_key(settings)
    hours, capped = fold_walltime(cfg, chosen, layers, inputs.hours, settings["epochs"])
    walltime = format_walltime(hours)
    jobs = plan_jobs(
        cfg, settings, inputs, paths, env_dir, graph_dir, facts, chosen, walltime
    )
    rows = parse_lquota(quota, cfg["project"])
    code_bytes, code_files = code_footprint(revision, root)
    problems = [
        *dataset_problems(inputs, found),
        headroom_problem(
            rows,
            "scratch",
            code_bytes + (GRAPH_BYTES if facts["graph"] != "ready" else 0),
            code_files + 100,
        ),
    ]
    if facts["env"] != "ready":
        problems.append(headroom_problem(rows, "gdata", 6 * 2**30, ENVIRONMENT_FILES))
    estimate = sum(job.su for job in jobs)
    available = parse_available_su(account)
    if estimate > available:
        problems.append(f"the run can cost {estimate:.0f} SU; {available:.0f} SU left")
    problems = [problem for problem in problems if problem]
    if problems:
        raise BlockedError("Gadi cannot take this run:\n  " + "\n  ".join(problems))
    warnings = plan_warnings(rows, estimate, available, chosen, load)
    return Plan(
        str(model_config),
        settings,
        revision,
        user,
        facts,
        inputs,
        env_dir,
        graph_dir,
        chosen,
        load,
        walltime,
        capped,
        epoch_seconds(cfg, chosen, layers)[1],
        jobs,
        estimate,
        available,
        warnings,
    )


def describe(plan, cfg):
    profile = cfg["queues"][plan.queue]
    load = ", ".join(
        f"{queue} {queued} queued / {running} running"
        for queue, (queued, running) in sorted(plan.queue_load.items())
    )
    lines = [
        f"Revision: {plan.revision}",
        f"Gadi user: {plan.user}; project {cfg['project']}",
        "Tables (published on Gadi):",
        *(f"  {key}: {path}" for key, path in plan.inputs.tables.items()),
        f"Graphs: {plan.inputs.graph_config} -> {plan.graph_dir} "
        + (
            "(built by an earlier run)"
            if plan.facts["graph"] == "ready"
            else "(built first, by a CPU job)"
        ),
        f"Environment: {plan.env_dir} "
        + ("(ready)" if plan.facts["env"] == "ready" else "(built first, by copyq)"),
        f"Queues now: {load or 'unknown'}",
        f"Folds: 5 jobs on {plan.queue}, each {profile['ncpus']} CPUs, "
        f"{profile['ngpus']} GPU, {profile['mem']} memory, walltime {plan.walltime} "
        f"for {plan.settings['epochs']} epochs over {plan.inputs.hours} hours",
    ]
    if not plan.measured:
        lines.append(
            f"  No measured epoch time for {layer_key(plan.settings)} on {plan.queue}; "
            "the walltime is 1.5 times its slowest measured layer set."
        )
    if plan.capped:
        lines.append(
            f"  WARNING: every epoch would need more than {plan.walltime}, the "
            "queue's limit; a fold stopped by it cannot be scored."
        )
    lines += [
        "Then one CPU job joins the folds and scores every method; PBS emails you "
        "when it ends, or if any job is aborted.",
        "Jobs: " + ", ".join(job.name for job in plan.jobs),
        *(f"WARNING: {warning}" for warning in plan.warnings),
        f"Most it can cost: {plan.su_estimate:.0f} SU "
        f"({plan.su_available:.0f} SU left)",
    ]
    return "\n".join(lines)


def remote_settings(settings, inputs, graph_dir, device):
    """The model config with its tables and graphs pointed at their places on Gadi.
    Committed files (split, seeds) keep their paths: jobs run in the checkout."""
    result = dict(settings, **inputs.tables)
    result["graphs_dir"] = graph_dir
    result["device"] = device
    result.pop("output_dir", None)
    return result


def write_state(run_dir, state):
    (run_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def read_state(run_dir):
    path = Path(run_dir) / "state.json"
    if not path.is_file():
        raise BlockedError(f"{path} not found; is {run_dir} a run folder from submit?")
    return json.loads(path.read_text())


def ship_code(plan, cfg, ssh, paths, root=ROOT):
    """Bring the Gadi repository up to the revision, sending only what it lacks."""
    login, transfer = cfg["login_host"], cfg["transfer_host"]
    if plan.facts["has_revision"] == "yes":
        return
    if git("rev-parse", "HEAD", root=root) != plan.revision:
        raise BlockedError("HEAD moved since the plan was made; run again")
    head = plan.facts.get("repo_head", "")
    known = (
        head
        and subprocess.run(
            ["git", "cat-file", "-e", f"{head}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    with tempfile.TemporaryDirectory() as folder:
        bundle = Path(folder) / "code.bundle"
        git(
            "bundle",
            "create",
            str(bundle),
            "HEAD",
            *([f"^{head}"] if plan.facts["repo"] == "present" and known else []),
            root=root,
        )
        remote = f"{paths['bundles']}/{plan.revision}.bundle"
        ssh.send_file(
            transfer,
            f"mkdir -p {quote(paths['bundles'])} && "
            f"cat > {quote(remote + '.partial')} && "
            f"mv {quote(remote + '.partial', remote)}",
            bundle,
        )
    if plan.facts["repo"] == "absent":
        ssh.run(
            login, f"git clone --quiet --no-checkout {quote(remote, paths['repo'])}"
        )
    else:
        ssh.run(
            login, f"git -C {quote(paths['repo'])} fetch --quiet {quote(remote)} HEAD"
        )
    ssh.run(
        login,
        f"git -C {quote(paths['repo'])} cat-file -e "
        f"{quote(plan.revision + '^{commit}')} && rm {quote(remote)}",
    )


def submit(plan, cfg, ssh, run_dir, root=ROOT):
    """Upload the code and run config, then queue every job with its dependencies."""
    login = cfg["login_host"]
    run_id = run_dir.name
    paths = remote_paths(cfg, plan.user, run_id)
    device = cfg["queues"][plan.queue]["device"]
    config_text = yaml.safe_dump(
        remote_settings(plan.settings, plan.inputs, plan.graph_dir, device),
        sort_keys=False,
    )
    (run_dir / "run.yaml").write_text(config_text)
    logs = {job.name: f"{paths['logs']}/{job.name}.log" for job in plan.jobs}
    state = {
        "run": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "UPLOADING",
        "model_config": plan.model_config,
        "revision": plan.revision,
        "login_host": login,
        "transfer_host": cfg["transfer_host"],
        "remote": paths,
        "queue": plan.queue,
        "walltime": plan.walltime,
        "graph_dir": plan.graph_dir,
        "env_dir": plan.env_dir,
        "su_estimate": plan.su_estimate,
        "storage": storage_directive(
            paths["run"], paths["shared"] + "/", paths["graph_cache"] + "/"
        ),
        "layers": layer_key(plan.settings),
        "hours": plan.inputs.hours,
        "epochs": plan.settings["epochs"],
        "logs": logs,
        "after": {job.name: job.after for job in plan.jobs},
        "jobs": {},
    }
    write_state(run_dir, state)
    try:
        ssh.run(
            cfg["transfer_host"],
            f"mkdir -p {quote(paths['runs'])} && mkdir {quote(paths['run'])} && "
            f"mkdir {quote(paths['logs'])} && cat > {quote(paths['config'])}",
            config_text.encode(),
        )
        ship_code(plan, cfg, ssh, paths, root)
        code = quote(paths["code"])
        # The checkout links straight to this lock's environment, so a later run that
        # relinks the shared checkout never changes what this run's jobs import.
        ssh.run(
            login,
            f"git -C {quote(paths['repo'])} worktree add --quiet --detach "
            f"{code} {quote(plan.revision)} && "
            f"ln -s {quote(plan.env_dir, paths['code'] + '/.venv')}",
        )
        storage = state["storage"]
        name = short_name(run_id)
        # The plan's jobs name a placeholder run; these name this run's folders.
        jobs = plan_jobs(
            cfg,
            plan.settings,
            plan.inputs,
            paths,
            plan.env_dir,
            plan.graph_dir,
            plan.facts,
            plan.queue,
            plan.walltime,
        )
        for job in jobs:
            after = [state["jobs"][previous] for previous in job.after]
            if job.name == "prepare":
                command = (
                    f"git -C {quote(paths['repo'])} checkout --quiet --detach "
                    f"{quote(plan.revision)} && cd {quote(paths['repo'])} && "
                    f"qsub -o {quote(logs['prepare'])} -m a "
                    "scripts/nci/prepare_environment.pbs"
                )
            else:
                command = f"cd {code} && " + quote(
                    *qsub_command(
                        cfg["project"],
                        job.job,
                        job.script,
                        job.variables,
                        storage,
                        logs[job.name],
                        after,
                        name=f"{name}-{job.name}",
                        mail="ae" if job.name == "score" else "a",
                    )
                )
            state["jobs"][job.name] = ssh.run(login, command).strip()
            write_state(run_dir, state)  # on record before the next step can fail
    except BaseException as error:
        state.update(status="SUBMIT_FAILED", error=str(error) or type(error).__name__)
        write_state(run_dir, state)
        if state["jobs"] and isinstance(error, BlockedError):
            ids = " ".join(state["jobs"].values())
            raise BlockedError(
                f"{error}\nThese jobs were already submitted and stay on Gadi: "
                f"{ids}. Cancel them with qdel {ids} if you do not want them."
            ) from error
        raise
    state["status"] = "SUBMITTED"
    write_state(run_dir, state)
    return state


def short_name(run_id):
    """The run's label, or "run", for PBS job names: <label>-fold-0 and so on."""
    return (run_id.split("_", 2)[2] if run_id.count("_") >= 2 else "run")[:8]


def resume(run_dir, cfg, ssh, queue=None):
    """Resubmit the folds of a finished run that did not succeed, then a new scoring
    job. A fold that saved last.pt carries on where it stopped; one that never
    started runs from the beginning. Returns the updated state."""
    run_dir = Path(run_dir)
    state = read_state(run_dir)
    if state.get("status") not in ("SUBMITTED", "RESUMED"):
        raise BlockedError(f"{run_dir} was never fully submitted; submit it again")
    if (run_dir / "results").exists():
        raise BlockedError(f"{run_dir} was already fetched; submit a new run instead")
    exits, last = {}, {}
    poll(ssh, state, exits, last, lambda text: None)
    waiting = [name for name in state["jobs"] if name not in exits]
    if waiting:
        raise BlockedError(f"wait for {', '.join(waiting)} to finish first")
    for name in ("prepare", "graph"):
        if name in exits and exits[name] != 0:
            raise BlockedError(
                f"the {name} job failed (log: {state['logs'][name]}); fix what it "
                "reports and submit a new run"
            )
    folds = [f"fold-{fold}" for fold in FOLDS if exits.get(f"fold-{fold}") != 0]
    if not folds and exits.get("score") == 0:
        raise BlockedError("every job succeeded; fetch the run instead")
    queue = queue or state["queue"]
    if queue not in cfg["queues"]:
        raise BlockedError(f"queue {queue} is not in remote.yaml")
    hours, _ = fold_walltime(
        cfg, queue, state["layers"], state["hours"], state["epochs"]
    )
    profile = dict(cfg["queues"][queue], queue=queue, walltime=format_walltime(hours))
    output, code = state["remote"]["output"], state["remote"]["code"]
    saved = ssh.run(
        state["login_host"],
        " ; ".join(
            f"[[ -f {quote(fold_dir(output, name[5:]) + '/last.pt')} ]] && echo {name}"
            for name in folds
        )
        + " ; true",
    ).split()
    attempt = len(state.get("previous", [])) + 2
    state.setdefault("previous", []).append(dict(state["jobs"]))
    jobs = {name: job for name, job in state["jobs"].items() if name not in folds}
    jobs.pop("score", None)
    name = short_name(state["run"])
    for fold in folds:
        variables = {
            "NCI_MODEL_CONFIG": state["remote"]["config"],
            "NCI_OUTPUT_DIR": fold_dir(output, fold[5:]),
            "NCI_FOLDS": fold[5:],
        }
        if profile["device"] == "cpu":
            variables["NCI_DEVICE"] = "cpu"
        if fold in saved:
            variables["NCI_RESUME"] = "1"
        state["logs"][fold] = f"{state['remote']['logs']}/{fold}.{attempt}.log"
        state["after"][fold] = []
        command = qsub_command(
            cfg["project"],
            profile,
            "scripts/nci/train.pbs",
            variables,
            state["storage"],
            state["logs"][fold],
            name=f"{name}-{fold}",
            mail="a",
        )
        jobs[fold] = ssh.run(
            state["login_host"], f"cd {quote(code)} && {quote(*command)}"
        ).strip()
        state["jobs"] = {**jobs}
        write_state(run_dir, state)
    job = cfg["score_job"]
    state["logs"]["score"] = f"{state['remote']['logs']}/score.{attempt}.log"
    command = qsub_command(
        cfg["project"],
        job,
        "scripts/nci/score.pbs",
        {
            "NCI_MODEL_CONFIG": state["remote"]["config"],
            "NCI_FOLD_RUNS": ":".join(fold_dir(output, f) for f in FOLDS),
            "NCI_OUTPUT_DIR": output,
        },
        state["storage"],
        state["logs"]["score"],
        [jobs[fold] for fold in folds],
        name=f"{name}-score",
        mail="ae",
    )
    state["after"]["score"] = folds
    jobs["score"] = ssh.run(
        state["login_host"], f"cd {quote(code)} && {quote(*command)}"
    ).strip()
    state.update(jobs=jobs, status="RESUMED", queue=queue)
    write_state(run_dir, state)
    return state


def job_state(ssh, state, name):
    """(state letter, exit status or None) for one of the run's jobs.

    The job's log is read first: Gadi ends it with the exit status, and it is
    still there after qstat has forgotten the job. qstat answers until then.
    """
    job_id = state["jobs"][name]
    log = state["logs"][name]
    tail = ssh.run(state["login_host"], f"tail -n 30 {quote(log)} 2>/dev/null || true")
    exit_status = parse_exit_status(tail)
    if exit_status is not None:
        return "F", exit_status
    try:
        text = ssh.run(state["login_host"], f"qstat -xf -F json {quote(job_id)}")
        return parse_qstat(text, job_id)
    except ConnectionLostError:
        raise
    except BlockedError as error:
        raise BlockedError(
            f"{error}\n{log} has no exit status either, so {job_id} may never have "
            "started (for example, it was removed while queued)"
        ) from error


def poll(ssh, state, finished, last, say):
    """One look at every unfinished job; updates finished and last in place."""
    for name in state["jobs"]:
        if name in finished:
            continue
        try:
            letter, exit_status = job_state(ssh, state, name)
        except ConnectionLostError:
            raise
        except BlockedError:
            # PBS deletes a job whose dependency failed, so it never runs or logs.
            if any(finished.get(dep, 0) != 0 for dep in state["after"].get(name, [])):
                finished[name] = None
                continue
            raise
        if last.get(name) != letter:
            say(f"{name} {state['jobs'][name]}: {JOB_STATES.get(letter, letter)}")
            last[name] = letter
        if letter == "F":
            finished[name] = exit_status


def wait(ssh, state, poll_seconds, sleep=time.sleep, say=print, retries=3):
    """Poll until every job has finished; returns {job name: exit status}.

    A dropped ssh connection is tried again at the next poll, up to `retries`
    times in a row, so one network drop does not end a long wait.
    """
    finished, last, failures = {}, {}, 0
    while len(finished) < len(state["jobs"]):
        try:
            poll(ssh, state, finished, last, say)
        except ConnectionLostError:
            failures += 1
            if failures > retries:
                raise
            say(
                f"lost the connection to Gadi ({failures} of {retries}); "
                f"trying again in {poll_seconds} s"
            )
        else:
            failures = 0
        if len(finished) < len(state["jobs"]):
            sleep(poll_seconds)
    return finished


def verdict(results, exits):
    for name, code in exits.items():
        if code != 0:
            return f"FAILED: {name} job exit status {code}"
    report = results / "output" / "scored" / "report.md"
    if not report.is_file():
        return "FAILED: output/scored/report.md missing"
    if re.search(
        r"^Status: (FAILED|RUNNING|PARTIAL)", report.read_text(), re.MULTILINE
    ):
        return "FAILED: report.md does not show a completed run"
    return "DONE"


def download(ssh, state, run_dir, with_predictions):
    results, partial = run_dir / "results", run_dir / "results.partial"
    # A download that stops partway is left in results.partial and redone next time,
    # so results/ only ever holds a whole download.
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    exclude = "" if with_predictions else "--exclude=predictions.csv "
    ssh.receive_tar(
        state["transfer_host"],
        f"cd {quote(state['remote']['run'])} && tar -cf - {exclude}"
        "$(ls -d output logs run.yaml 2>/dev/null)",
        partial,
    )
    partial.rename(results)
    return results


def fetch(run_dir, ssh, poll_seconds, block=False, with_predictions=False, **hooks):
    """Download a finished run into <run>/results/; hooks: sleep and say.

    Without block, a run with unfinished jobs is left alone and reported."""
    run_dir = Path(run_dir)
    state = read_state(run_dir)
    if (run_dir / "results").exists():
        raise BlockedError(
            f"{run_dir / 'results'} already exists; results are never overwritten"
        )
    say = hooks.get("say", print)
    if block:
        exits = wait(ssh, state, poll_seconds, hooks.get("sleep", time.sleep), say)
    else:
        exits, last = {}, {}
        poll(ssh, state, exits, last, lambda text: None)
        waiting = [name for name in state["jobs"] if name not in exits]
        if waiting:
            return dict(state, status=f"WAITING: {', '.join(waiting)} not finished")
    results = download(ssh, state, run_dir, with_predictions)
    state.update(
        exits=exits,
        status=verdict(results, exits),
        fetched_utc=datetime.now(UTC).isoformat(),
    )
    write_state(run_dir, state)
    return state


def unfetched_runs(root):
    base = root / RUNS_DIR
    if not base.is_dir():
        return []
    return sorted(
        path
        for path in base.iterdir()
        if (path / "state.json").is_file()
        and not (path / "results").exists()
        and read_state(path).get("status") in ("SUBMITTED", "RESUMED")
    )


DOCTOR = r"""
project=$1 shared=$2
echo "user=$(id -un)"
id -nG | tr ' ' '\n' | grep -qx "$project" && echo member=yes || echo member=no
[[ -r $shared/python/py314/bin/python3.14 ]] && echo python=yes || echo python=no
echo '--- lquota'
lquota
echo '--- nci_account'
nci_account -P "$project"
"""


def doctor(cfg, ssh, say=print):
    """Check a teammate's setup from their machine; returns the problems found."""
    problems = []
    try:
        facts, _, quota, account, _ = parse_probe(
            ssh.run(
                cfg["login_host"],
                "bash -s -- "
                + quote(
                    cfg["project"], cfg["shared_root"].format(project=cfg["project"])
                ),
                data=DOCTOR.encode(),
            )
        )
    except ConnectionLostError as error:
        return [f"{error}\nSet up an SSH key for Gadi; see docs/nci/ONBOARDING.md"]
    say(f"Gadi user: {facts.get('user')}")
    try:
        ssh.run(cfg["transfer_host"], "true")
    except ConnectionLostError as error:
        problems.append(
            f"{error}\n{cfg['transfer_host']} carries uploads and downloads; log in "
            f"to it once with `ssh {cfg['transfer_host']} hostname` to check and "
            "accept its host key"
        )
    if facts.get("member") != "yes":
        problems.append(
            f"you are not in project {cfg['project']}; ask its lead on MyNCI"
        )
    if facts.get("python") != "yes":
        problems.append(
            "the shared Python is missing or unreadable; tell the project lead"
        )
    try:
        say(f"SU left in {cfg['project']}: {parse_available_su(account):.0f}")
    except BlockedError as error:
        problems.append(str(error))
    rows = parse_lquota(quota, cfg["project"])
    for fs, row in sorted(rows.items()):
        say(f"{fs}: {row['files']} of {row['file_quota']} files used")
        if row["files"] > 0.95 * row["file_quota"]:
            problems.append(f"{fs} is nearly out of files; runs may fail to write")
    try:
        git("rev-parse", "HEAD", root=ROOT)
    except BlockedError as error:
        problems.append(str(error))
    return problems


def main(argv=None, ssh_factory=Ssh, confirm=input, root=None):
    root = root or ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remote-config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "submit"):
        sub = commands.add_parser(name)
        sub.add_argument("model_config", type=Path)
        sub.add_argument(
            "--queue",
            default="auto",
            help="gpuvolta, dgxa100, normal, or auto: the GPU queue with the "
            "shorter wait",
        )
        if name == "submit":
            sub.add_argument("--label")
            sub.add_argument("--yes", action="store_true", help="skip the confirmation")
            sub.add_argument(
                "--wait", action="store_true", help="wait for the jobs, then fetch"
            )
    status = commands.add_parser("status")
    status.add_argument("run_dirs", type=Path, nargs="*")
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("run_dirs", type=Path, nargs="*")
    fetch_parser.add_argument("--wait", action="store_true")
    fetch_parser.add_argument(
        "--with-predictions",
        action="store_true",
        help="also download predictions.csv (hundreds of MB per run)",
    )
    resume_parser = commands.add_parser("resume")
    resume_parser.add_argument("run_dir", type=Path)
    resume_parser.add_argument("--queue", help="run the remaining folds on this queue")
    commands.add_parser("doctor")
    args = parser.parse_args(argv)
    run_dir = None
    try:
        cfg = load_remote_config(root / args.remote_config)
        ssh = ssh_factory(cfg)
        if args.command == "resume":
            state = resume(args.run_dir, cfg, ssh, args.queue)
            print(f"Resubmitted {state['jobs']}")
            return 0
        if args.command == "doctor":
            problems = doctor(cfg, ssh)
            for problem in problems:
                print(f"PROBLEM: {problem}")
            print("Ready to submit." if not problems else "Fix the problems above.")
            return 1 if problems else 0
        if args.command in ("status", "fetch"):
            runs = args.run_dirs or unfetched_runs(root)
            if not runs:
                print("No submitted runs waiting to be fetched.")
            for run_dir in runs:
                if args.command == "status":
                    state = read_state(run_dir)
                    print(f"{run_dir}:")
                    for name, job_id in state["jobs"].items():
                        letter, exit_status = job_state(ssh, state, name)
                        print(
                            f"  {name} {job_id}: {JOB_STATES.get(letter, letter)}"
                            + ("" if exit_status is None else f", exit {exit_status}")
                        )
                else:
                    result = fetch(
                        run_dir,
                        ssh,
                        cfg["poll_seconds"],
                        args.wait,
                        args.with_predictions,
                    )
                    print(f"{run_dir}: {result['status']}")
            return 0
        plan = make_plan(args.model_config, cfg, ssh, root, args.queue)
        print(describe(plan, cfg))
        if args.command == "plan":
            print("Plan only: nothing was uploaded or submitted.")
            return 0
        if not args.yes and confirm("Type yes to submit: ").strip() != "yes":
            print("Cancelled: nothing was uploaded or submitted.")
            return 1
        run_dir = new_run_dir(root / RUNS_DIR, args.label, datetime.now())
        state = submit(plan, cfg, ssh, run_dir, root)
        print(f"Submitted {state['jobs']}; run folder {run_dir}")
        if args.wait:
            print(fetch(run_dir, ssh, cfg["poll_seconds"], block=True)["status"])
        return 0
    except BlockedError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        if run_dir is not None:
            print(f"Run folder: {run_dir}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if run_dir is not None:
            print(
                f"\nStopped waiting; jobs keep running on Gadi. Fetch later with: "
                f"uv run python -m scripts.nci.remote fetch {run_dir}",
                file=sys.stderr,
            )
        return 130


if __name__ == "__main__":
    sys.exit(main())
