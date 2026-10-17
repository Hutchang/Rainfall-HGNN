"""Run Component 06 training on NCI Gadi from a clean local checkout.

    uv run python -m scripts.nci.remote plan   config/models/hgnn.yaml
    uv run python -m scripts.nci.remote run    config/models/hgnn.yaml --label NAME
    uv run python -m scripts.nci.remote submit config/models/hgnn.yaml --label NAME
    uv run python -m scripts.nci.remote status experiments/nci/<run>
    uv run python -m scripts.nci.remote fetch  experiments/nci/<run>

Code travels as a git bundle of the committed revision, so Gadi runs exactly that
commit. Inputs git does not carry (Component 02 tables, the Component 03 split,
Component 04 graphs) are uploaded once into a folder named by their fingerprints and
reused by later runs. Nothing on Gadi is overwritten and results land in a new local
folder. SSH must already log in without a password prompt; see docs/nci/README.md.
"""

import argparse
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.models.run import (
    INPUT_KEYS,
    file_digest,
    input_paths,
    read_settings,
    validate_settings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("config/nci/remote.yaml")
RUNS_DIR = Path("experiments/nci")
CONFIG_KEYS = (
    "project",
    "login_host",
    "transfer_host",
    "ssh",
    "scratch_root",
    "inputs_root",
    "job",
    "su_per_cpu_hour",
    "poll_seconds",
)
JOB_KEYS = ("queue", "ncpus", "ngpus", "mem", "jobfs", "walltime")
SIZE_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}
SU_UNITS = {"SU": 1, "KSU": 1e3, "MSU": 1e6}
JOB_STATES = {
    "Q": "queued",
    "H": "held",
    "W": "waiting",
    "R": "running",
    "E": "exiting",
    "F": "finished",
}

# Read-only facts about the remote side, printed as key=value then two raw reports.
PROBE = r"""
repo=$1 inputs=$2 revision=$3 project=$4
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 \
    && echo repo=present || echo repo=absent
echo "repo_head=$(git -C "$repo" rev-parse HEAD 2>/dev/null)"
git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null \
    && echo has_revision=yes || echo has_revision=no
[[ -f $inputs/COMPLETE ]] && echo inputs=present || echo inputs=absent
[[ -x $repo/.venv/bin/python ]] && echo venv=present || echo venv=absent
echo '--- lquota'
lquota
echo '--- nci_account'
nci_account -P "$project"
"""


class BlockedError(Exception):
    """A check failed; the message says what to fix."""


def quote(*parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def load_remote_config(path):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise BlockedError(f"{path} must contain a YAML mapping")
    missing = [key for key in CONFIG_KEYS if key not in cfg]
    missing += [f"job.{key}" for key in JOB_KEYS if key not in cfg.get("job", {})]
    if missing:
        raise BlockedError(f"{path} is missing {', '.join(missing)}")
    for key in JOB_KEYS:
        if re.search(r"[\s,=+]", str(cfg["job"][key])):
            raise BlockedError(
                f"job.{key} must be a single qsub value, got {cfg['job'][key]!r}"
            )
    walltime_hours(cfg["job"]["walltime"])
    return cfg


def walltime_hours(text):
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", str(text))
    if not match:
        raise BlockedError(f"walltime must be HH:MM:SS, got {text!r}")
    hours, minutes, seconds = map(int, match.groups())
    return hours + minutes / 60 + seconds / 3600


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


def is_tracked(path, root=ROOT):
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def committed_digest(path, revision, root=ROOT):
    """sha256 and size of a file as committed, which is what Gadi checks out.

    The working copy can differ without git calling it edited: on Windows git
    checks text files out with CRLF line endings but commits them with LF.
    """
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise BlockedError(
            f"git cat-file failed for {path}: "
            + result.stderr.decode(errors="replace").strip()
        )
    return hashlib.sha256(result.stdout).hexdigest(), len(result.stdout)


@dataclass(frozen=True)
class Input:
    label: str
    local: str
    remote: str  # under the inputs folder, or under the checkout when tracked
    sha256: str
    size: int
    tracked: bool


def plan_inputs(settings, root=ROOT, revision="HEAD"):
    """Lists every file that the model config reads. Files committed to git ride along with
    the code when we upload the git bundle. 

    Stops if any files are missing
    """
    paths = input_paths(settings)
    missing = [
        f"{label}: {path}"
        for label, path in paths.items()
        if not (root / path).is_file()
    ]
    if missing:
        raise BlockedError("missing input files:\n  " + "\n  ".join(missing))
    inputs = []
    for label, path in paths.items():
        local = root / path
        tracked = is_tracked(path, root)
        if tracked:
            remote = Path(path).as_posix()
        elif label.startswith("fold-"):
            remote = f"graphs/{label}.pt"
        else:
            remote = f"{label}/{local.name}"
        sha256, size = (
            committed_digest(remote, revision, root)
            if tracked
            else (file_digest(local), local.stat().st_size)
        )
        inputs.append(Input(label, str(local), remote, sha256, size, tracked))
    if len({item.tracked for item in inputs if item.label.startswith("fold-")}) > 1:
        raise BlockedError(
            "graphs_dir is partly tracked by git; track all of it or none"
        )
    return inputs


def inputs_fingerprint(inputs):
    lines = sorted(
        f"{item.remote} {item.sha256}" for item in inputs if not item.tracked
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def remote_settings(settings, inputs, inputs_dir, output_dir):
    """Rewrites the model config for Gadi, pointing uploaded inputs at the Gadi
    input folder and output at the run folder.
    """
    result = dict(settings)
    by_label = {item.label: item for item in inputs}
    for key in INPUT_KEYS:
        if not by_label[key].tracked:
            result[key] = f"{inputs_dir}/{by_label[key].remote}"
    if "fold-0/train" in by_label and not by_label["fold-0/train"].tracked:
        result["graphs_dir"] = f"{inputs_dir}/graphs"
    result["device"] = "cuda"
    result["output_dir"] = output_dir
    return result


def remote_paths(cfg, user, run_id, fingerprint):
    """
    Decides where everything lives on Gadi
    
    """
    fill = {"project": cfg["project"], "user": user}
    scratch = cfg["scratch_root"].format(**fill)
    run = f"{scratch}/runs/{run_id}"
    return {
        "repo": f"{scratch}/repo",
        "bundles": f"{scratch}/bundles",
        "runs": f"{scratch}/runs",
        "run": run,
        "code": f"{run}/code",
        "config": f"{run}/run.yaml",
        "output": f"{run}/output",
        "log": f"{run}/pbs.log",
        "prepare_log": f"{run}/prepare.log",
        "inputs": f"{cfg['inputs_root'].format(**fill)}/{fingerprint[:16]}",
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


def qsub_command(cfg, script, variables, storage, log, depend=None):
    job = cfg["job"]
    resources = ",".join(
        f"{key}={job[key]}" for key in ("ncpus", "ngpus", "mem", "jobfs", "walltime")
    )
    for name, value in variables.items():
        if re.search(r"[,\s'\"]", value):
            raise BlockedError(f"{name}={value!r} cannot be passed through qsub -v")
    command = [
        "qsub",
        "-P",
        cfg["project"],
        "-q",
        str(job["queue"]),
        "-l",
        f"{resources},storage={storage}",
        "-o",
        log,
        "-v",
        ",".join(f"{name}={value}" for name, value in variables.items()),
    ]
    if depend:
        command += ["-W", f"depend=afterok:{depend}"]
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
    """(state letter, exit status or None) from `qstat -xf -F json`.
    
    qstat is the command on Gadi showing the state of batch jobs
    """
    job = json.loads(text).get("Jobs", {}).get(job_id)
    if job is None:
        raise BlockedError(f"qstat has no record of {job_id}")
    exit_status = job.get("Exit_status")
    return job["job_state"], None if exit_status is None else int(exit_status)


def parse_probe(text):
    head, _, rest = text.partition("--- lquota\n")
    quota, _, account = rest.partition("--- nci_account\n")
    facts = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    return facts, quota, account


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

    def send_tar(self, host, command, files):
        with tempfile.TemporaryFile() as errors:
            with subprocess.Popen(
                self._argv(host, command), stdin=subprocess.PIPE, stderr=errors
            ) as process:
                write_tar(process.stdin, files)
                process.stdin.close()
            errors.seek(0)
            self._check(
                host,
                command,
                process.returncode,
                errors.read().decode(errors="replace"),
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
class Plan:
    model_config: str
    settings: dict
    revision: str
    inputs: list
    fingerprint: str
    user: str
    facts: dict
    su_estimate: float
    su_available: float
    upload_bytes: int


def make_plan(model_config, cfg, ssh, root=ROOT):
    """Run every check before actually making changes. If everything passes,
    return a Plan object which"""
    settings = read_settings(root / model_config)
    validate_settings(dict(settings, device="cpu"))  # CUDA is checked on the GPU node
    revision = committed_revision(root)
    inputs = plan_inputs(settings, root, revision)
    fingerprint = inputs_fingerprint(inputs)
    user = ssh.run(cfg["login_host"], "id -un").strip()
    paths = remote_paths(cfg, user, "probe", fingerprint)
    facts, quota, account = parse_probe(
        ssh.run(
            cfg["login_host"],
            "bash -s -- "
            + quote(paths["repo"], paths["inputs"], revision, cfg["project"]),
            data=PROBE.encode(),
        )
    )
    rows = parse_lquota(quota, cfg["project"])
    uploads = [item for item in inputs if not item.tracked]
    upload_bytes = (
        sum(item.size for item in uploads) if facts["inputs"] == "absent" else 0
    )
    upload_files = len(uploads) + 8 if upload_bytes else 0  # files plus their folders
    code_bytes, code_files = code_footprint(revision, root)
    problems = [
        headroom_problem(
            rows,
            storage_directive(paths["inputs"]).split("/")[0],
            upload_bytes,
            upload_files,
        ),
        headroom_problem(rows, "scratch", code_bytes, code_files),
    ]
    estimate = su_estimate(cfg["job"], cfg["su_per_cpu_hour"])
    available = parse_available_su(account)
    if estimate > available:
        problems.append(f"the job can cost {estimate:.0f} SU; {available:.0f} SU left")
    problems = [problem for problem in problems if problem]
    if problems:
        raise BlockedError("Gadi cannot take this run:\n  " + "\n  ".join(problems))
    return Plan(
        str(model_config),
        settings,
        revision,
        inputs,
        fingerprint,
        user,
        facts,
        estimate,
        available,
        upload_bytes,
    )


def describe(plan, cfg):
    job = cfg["job"]
    uploads = [item for item in plan.inputs if not item.tracked]
    tracked = [item.label for item in plan.inputs if item.tracked]
    lines = [
        f"Revision: {plan.revision} (as a git bundle)",
        f"Gadi user: {plan.user}; project {cfg['project']}",
        f"Inputs fingerprint: {plan.fingerprint[:16]}",
        "Inputs to upload:" if plan.upload_bytes else "Inputs: already on Gadi",
    ]
    if plan.upload_bytes:
        lines += [
            f"  {item.label}: {item.local} ({item.size / 2**20:.1f} MiB)"
            for item in uploads
        ]
        lines.append(f"  total {plan.upload_bytes / 2**20:.1f} MiB")
    lines += [
        f"With the code: {', '.join(tracked) or 'none'}",
        f"Job: {job['queue']}, {job['ncpus']} CPUs, {job['ngpus']} GPU, {job['mem']} "
        f"memory, walltime {job['walltime']}",
        f"Most it can cost: {plan.su_estimate:.1f} SU "
        f"({plan.su_available:.0f} SU left)",
    ]
    if plan.facts["venv"] == "absent":
        lines.append(
            "Environment missing: a copyq preparation job runs first "
            "(1 CPU, up to 30 minutes, extra SU)."
        )
    return "\n".join(lines)


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


def ship_inputs(plan, cfg, ssh, paths, run_id):
    """Upload to a partial folder, verify checksums on Gadi, then publish read-only."""
    uploads = [item for item in plan.inputs if not item.tracked]
    if plan.facts["inputs"] == "present" or not uploads:
        return
    sums = "".join(f"{item.sha256}  {item.remote}\n" for item in uploads).encode()
    partial = f"{paths['inputs']}.partial-{run_id}"
    ssh.send_tar(
        cfg["transfer_host"],
        f"mkdir -p {quote(partial)} && tar -xf - -C {quote(partial)} && "
        f"cd {quote(partial)} && sha256sum --check --quiet SHA256SUMS && "
        f"touch COMPLETE && chmod -R a-w . && mv -T {quote(partial, paths['inputs'])}",
        [(sums, "SHA256SUMS"), *((item.local, item.remote) for item in uploads)],
    )


def submit(plan, cfg, ssh, run_dir, login_check=True, root=ROOT):
    """Upload, check out the revision, check inputs on the login node, then qsub."""
    login = cfg["login_host"]
    run_id = run_dir.name
    paths = remote_paths(cfg, plan.user, run_id, plan.fingerprint)
    settings = remote_settings(
        plan.settings, plan.inputs, paths["inputs"], paths["output"]
    )
    config_text = yaml.safe_dump(settings, sort_keys=False)
    (run_dir / "run.yaml").write_text(config_text)
    state = {
        "run": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "UPLOADING",
        "model_config": plan.model_config,
        "revision": plan.revision,
        "login_host": login,
        "transfer_host": cfg["transfer_host"],
        "remote": paths,
        "inputs_fingerprint": plan.fingerprint,
        "inputs": [asdict(item) for item in plan.inputs],
        "job": cfg["job"],
        "su_estimate": plan.su_estimate,
        "jobs": {},
    }
    write_state(run_dir, state)
    ssh.run(
        cfg["transfer_host"],
        f"mkdir -p {quote(paths['runs'])} && "
        f"mkdir {quote(paths['run'])} && "
        f"cat > {quote(paths['config'])}",
        config_text.encode(),
    )
    ship_code(plan, cfg, ssh, paths, root)
    ship_inputs(plan, cfg, ssh, paths, run_id)
    code = quote(paths["code"])
    ssh.run(
        login,
        f"git -C {quote(paths['repo'])} worktree add --quiet --detach "
        f"{code} {quote(plan.revision)} && "
        f"ln -s {quote(paths['repo'] + '/.venv', paths['code'] + '/.venv')}",
    )
    ready = ssh.run(
        login,
        f"cd {code} && [[ -x .venv/bin/python ]] && "
        ".venv/bin/python -B scripts/nci/check_environment.py "
        ">/dev/null 2>&1 && echo ready || echo prepare",
    ).strip()
    storage = storage_directive(paths["run"], paths["inputs"])
    if ready == "ready" and login_check:
        ssh.run(
            login,
            f"cd {code} && .venv/bin/python -B -m src.models.run "
            f"{quote(paths['config'])} --check-inputs --device cpu",
        )
    if ready != "ready":
        state["jobs"]["prepare"] = ssh.run(
            login,
            f"git -C {quote(paths['repo'])} checkout --quiet --detach "
            f"{quote(plan.revision)} && cd {quote(paths['repo'])} && "
            f"qsub -o {quote(paths['prepare_log'])} "
            "scripts/nci/prepare_environment.pbs",
        ).strip()
    command = qsub_command(
        cfg,
        "scripts/nci/train.pbs",
        {"NCI_MODEL_CONFIG": paths["config"], "NCI_OUTPUT_DIR": paths["output"]},
        storage,
        paths["log"],
        state["jobs"].get("prepare"),
    )
    state["jobs"]["train"] = ssh.run(login, f"cd {code} && {quote(*command)}").strip()
    state["status"] = "SUBMITTED"
    write_state(run_dir, state)
    return state


def job_state(ssh, state, name):
    job_id = state["jobs"][name]
    text = ssh.run(state["login_host"], f"qstat -xf -F json {quote(job_id)}")
    return parse_qstat(text, job_id)


def wait(ssh, state, poll_seconds, sleep=time.sleep, say=print):
    """Poll until every job has finished; returns {job name: exit status}."""
    finished, last = {}, {}
    while len(finished) < len(state["jobs"]):
        for name in state["jobs"]:
            if name in finished:
                continue
            try:
                letter, exit_status = job_state(ssh, state, name)
            except BlockedError:
                if name == "train" and finished.get("prepare", 0) != 0:
                    finished[name] = None  # PBS drops a job whose dependency failed
                    continue
                raise
            if last.get(name) != letter:
                say(f"{name} {state['jobs'][name]}: {JOB_STATES.get(letter, letter)}")
                last[name] = letter
            if letter == "F":
                finished[name] = exit_status
        if len(finished) < len(state["jobs"]):
            sleep(poll_seconds)
    return finished


def verdict(results, exits):
    if exits.get("prepare") not in (None, 0):
        return f"FAILED: environment preparation exited {exits['prepare']}"
    if exits.get("train") != 0:
        return f"FAILED: training job exit status {exits.get('train')}"
    report = results / "output" / "report.md"
    if not (results / "output" / "predictions.csv").is_file() or not report.is_file():
        return "FAILED: predictions.csv or report.md missing"
    if re.search(r"^Status: (FAILED|RUNNING)", report.read_text(), re.MULTILINE):
        return "FAILED: report.md does not show a completed run"
    return "DONE"


def fetch(run_dir, ssh, poll_seconds, sleep=time.sleep, say=print):
    """Wait for the jobs to finish , then download the run folder into <run>/results/.
    
    
    """
    run_dir = Path(run_dir)
    state = read_state(run_dir)
    results = run_dir / "results"
    if results.exists():
        # refuse overwrite if results already exist for that specific run_dir.
        # BUG: what if something stops mid write and we get a partial write
        raise BlockedError(f"{results} already exists; results are never overwritten")
    exits = wait(ssh, state, poll_seconds, sleep, say)
    partial = run_dir / "results.partial"
    shutil.rmtree(partial, ignore_errors=True)  # an earlier download that broke off
    partial.mkdir()
    remote = state["remote"]["run"]
    ssh.receive_tar(
        state["transfer_host"],
        f"cd {quote(remote)} && tar -cf - $(ls -d output pbs.log "
        "prepare.log run.yaml 2>/dev/null)",
        partial,
    )
    partial.rename(results)
    state.update(
        exits=exits,
        status=verdict(results, exits),
        fetched_utc=datetime.now(UTC).isoformat(),
    )
    write_state(run_dir, state)
    return state


def main(argv=None, ssh_factory=Ssh, confirm=input, root=None):
    """
    1. reads the command and config/nci/remote.yaml, then opens the SSH helper
    2. status / fetch: loads saved run state and either shows the job states or calls `fetch`
    
    """


    root = root or ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remote-config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "submit", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("model_config", type=Path)
        if name != "plan":
            sub.add_argument("--label")
            sub.add_argument("--yes", action="store_true", help="skip the confirmation")
            sub.add_argument(
                "--skip-login-check",
                action="store_true",
                help="do not run --check-inputs on the login node",
            )
    for name in ("status", "fetch"):
        commands.add_parser(name).add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = None
    try:
        cfg = load_remote_config(root / args.remote_config)
        ssh = ssh_factory(cfg)
        if args.command in ("status", "fetch"):
            run_dir = args.run_dir
            if args.command == "status":
                # just read job state, don't fetch
                
                state = read_state(run_dir)
                for name, job_id in state["jobs"].items():
                    letter, exit_status = job_state(ssh, state, name)
                    print(
                        f"{name} {job_id}: {JOB_STATES.get(letter, letter)}"
                        + ("" if exit_status is None else f", exit {exit_status}")
                    )
                return 0
            print(fetch(run_dir, ssh, cfg["poll_seconds"])["status"])
            return 0
        #
        plan = make_plan(args.model_config, cfg, ssh, root)
        print(describe(plan, cfg))
        if args.command == "plan":
            print("Plan only: nothing was uploaded or submitted.")
            return 0
        if not args.yes and confirm("Type yes to upload and submit: ").strip() != "yes":
            print("Cancelled: nothing was uploaded or submitted.")
            return 1
        run_dir = new_run_dir(root / RUNS_DIR, args.label, datetime.now())
        state = submit(plan, cfg, ssh, run_dir, not args.skip_login_check, root)
        print(f"Submitted {state['jobs']}; run folder {run_dir}")
        if args.command == "run":
            print(fetch(run_dir, ssh, cfg["poll_seconds"])["status"])
        return 0
    except BlockedError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        if run_dir is not None:
            print(f"Run folder: {run_dir}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if run_dir is not None:
            print(
                f"\nStopped waiting; jobs keep running on Gadi. Resume with: "
                f"python -m scripts.nci.remote fetch {run_dir}",
                file=sys.stderr,
            )
        return 130


if __name__ == "__main__":
    sys.exit(main())
