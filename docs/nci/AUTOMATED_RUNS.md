# Automated Gadi runs

`scripts/nci/remote.py` sends one Component 06 training run from your own machine to
NCI's Gadi supercomputer and brings the results back. You do not log in to Gadi or copy
files by hand: the script checks that Gadi can take the run, uploads what it needs,
submits the jobs, waits for them and downloads the run folder. It automates sections 2
to 4 of the [Gadi guide](README.md).

It has been tested against a stand-in for Gadi only. Nothing has run through it on Gadi
yet; what still needs checking is listed under [TO VALIDATE](#to-validate).

## One-time setup

The script never handles passwords, so ssh must log in on its own.

1. Add your public key to Gadi's `~/.ssh/authorized_keys` (for example with
   `ssh-copy-id`).
2. Put your NCI username in `~/.ssh/config`:

   ```text
   Host gadi.nci.org.au gadi-dm.nci.org.au
       User YOUR_NCI_USERNAME
       IdentityFile ~/.ssh/id_ed25519
   ```

3. Check that `ssh gadi.nci.org.au hostname` works without a prompt. On Windows, use the
   same OpenSSH client for this check and for the script.

The project, hosts, folders, job resources and charge rate are in
[`config/nci/remote.yaml`](../../config/nci/remote.yaml); your username is not, because
the script asks Gadi for it. The committed job is one A100 on `dgxa100` with 16 CPUs,
48 GB and a one-hour walltime, so it can cost at most 72 SU. Raise `walltime` there for a
real training run.

## Commands

```console
uv run python -m scripts.nci.remote plan   config/models/hgnn.yaml
uv run python -m scripts.nci.remote run    config/models/hgnn.yaml --label sydney-baseline
uv run python -m scripts.nci.remote submit config/models/hgnn.yaml --label sydney-baseline
uv run python -m scripts.nci.remote status experiments/nci/<run folder>
uv run python -m scripts.nci.remote fetch  experiments/nci/<run folder>
```

| Command | What it does |
|---|---|
| `plan` | Runs every check and prints the plan. Changes nothing, locally or on Gadi. |
| `run` | Checks, prints the plan, asks you to type `yes`, submits, then waits and downloads. |
| `submit` | The same as `run`, but returns once the jobs are submitted. |
| `status` | Shows each job's state for a run folder. |
| `fetch` | Waits for a submitted run's jobs to end, then downloads it: after `submit`, after Ctrl-C, or days later. |

`--label` names the run folder (letters, digits, `.`, `_` and `-`). `--yes` skips the
confirmation. `--skip-login-check` skips the input check on the login node
([step 4](#4-submit)).

## What happens in a run

### 1. Checks, before anything changes

- Gadi runs the committed revision exactly, so uncommitted edits to tracked files are
  refused. Untracked files are ignored.
- Every input the model config names must exist on your machine.
- One ssh call asks Gadi whether its copy of the code already has this commit, whether
  these inputs are already uploaded, whether the shared Python environment exists and
  which `uv.lock` it was built from, the project's storage quota (`lquota`) and SU
  balance (`nci_account`), and which of your jobs have not finished.

The plan is refused if the upload would pass the size or file-count quota, if the job's
most expensive case (every CPU for the whole walltime, at the queue's charge rate) is
more than the SU left, or if the run needs a different environment while a job this
script sent is unfinished ([step 3](#3-the-environment)).

### 2. Upload

- **Code.** Sent as a git bundle of the commits Gadi's copy lacks (nothing, if it already
  has this one), then checked out into a new folder for this run. Every run has its own
  copy of the code at its own commit.
- **Inputs git does not carry:** the Component 02 tables, the Component 03 split and the
  Component 04 graphs. They are uploaded once, into a folder named by a fingerprint of
  their contents. Gadi checks every file's checksum, then makes the folder read-only,
  and a later run with the same inputs reuses it. Inputs that git does carry travel with
  the code. Raw source data is never sent.
- **The run's config.** The model config is rewritten as `run.yaml`, pointing at the
  uploaded inputs and at this run's output folder, with `device: cuda`.

### 3. The environment

Every run uses one shared Python environment on Gadi (`.venv`), built by
`uv sync --locked` from `uv.lock`. Each run's code folder links to it.

If the environment is missing, or was built from a different `uv.lock`, a setup job
(`prepare_environment.pbs`: `copyq`, 1 CPU, up to 30 minutes) is submitted first, and the
training job waits for it to succeed. If the setup job fails, PBS drops the training job.

**The rule:** while any job this script sent is still queued or running, a run that needs
a different environment is refused before anything is uploaded, because rebuilding the
environment would change the packages under that job. Wait for the jobs the message
names, or cancel them with `qdel`, then submit again. A run whose `uv.lock` matches the
environment's goes ahead alongside unfinished jobs. If an earlier setup job failed or has
not run yet, a second one is refused in the same way, though that refusal comes after
the upload.

### 4. Submit

When the environment is ready, Component 06's input check (`--check-inputs`) runs on the
login node first, so a missing or malformed input fails before any GPU time is booked.
Then `scripts/nci/train.pbs` is submitted with the queue and resources from
`config/nci/remote.yaml`. On the GPU node it checks the environment and CUDA, then runs
`python -m src.models.run`.

Each job id is written to `state.json` as soon as `qsub` returns it.

### 5. Wait and download

Every `poll_seconds` (five minutes in the committed config) the script reads the end of
each job's log in the run folder. Gadi closes every job log with a "Resource Usage" block
that includes `Exit Status`; once that block is there, the job has ended with that
status. Until then, `qstat` reports whether the job is queued or running. The log stays on
Gadi after `qstat` has forgotten the job, so `fetch` works days later. A dropped
connection is tried again three times before the wait stops, and `fetch` resumes it.

When every job has ended, the run folder (`output/`, `pbs.log`, `prepare.log` and
`run.yaml`) is downloaded into `results.partial/`, which is renamed `results/` only once
the download is complete. An existing `results/` is never overwritten.

The run is `DONE` only if every job exited 0, `predictions.csv` and `report.md` are
there, and the report does not say `FAILED` or `RUNNING`. Anything else is
`FAILED: <reason>`.

## Where things are

| Where | Path | Holds |
|---|---|---|
| Your machine | `experiments/nci/<date>_<time>_<label>/` | `state.json`, `run.yaml`, and `results/` after `fetch` |
| Gadi scratch | `/scratch/<project>/<user>/rainflow/repo/` | Gadi's copy of the code and the shared `.venv` |
| Gadi scratch | `/scratch/<project>/<user>/rainflow/runs/<run>/` | the run's `code/`, `run.yaml`, `output/`, `pbs.log` and `prepare.log` |
| Gadi gdata | `/g/data/<project>/<user>/rainflow-inputs/<fingerprint>/` | uploaded inputs, read-only, shared by runs with the same inputs |

`<project>` is `no50` in the committed config. A run folder is named by the local time it
was created; seconds are added if that name is taken.

## `state.json`

| `status` | Meaning |
|---|---|
| `UPLOADING` | Submit started and has not finished. `jobs` lists anything already submitted. |
| `SUBMIT_FAILED` | A step failed and `error` says which. `jobs` lists anything already submitted, which may still be on Gadi. |
| `SUBMITTED` | Every job was accepted; `jobs` holds their ids. |
| `DONE` | Fetched, and the run completed ([step 5](#5-wait-and-download)). |
| `FAILED: <reason>` | Fetched, but a job failed or the outputs do not show a completed run. |

It also records the revision; every input with its sha256 (for files git carries, the
committed copy's, which is what Gadi reads); the inputs' fingerprint; every Gadi path;
the job resources; the most the job can cost in SU; and, after `fetch`, each job's exit
status.

## When something goes wrong

- **`BLOCKED: ...`** The message says what to fix. At the plan stage nothing was uploaded
  or submitted; after that, `Run folder:` points at the `state.json` that says what was.
- **`SUBMIT_FAILED` with a setup job listed.** The setup job stays queued and builds the
  environment that the next run reuses. Cancel it with the `qdel` command the message
  gives if you do not want it.
- **Ctrl-C while waiting** stops only the local wait; the jobs keep running. Resume with
  `fetch <run folder>`.
- **"may never have started"** from `fetch` or `status`: neither the job's log nor
  `qstat` knows the job, for example because it was removed while still queued.
- **Environment refused.** Wait for the jobs named, or `qdel` them, then submit again.

## TO VALIDATE

The tests use a stand-in for Gadi. These have not been checked on Gadi itself:

- `qstat -F json` output, including `PBS_O_WORKDIR` under `Variable_List`;
- `lquota`, `nci_account`, `qselect` and `qsub` being on the `PATH` for non-interactive
  ssh commands;
- `git worktree` sharing one `.venv` through a symlink.

Checked on Gadi on 2026-10-17: job logs end with the "Resource Usage" block and its
`Exit Status`, on both a failed and a successful job.
