# Training on NCI Gadi

This is the deployment path for Component 06. It uses the checked-in `uv.lock`,
Component 02 standard tables, Component 03 splits/seeds and Component 04 saved graphs.
It does not download BoM data, build the preceding components or use legacy exports.
The runner is `python -m src.models.run`, as implemented under `src/models/`.

## Readiness

The scripts below prepare the locked environment and run a bounded synthetic or
real-input check. Verify your own NCI account, project membership, storage mounts,
queue access and job outputs before reporting completion. Private deployment logs and
job IDs belong with the experiment record, not the shared setup guide.

Real runs require Component 02 standard tables, Component 03 released splits/seeds,
and Component 04 validated graphs. Code in Git does not include those generated
artifacts. The input preflight reports missing files; it never creates substitutes.
A synthetic fixture checks software only and is labelled in its run configuration.

**The default queue is `dgxa100`.** The current lock pins PyTorch 2.13.0 with CUDA 13
dependencies on Linux. CUDA 13 does not support Volta/V100 GPUs, so the usual
`gpuvolta` example is incompatible. The supplied smoke job requests one A100 with
16 CPUs, 48 GB RAM and ten minutes. Its estimated maximum charge is 12 SU
(`16 × 4.5 × 10/60`), assuming the published charging rules. This is shared project
usage, not a personal allowance. [NCI queue limits](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits),
[PyTorch 2.13](https://pytorch.org/blog/pytorch-2-13-release-blog/),
[NVIDIA CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

The actual A100 driver and queue access still need checking. A CUDA 12 build for V100
would require a separately reviewed dependency/index/lock change. Loading an older
NCI PyTorch module or replacing torch with pip would bypass repository dependency
authority; neither is part of this setup.

## 1. Establish access and check the project

The verified browser route is [NCI ARE](https://are.nci.org.au/), followed by
**Gadi Terminal**. MyNCI is the account/project dashboard and uses a separate login;
being signed into MyNCI alone does not establish a Gadi shell. Complete credentials
privately, then verify identity and mounts in the terminal. ARE's file browser can
stage a reviewed code archive in the user's home folder; verify its checksum and
unpack it into an unused personal scratch directory from Gadi Terminal.

Complete the first login privately in a local terminal, replacing `YOUR_NCI_USERNAME`
with your assigned NCI username, if using SSH instead:

```console
ssh YOUR_NCI_USERNAME@gadi.nci.org.au
```

Verify any new host-key prompt through NCI's trusted access information and enter your
password yourself. Do not put passwords or private keys in commands, config, chat or
the repository. The scripts neither change SSH trust nor manage credentials.
[NCI login guide](https://opus.nci.org.au/spaces/Help/pages/12583143/How+to+login+to+Gadi).

Once authenticated, inspect without changing anything:

```console
hostname
id
groups
printf '%s\n' "$PROJECT"
nci_account -P no50
qstat -u "$USER"
ls -ld /scratch/no50 /g/data/no50
```

Check current storage **and inode quotas** for `no50` in MyNCI, in addition to compute
balance. A successful `ls` or free space from `df` does not prove project quota is
available. Inspect the team's actual directories before selecting a destination.
The commands below use `/scratch/no50/$USER/COMP3888-m17-01-p23` as a proposed personal
checkout, not a claim that this directory already exists. Source-project access is
separate from compute-project membership.

## 2. Prepare a Linux checkout and inputs

Prefer checking out the agreed revision on Gadi so run metadata retains its Git SHA.
Record any uncommitted changes as well; a commit SHA alone does not identify them.
Do not reuse or copy the macOS `.venv` on Linux.

For a reviewed manual transfer, first create the intended personal destination on
Gadi. NCI recommends its data movers for transfers. From the **local repo root**, this
template previews the code transfer without modifying the destination:

```console
rsync -avn --exclude='.venv/' --exclude='database/' --exclude='experiments/' \
  --exclude='.env*' --exclude='local/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
  ./ YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/no50/YOUR_NCI_USERNAME/COMP3888-m17-01-p23/
```

Review the file list and destination, then remove only `n` from `-avn` to perform the
transfer. This includes Git metadata for provenance; inspect remotes/config for embedded
credentials first. Do not transfer the whole personal vault. Copy released standard
tables and graphs separately only if they are not already accessible on Gadi. Leave
the original source files unchanged.

Set the model YAML paths to the verified files:

- `gauge_file`: Component 02 `gauge_hourly.csv`.
- `stations_file`: Component 02 `stations.csv`.
- `split_file`: Component 03 released five-fold JSON.
- `seeds_file`: Component 03 `config/seeds.yaml`.
- `graphs_dir`: Component 04 directory containing
  `fold-<i>/{train,validation,test}.pt` for every released fold.
- `output_dir`: a writable personal run destination with sufficient quota.

Relative paths resolve from the repository root. These are the runner's own config
keys, not a second data interface. Graph files must be trusted team-produced artifacts;
their presence is not scientific validation. Component 06 does not rebuild graphs or
splits. Review the [archived Sprint 1 06 contract](../archive/sprint1/06_MODEL_ADAPTER_REQUIREMENTS.md) and
the adjacent component outputs before production training.

## 3. Prepare the exact environment before the GPU job

Use a Linux `uv` installation and Python satisfying the checked-in `requires-python`
(`>=3.14` currently). Dependency authority is
[AGENTS.md](../../AGENTS.md), `pyproject.toml` and `uv.lock`.
See [uv installation](https://docs.astral.sh/uv/getting-started/installation/) and
[locked project environments](https://docs.astral.sh/uv/concepts/projects/sync/).

Run `uv sync --locked` from the checkout on a permitted network-enabled preparation
host/job, then run the input preflight. Do not install or resolve packages inside the
GPU training script. Gadi's `copyq` provides external networking for batch work;
ordinary compute queues do not. [NCI queue structure](https://opus.nci.org.au/spaces/Help/pages/90308824/Queue+Structure).

The supplied [preparation job](../../scripts/nci/prepare_environment.pbs) requests
one CPU, 8 GB RAM, 20 GB temporary job storage and 30 minutes in `copyq`:

```console
qsub scripts/nci/prepare_environment.pbs
```

It verifies the pinned official uv 0.10.5 Linux archive against its published checksum
and reviewed digest, installs uv only under the user's personal scratch directory,
and runs `uv sync --locked --python 3.14.3`. The interpreter and environment persist
on scratch; the download cache is temporary job storage. It preserves an existing,
different uv binary and does not change shell profiles or dependency authority files.
The Python pin matches the local validation environment.

After installation it checks the environment, generates a new explicitly synthetic
fixture for that job, and validates its inputs. Successful output includes
`ENVIRONMENT_STATUS=READY` and `NCI_FIXTURE_CONFIG` with the resulting absolute path.
This establishes a prepared Linux environment, not GPU compatibility or real-data
readiness. Missing production inputs remain a separate dependency. Retain the job
log and verify PBS `Exit_status=0` before using the printed fixture for a GPU check.

If installation fails, preserve the exact stage and error. Do not edit the lock or
fall back to a different Python or torch build merely to get a successful run.

## 4. Check inputs, then submit a bounded smoke run

On a login node, this is read-only and uses the prepared environment:

```console
bash scripts/nci/preflight.sh config/models/hgnn.yaml
```

It verifies Python, installed direct dependency versions against `uv.lock`, the torch
CUDA build on Linux, and invokes the runner's `--check-inputs --device cpu` so a CUDA
config can still have its input structure checked on a login CPU. It does not allocate
a GPU or prove driver compatibility. It is also not a replacement for `uv sync --locked`
or the repository acceptance tests.

The PBS file mounts only `scratch/no50`. Before submission, add every verified
`gdata/<project>` or other scratch project needed by the checkout, interpreter, inputs
or outputs to its `storage=` directive (or override it on the command line). A path
visible on the login node can be absent inside PBS if its mount was omitted.
[NCI PBS storage directives](https://opus.nci.org.au/spaces/Help/pages/90308829/PBS+Directives+Explained).

After those checks, from the Gadi repository root:

```console
qsub scripts/nci/train_smoke.pbs
```

To use a separately reviewed small-input config while retaining the same training
entrypoint and released data contract:

```console
qsub -v NCI_MODEL_CONFIG=/absolute/path/to/smoke.yaml scripts/nci/train_smoke.pbs
```

`NCI_SMOKE_EPOCHS` defaults to 2 and accepts only 1 or 2. The job overrides `device` to
`cuda`, checks the allocated GPU, driver and a CUDA forward/backward operation, reruns
the input preflight, then invokes:

```console
.venv/bin/python -B -u -m src.models.run config/models/hgnn.yaml --epochs 2 --device cuda
```

The ten-minute limit bounds runtime; it does not guarantee five full production folds
finish within that time. A terminated smoke run is a failure to inspect, not a valid
scientific result. The job writes its environment, lock hash and code revision to the
PBS log; retain that log beside the resulting run record. It performs no downloads.

Inspect the returned job ID and both scheduler state and outputs:

```console
qstat -swx JOB_ID
qstat -xf JOB_ID
```

Check `Exit_status`, the `rainflow-smoke.o*` log, and the runner's `config.yaml`,
`history.csv`, `best.pt`, `predictions.csv` and `report.md`. Submission or a disappearing
queue entry alone is not completion. Require the runner's exact target-row check and
review predictions before scheduling a longer run. No production training duration,
checkpoint recovery or A100 speed has yet been verified on Gadi.
