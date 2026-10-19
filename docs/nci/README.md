# Running on NCI Gadi

Sections 1 to 5 are the deployment path for Component 06 training: the checked-in
`uv.lock`, Component 02 standard tables, Component 03 splits/seeds and Component 04
saved graphs. The runner is `python -m src.models.run`, as implemented under
`src/models/`. [Section 6](#6-the-sydney-fusion-runs-d09-end-to-end) builds the radar
and satellite tables and the D09 graphs on Gadi as well, because the archives they read
are only there. Nothing here downloads BoM data or uses legacy exports.

## Readiness

The scripts below prepare the locked environment and run a bounded synthetic or
real-input check. Verify your own NCI account, project membership, storage mounts,
queue access and job outputs before reporting completion. Private deployment logs and
job IDs belong with the experiment record, not the shared setup guide.

Real runs require Component 02 standard tables, Component 03 released splits/seeds,
and Component 04 validated graphs. Code in Git does not include those generated
artifacts. The input preflight reports missing files; it never creates substitutes.
A synthetic fixture checks software only and is labelled in its run configuration.

**Both GPU queues run the same environment.** `pyproject.toml` takes PyTorch 2.13.0's
CUDA 12.6 build on Linux (`[tool.uv.sources]`), which has kernels for the V100 (sm_70)
and the A100 (sm_80). PyPI's default Linux build is CUDA 13, which dropped the V100.
macOS and Windows keep PyPI's build. One `.venv`, prepared by
`scripts/nci/prepare_environment.pbs`, therefore serves:

| Queue | GPU | Per GPU | Charge | Measured, G+R on the 2022 tables |
|---|---|---|---|---|
| `gpuvolta` | V100 32 GB | 12 CPUs | 3 SU per CPU-hour (36 SU per GPU-hour) | ~212 s per epoch (raw-radar G+R) |
| `dgxa100` | A100 80 GB | 16 CPUs | 4.5 SU per CPU-hour (72 SU per GPU-hour) | ~124 s per epoch |
| `normal` | none | 16 CPUs | 2 SU per CPU-hour (32 SU per job-hour) | ~277 s per epoch |

`gpuvolta` is the much larger pool and usually starts sooner; `dgxa100` jobs have
waited more than a day for a node. `scripts/nci/check_environment.py` checks that the
environment's CUDA build matches `uv.lock` and, inside a GPU job, that the build has
kernels for the allocated GPU. Loading an NCI PyTorch module or replacing torch with
pip would bypass `uv.lock`; neither is part of this setup.
[NCI queue limits](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits),
[PyTorch 2.13](https://pytorch.org/blog/pytorch-2-13-release-blog/).

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

The project shares one environment per `uv.lock`, so a teammate links to it instead of
building their own. A private environment holds about 30,000 files, and project
scratch allows 202,000 files in total. The supplied
[preparation job](../../scripts/nci/prepare_environment.pbs) requests one CPU, 8 GB
RAM, 20 GB temporary job storage and 30 minutes in `copyq`:

```console
qsub scripts/nci/prepare_environment.pbs
```

It verifies the pinned official uv 0.10.5 Linux archive against its published checksum
and reviewed digest, and installs uv under the user's personal scratch directory. It
then looks for `/g/data/no50/rainflow/envs/<first 16 characters of uv.lock's
SHA-256>/`, set by `NCI_SHARED_ROOT`:

- If that environment exists and holds `READY`, the job only links the checkout's
  `.venv` to it.
- Otherwise it builds it with `uv sync --locked`, writes `READY`, and makes the whole
  folder read-only. Two jobs never build the same environment, because the folder is
  created as a lock. A failed build leaves the folder without `READY`; remove it and
  submit again.

A changed `uv.lock` gets a new folder beside the old one, so an environment a job is
using is never rebuilt underneath it. A checkout that still has its own `.venv`
directory is refused rather than replaced; remove it once no job uses it.

The interpreter is a conda-forge Python 3.14, because uv's own Python 3.14 segfaults on
Gadi. It is installed once for the project at `/g/data/no50/rainflow/python/py314` by
[`prepare_python.pbs`](../../scripts/nci/prepare_python.pbs), which takes a
micromamba binary:

```console
qsub -v NCI_MICROMAMBA=/path/to/micromamba scripts/nci/prepare_python.pbs
```

Every job script mounts `gdata/no50` (`#PBS -l storage=scratch/no50+gdata/no50`)
so it can reach the shared environment.

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

## 5. Run from your own machine with one command

`scripts/nci/remote.py` does sections 2 to 4 for you from a clean local checkout, and
brings the results back:

```console
uv run python -m scripts.nci.remote plan config/models/hgnn.yaml
uv run python -m scripts.nci.remote run  config/models/hgnn.yaml --label sydney-baseline
```

[AUTOMATED_RUNS.md](AUTOMATED_RUNS.md) covers the one-time ssh setup, every command, what
happens at each step, where everything is kept, and what to do when something fails.

## 6. The Sydney fusion runs (D09), end to end

The radar (`rq0`) and satellite (`rv74`) archives are read in place on Gadi, so the grid
tables, the graphs built from them and the training all run there. The gauge tables are
built on your own machine and uploaded; they are small.

| Step | Where | Job | Writes |
|---|---|---|---|
| Environment | `copyq` | `prepare_environment.pbs` (section 3) | `.venv/` |
| Gauge tables | your machine | Component 02, then upload | `database/standard/gauge_hourly.csv`, `stations.csv` |
| Grid tables | `normal`, 1 CPU, 32 GB | `build_grids.pbs` | `database/standard/{radar,satellite}_{hourly,nodes}.csv`, `grid_report.md` |
| Graphs | `normal`, 24 CPUs, 96 GB | `build_graphs.pbs` | `database/graphs/d09_round_robin_{g,gr,grs}/` |
| Training | `gpuvolta` (1 V100) or `dgxa100` (1 A100); `normal` on CPU | `train_volta.pbs`, `train.pbs`, `train_cpu.pbs` | `experiments/nci/<run>/` |

The three runs are G, G+R and G+R+S on the round-robin split
([DEC-03](../DECISIONS.md#dec-03)), configured in `config/graphs/d09_round_robin_*.yaml`
and `config/models/d09_round_robin_*.yaml`. Each HGNN run also scores IDW and network
mean on the same rows. All three drop the hours radar or satellite lack
([DEC-01](../DECISIONS.md#dec-01)), so every row of D09's table covers the same hours.

### Before the first job

On a login node, confirm the archive memberships with the counts NCI's public listing
gives, and the quotas:

```console
ls /g/data/rq0/rainfields3/71/2021/prcp-m15 | wc -l      # 359 daily files
ls /g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph/v2.1/2022/01/03 | wc -l   # 142
lquota
nci_account -P no50
```

Check out the revision to run under `/scratch/no50/$USER/COMP3888-m17-01-p23` (section
2) and prepare the environment (section 3).

### Gauge tables

The gauge tables come from the national download in `database.zip`: the stations inside
`config/ingestion/sydney.yaml`'s box are copied out with their files unchanged (184
stations, 4,784 files, each checked against the index's sha256), and Component 02 builds
the tables from them. Built with Unix line endings, `gauge_hourly.csv` has sha256
`7ccf6813…` and `stations.csv` `6f624f27…`, the same files D08's runs used, with the same
127 study stations as both released splits. Upload them through NCI's data mover:

```console
ssh gadi.nci.org.au mkdir -p /scratch/no50/YOUR_NCI_USERNAME/COMP3888-m17-01-p23/database/standard
rsync -av database/standard/gauge_hourly.csv database/standard/stations.csv \
  YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/no50/YOUR_NCI_USERNAME/COMP3888-m17-01-p23/database/standard/
```

Then on Gadi, `sha256sum database/standard/{gauge_hourly,stations}.csv` should print the
two hashes above.

### Time the grid readers first

Nobody has run the readers against the whole archive, so first read one week:

```console
qsub -v NCI_GRID_TRIAL=2022-01-01:2022-01-07 -l walltime=02:00:00 scripts/nci/build_grids.pbs
```

The trial writes to its own `database/grid-trial-<job>/`. Its log ends with
`GRID_STATUS=READY seconds=N` and the report's coverage lines. The full two years take
about `N × 730 / 7` seconds; if that is near 24 hours, raise the walltime when submitting
the full build (`qsub -l walltime=…`).

### Submit the chain

```console
bash scripts/nci/submit_pipeline.sh
```

From a login node, this submits the full grid build, then the graph build, then a
one-epoch G+R+S timing run, each starting only when the one before it succeeded
(`qsub -W depend=afterok`). It refuses to start if the gauge tables or `.venv` are
missing, or if any grid output already exists. `--no-timing` leaves out the timing run.

- **Grid build:** `grid_report.md` gives, per source, the hours present and missing
  over 2021–2022 and every hour not written, with its cause. That is the measurement
  [ISS-04](../ISSUES.md#iss-04) is waiting for; copy the report back.
- **Graph build:** each report's `Timeline (DEC-01)` line says how many hours were
  dropped and why; `dropped_hours.csv` lists them. The job then runs Component 06's
  input check on each set, so a mismatch shows up before any GPU time is spent.
- **Timing run:** its log's `TRAIN_STATUS=DONE seconds=N` covers five folds of one
  epoch plus prediction. A full run trains up to 100 epochs per fold, stopping after
  10 without improvement, so book well above `N × 30`, capped at the queue's 48 hours.

### Full runs

```console
qsub -l walltime=HH:MM:SS -v NCI_MODEL_CONFIG=config/models/d09_round_robin_g.yaml,NCI_OUTPUT_DIR=experiments/nci/d09-g scripts/nci/train.pbs
qsub -l walltime=HH:MM:SS -v NCI_MODEL_CONFIG=config/models/d09_round_robin_gr.yaml,NCI_OUTPUT_DIR=experiments/nci/d09-gr scripts/nci/train.pbs
qsub -l walltime=HH:MM:SS -v NCI_MODEL_CONFIG=config/models/d09_round_robin_grs.yaml,NCI_OUTPUT_DIR=experiments/nci/d09-grs scripts/nci/train.pbs
```

A run never overwrites an existing output folder. Copy each folder back with
`rsync -av YOUR_NCI_USERNAME@gadi-dm.nci.org.au:/scratch/no50/YOUR_NCI_USERNAME/COMP3888-m17-01-p23/experiments/nci/ experiments/nci/`.

### One job per fold

A 48-hour request waits longest and holds back the most SU while it waits. Five
few-hour jobs per configuration start sooner and run side by side. Train each fold on its
own, then join and score. One A100 epoch of G+R+S takes about 150 s on 2021–2022, so
100 epochs need about 4.2 hours; book `05:00:00` for 2021–2022, and about half for one
year of data:

```console
for f in 0 1 2 3 4; do
  qsub -l walltime=05:00:00 -v NCI_MODEL_CONFIG=config/models/d09_round_robin_grs.yaml,NCI_OUTPUT_DIR=experiments/nci/d09-grs-fold$f,NCI_FOLDS=$f scripts/nci/train.pbs
done
# once all five have ended:
.venv/bin/python -m src.models.combine_folds experiments/nci/d09-grs-joined experiments/nci/d09-grs-fold{0,1,2,3,4}
qsub -l walltime=01:00:00 -v NCI_MODEL_CONFIG=config/models/d09_round_robin_grs.yaml,NCI_OUTPUT_DIR=experiments/nci/d09-grs,NCI_PREDICT_FROM=experiments/nci/d09-grs-joined/best.pt scripts/nci/train.pbs
```

Each fold job writes a `PARTIAL` report with no scores. `combine_folds` refuses a fold
whose training was cut off, so a fold that hits its walltime must be run again. The
scoring job predicts every method on every fold and writes the full report.
`NCI_MODEL_SEED=<n>` repeats a run with another model seed.

### Cost and space

Charges follow NCI's published rates (check them before booking): `normal` 2 SU per CPU
hour, with every 4 GB of memory counted as one CPU; `dgxa100` 4.5 SU per CPU hour. A job
is charged for the time it uses, so these are ceilings at the default walltimes:

| Job | Charged as | Most it can cost |
|---|---|---|
| Grid trial (2 h) | 8 CPUs | 32 SU |
| Grid build (24 h) | 8 CPUs | 384 SU |
| Graph build (8 h) | 24 CPUs | 384 SU |
| Timing run (4 h) | 16 CPUs | 288 SU |
| Each full run (48 h) | 16 CPUs | 3,456 SU |

On `/scratch/no50` expect about 4 GB of grid tables (the satellite table is about 82
million rows), 11 GB of G+R+S graphs, 1.5 GB of G+R graphs, and about 0.4 GB of
predictions per run.

### TO VALIDATE on the first run

- The archive paths and counts above, from a Gadi shell rather than the public listing.
- How fast the readers are (the trial answers it) and whether 32 GB is enough.
- Whether 96 GB holds the satellite table while the graphs are built.
- How long one G+R+S epoch takes on an A100. Measured on 2026-10-17 (job 181471772,
  2021–2022, fold 0): about 150 s per epoch after about 3 minutes of loading, against
  about 9.6 minutes per epoch on 16 CPUs. The job waited two hours for its A100.
- The queue limits and charge rates used above.

The remote runner in section 5 submits this same `train.pbs`. It passes its own project,
queue and resources on the `qsub` command line, which override the defaults above.
