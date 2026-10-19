# Running experiments on Gadi

`scripts/nci/remote.py` sends a committed model config from your own machine to NCI's
Gadi supercomputer as one job per fold, scores the run when every fold has finished, and
brings the results back. You never log in to Gadi, write a PBS script, or copy data. If
you have not set up your account yet, start with [ONBOARDING.md](ONBOARDING.md).

## Commands

| Command | What it does |
|---|---|
| `uv run python -m scripts.nci.remote plan config/models/<run>.yaml` | Runs every check and prints what would be submitted and what it can cost. Changes nothing. |
| `uv run python -m scripts.nci.remote submit config/models/<run>.yaml --label <name>` | The same checks, then asks for `yes` and submits. |
| `uv run python -m scripts.nci.remote status` | Shows every job of every run you have not fetched yet. Name a run folder to see only that run. |
| `uv run python -m scripts.nci.remote fetch` | Downloads every finished run you have not fetched yet into its `results/` folder. Unfinished runs are left alone and reported. |
| `uv run python -m scripts.nci.remote resume experiments/nci/<run>` | Resubmits the folds of a run that did not all succeed, continuing each from where it stopped, then scores again. |
| `uv run python -m scripts.nci.remote doctor` | Checks your SSH login, project membership, the shared environment and the project's quotas. |

`plan` and `submit` take `--queue gpuvolta`, `--queue dgxa100` or `--queue normal`. The
default, `auto`, picks the GPU queue with fewer jobs waiting for each job running, then
switches to `normal` if a fold would finish sooner on CPUs. That is the case for G+R+S,
whose folds take about 60 seconds per 1,000 hours on an A100 and 35 on 16 CPUs.
`submit --wait` waits for the run and fetches it. `fetch --with-predictions` also
downloads each run's `predictions.csv`, which is several hundred MB.

Your run folder is `experiments/nci/<date>_<time>_<label>/` on your machine. Git ignores
it.

## What a run does

`plan` and `submit` read only committed files on your machine; no data needs to be on
your machine at all.

1. **Checks.** Your tracked files must be committed, because Gadi runs exactly that
   commit. The split and seeds files must be committed too. The tables the config reads
   must be listed in [`config/nci/datasets.yaml`](../../config/nci/datasets.yaml), and
   the folder on Gadi must hold the same files, checked by checksum. `graphs_dir` must
   be the `output_dir` of exactly one committed `config/graphs/*.yaml`, built for the
   same split. The project must have enough SU and space, counting a graph build if
   one is needed. Anything that fails stops the plan before anything is sent.

   The plan also warns, without stopping, when the run could use more than a quarter of
   the SU the project has left, when scratch or `/g/data` is more than 90 % full by size
   or by file count, or when the chosen queue has more than twice as many jobs waiting
   as running. Read the warnings before typing `yes`.
2. **Code.** The commit goes to Gadi as a git bundle, into your checkout under
   `/scratch/no50/<you>/rainflow/repo`, and the run gets its own copy of it.
3. **Environment.** Every run uses the project's shared environment for its `uv.lock`.
   If nobody has built that one yet, a `copyq` job builds it first.
4. **Graphs.** Graphs are built on Gadi into a cache the project shares, named by a
   fingerprint of everything that decides them: the graph config, split, seeds, tables
   and the code in `src/graph`, `src/canonical` and `src/study`. A run that finds its
   graphs already built reuses them; otherwise a CPU job builds them before the folds
   start. Changing a model setting, such as `hidden_channels`, reuses the same graphs.
5. **Folds.** Each of the five folds is its own job, so each can start as soon as a GPU
   is free. The folds do not wait for each other.
6. **Scoring.** One CPU job waits for all five folds, joins them and scores the HGNN,
   IDW and network mean on the full set of test rows. PBS emails you when it ends.

## Queues and walltime

| Queue | GPU | Per fold job | Charge |
|---|---|---|---|
| `gpuvolta` | V100 32 GB | 12 CPUs, 1 GPU, 90 GB | 36 SU per hour |
| `dgxa100` | A100 80 GB | 16 CPUs, 1 GPU, 128 GB | 72 SU per hour |
| `normal` | none | 16 CPUs, 64 GB | 32 SU per hour |

A fold's walltime is set from the measured time per epoch for that queue and input
layers, in [`config/nci/remote.yaml`](../../config/nci/remote.yaml), for the config's
full number of epochs, plus 30 % and 15 minutes. Shorter, accurate requests start sooner,
because the scheduler fits them into gaps before larger jobs. Most folds stop early, so
they finish well inside it. The plan says when a layer set has no measured time on the
chosen queue.

## When a fold runs out of time

Resuming is never automatic: nothing continues a stopped fold unless you run `resume`.
A fold saves `last.pt` after every epoch: the weights, the optimizer, the early-stopping
count and every random generator's state. If PBS stops the job at its walltime, the
scoring job never starts, and `fetch` reports the run as failed. Then run

```console
uv run python -m scripts.nci.remote resume experiments/nci/<run>
```

It resubmits only the folds that did not succeed. A fold with a `last.pt` carries on
from its last finished epoch, and the result is the same as if it had never stopped,
which `tests/test_model_resume.py` checks. A fold that never started runs from the
beginning. A new scoring job follows. `--queue` moves the remaining folds to another
queue.

## Shared data

Nothing a run reads is uploaded from your machine.

- **Tables** are published once into `/g/data/no50/rainflow/data/<name>-<fingerprint>/`
  by [`publish_tables.pbs`](../../scripts/nci/publish_tables.pbs), then listed in
  `config/nci/datasets.yaml` under the folder configs read locally, such as
  `database/standard`. A published folder is read-only and never changed; new tables
  get a new folder.
- **Environments** live in `/g/data/no50/rainflow/envs/<uv.lock hash>/`
  ([README section 3](README.md#3-prepare-the-exact-environment-before-the-gpu-job)).
- **Graphs** live in `/scratch/no50/rainflow/graphs/<fingerprint>/`. Scratch deletes
  files nobody has read for 100 days; a run that finds its graphs incomplete builds
  them again.

## Where things are

| What | On your machine | On Gadi |
|---|---|---|
| Run state | `experiments/nci/<run>/state.json` | |
| Run config | `experiments/nci/<run>/run.yaml` | `/scratch/no50/<you>/rainflow/runs/<run>/run.yaml` |
| Code | your checkout | `.../runs/<run>/code/` |
| Fold results | `experiments/nci/<run>/results/output/fold-<n>/` | `.../runs/<run>/output/fold-<n>/` |
| Scored run | `experiments/nci/<run>/results/output/scored/` | `.../runs/<run>/output/scored/` |
| Job logs | `experiments/nci/<run>/results/logs/` | `.../runs/<run>/logs/<job>.log` |

## `state.json`

It records the commit, the queue and walltime, the shared environment and graph folders,
every job ID and the log each writes, which jobs wait for which, and the outcome:
`SUBMITTED`, `RESUMED`, `SUBMIT_FAILED`, then after `fetch` either `DONE` or
`FAILED: <job> ...`. `previous` keeps the job IDs of earlier attempts after a `resume`.

## When something goes wrong

- **`BLOCKED` before submitting.** Nothing was sent. The message names what to fix.
- **`SUBMIT_FAILED`.** The message lists the jobs that were already submitted and the
  `qdel` command to cancel them.
- **A fold failed.** Read its log in `results/logs/` after `fetch`, or on Gadi under the
  run's `logs/`. Exit status 271 means PBS stopped it, usually at its walltime; use
  `resume`.
- **The environment or graph job failed.** The folds never start. Read that job's log,
  fix the cause, commit, and submit a new run.
- **A teammate's run uses the same graphs.** That is expected: the cache is shared.

## TO VALIDATE

PBS mail (`-m ae`) reaches the address NCI holds for the user. The email for scoring
job 181547997 confirmed it on 2026-10-19.


- A full run through `submit` on Gadi, from `plan` to `fetch`.
