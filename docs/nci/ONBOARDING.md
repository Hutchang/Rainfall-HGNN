# Running your first Gadi experiment

This is the one-time setup for a teammate who wants to run experiments on NCI's Gadi
supercomputer. After it, an experiment is a commit and one command. Everything is done
from your own machine; you never need to edit files on Gadi.

## 1. Get an NCI account in project no50

1. Sign up at [MyNCI](https://my.nci.org.au/) if you have no NCI account.
2. Ask to join project **no50** from MyNCI and tell the project lead, who approves it.
3. Wait for the approval email. Your NCI username looks like `ab1234`.

## 2. Let your machine log in to Gadi without a password

The tool never handles passwords, so SSH must log in by key.

1. If you have no SSH key, make one: `ssh-keygen -t ed25519` (press Enter to accept the
   defaults).
2. Copy it to Gadi, replacing `ab1234` with your username. This asks for your NCI
   password once:

   ```console
   ssh-copy-id -i ~/.ssh/id_ed25519.pub ab1234@gadi.nci.org.au
   ```

   On Windows, where `ssh-copy-id` is missing, log in with
   `ssh ab1234@gadi.nci.org.au` and paste the contents of `id_ed25519.pub` as a new line
   of `~/.ssh/authorized_keys` there.
3. Log in once to each of the two hosts the tool uses, and accept each host key when
   asked, after checking it against [NCI's published fingerprints](https://opus.nci.org.au/spaces/Help/pages/12583143/How+to+login+to+Gadi):

   ```console
   ssh ab1234@gadi.nci.org.au hostname
   ssh ab1234@gadi-dm.nci.org.au hostname
   ```

   `gadi-dm` is NCI's data mover, which carries uploads and downloads. Each command
   should then print a host name with no prompt.

## 3. Set up the repository

```console
git clone <the repository URL>
cd COMP3888-m17-01-p23
uv sync --locked
uv run python -m scripts.nci.remote doctor
```

Every command asks for your NCI username, so the same checkout works for whoever uses
it. To skip the question, add `--user ab1234` to the command or set the variable
`NCI_USERNAME=ab1234` in your shell. `status` and `fetch` reuse the username each run
was submitted with.

`doctor` checks your login, your membership of no50, that you can read the shared
environment, and the project's SU and file quotas. It prints `Ready to submit.` or says
what to fix.

## 4. Run an experiment

1. Copy a model config that already runs, for example
   `config/models/d09_round_robin_gr.yaml`, to a new file, and change what you are
   testing.
2. Commit it. Gadi runs exactly your commit, so the tool refuses uncommitted changes to
   tracked files.
3. See what would happen, then submit:

   ```console
   uv run python -m scripts.nci.remote plan   config/models/<yours>.yaml
   uv run python -m scripts.nci.remote submit config/models/<yours>.yaml --label <short-name>
   ```

   The plan shows the tables and graphs it will use, the queue, each fold's walltime and
   the most the run can cost. Type `yes` to submit.
4. Wait for the email from PBS saying the scoring job has ended, or check with
   `uv run python -m scripts.nci.remote status`.
5. Download every finished run:

   ```console
   uv run python -m scripts.nci.remote fetch
   ```

   Results land in `experiments/nci/<run>/results/`. The scores are in
   `output/scored/report.md`, and each fold's loss history is in
   `output/fold-<n>/history.csv`.

If a fold ran out of time, `fetch` says the run failed and
`uv run python -m scripts.nci.remote resume experiments/nci/<run>` carries on from where
it stopped. [AUTOMATED_RUNS.md](AUTOMATED_RUNS.md) explains every step and what to do
when something goes wrong.

## Rules that keep the project's runs safe

- **Commit, then submit.** A run is traceable only to a commit.
- **Use only published tables.** If your config reads tables not listed in
  `config/nci/datasets.yaml`, ask the project lead to publish them.
- **Do not delete or edit anything under `/g/data/no50/rainflow` or
  `/scratch/no50/rainflow`.** Everyone's runs read them. They are read-only for this
  reason.
- **SU are shared.** Every plan prints the most a run can cost. Plan before submitting,
  and cancel a run you do not need with `qdel <job IDs>` on Gadi.
- **Scratch is close to its file limit.** Do not build your own Python environment on
  Gadi; the shared one exists so that nobody needs to.
