# D07 — Experiment runner

Template: component standard (the eleven sections in
[AUTHORING_STANDARD.md](AUTHORING_STANDARD.md)). The rule for what counts as work, and the rules everyone follows, are in
[README.md](README.md).

## 1. Summary

One command runs the whole pipeline from the config files under `config/` as they are,
and leaves behind one folder per run that says exactly what went in and what came out.
Nothing ever overwrites a result.

**READ ME** (Guiding user story): *"I'm a researcher and
I have been running experiments with different gauge subsets,
different preprocessing settings and different splits. For each run I want to see the
past runs, the config each one used, and its results, so I can tell what change led to
what effect. If everything is written to one file, I will never know what changed over
time."* 

Each step of a run is one component's program — Component 02, 03, 04 or 06 — with its
own config file, reading the files the step before it wrote and writing the files the
step after it reads. Today that person is served for the last step (Component 06, the
model) only. Every component before it writes to one fixed folder, and one of them
overwrites silently.

## 2. System position

A run is four components in order. Each is a program that already exists; the runner
only decides where each one reads and writes, and records what happened.

| Component | Program | Reads | Writes |
|---|---|---|---|
| 02 — standard tables | `src.canonical.gauge.build`, plus `src.canonical.grids` when radar is on | the raw cache, through its index | `gauge_hourly.csv`, `stations.csv`, `days.csv`, and the grid tables |
| 03 — split | `src.study.split` | `stations.csv`, `config/seeds.yaml` | `<name>.json`, the fold summary and map |
| 04 — graphs | `src.graph.builder_new` | the tables, the split, the seeds | `fold-<i>/{train,validation,test}.pt` |
| 06 — model | `src.models.run` | the tables, the split, the seeds, the graphs | `predictions.csv`, `best.pt`, `history.csv`, `report.md` |

Component 01 (fetching) is not run by the runner. It fills the shared raw cache once,
ahead of any run, and every run reads from that cache; which part of the cache a run
reads is what FR05 records. Component 02 has two programs (gauge tables and, when
radar is on, grid tables); the runner treats them as one step.

```text
config/experiments/<name>.yaml    names the five component config files this run uses
      |
      v
config/ingestion/<region>.yaml, config/canonical/*.yaml, config/splits/<name>.yaml,
config/graphs/*.yaml, config/models/*.yaml    (edited in place, as today)
      |
      v
  runner  --copies all six, fills in defaults-->  experiments/<date>_<time>_<label>/config/
          --reads-->  database/raw/   (Component 01's cache: append-only, shared)
      |
      +--> 02 standard tables --> 03 split --> 04 graphs --> 06 model
      |
      v
  experiments/<date>_<time>_<label>/   one folder per run: config copies, provenance record, outputs, results
  experiments/registry.csv             one row per run, append-only
```

**Worked example.** A researcher lowers `min_coverage` from 0.90 to 0.85 (the share of
hours a station must have to enter the study) and reruns.

#### Today (pre-D07): 
`python -m src.canonical.gauge.build` writes over `database/standard/` with no
warning (culprit code: `src/canonical/gauge/build.py:97-136`). Every graph built from the old tables,
and every past model run's recorded input fingerprints (section 4 explains the word), now
point at bytes that no longer exist. The old result is detectable as stale but not recoverable, and any old `best.pt`
can never be replayed.

#### Tomorrow (post-D07): 
The runner writes a new run folder. The old folder is untouched. The
registry has two rows. `diff` (subtask D07.4) prints the one key that changed.

## 3. Inputs

| Input | Who owns it | Notes |
|---|---|---|
| the experiment file, `config/experiments/<name>.yaml` | the person running the experiment | five paths — one per component config — and an optional `label`. It names *which* files a run uses and holds no settings of its own. Named on the command line. |
| the component config files under `config/` | the person running the experiment, by editing them in place as today | `config/ingestion/<region>.yaml` (its `region` key drives every path), `config/canonical/standard.yaml`, `config/splits/<name>.yaml`, `config/graphs/gauge_baseline.yaml`, `config/models/hgnn.yaml` — or whichever files the experiment file points at. No new file format. |
| `database/raw/` | Component 01 | the raw cache. Shared, append-only. Files are keyed by station, month and source; index rows carry each file's SHA-256. |
| `config/seeds.yaml` | Component 03 | unchanged |
| `config/splits/<name>.json` | Component 03 | a *released* split is a pinned, named copy of Component 03's output. An experiment may reference one by name instead of running Component 03. |

## 4. Functionality requirements

### What is being built

One new module, `src/experiments/run.py`, with one command that takes an *experiment
file*:

```console
uv run python -m src.experiments.run config/experiments/sydney_baseline.yaml
```

The experiment file is five paths and an optional label. It says *which* component config
files this run uses and holds no settings of its own; the component files stay the ones
people edit:

```yaml
# config/experiments/sydney_baseline.yaml
label: blocked-folds                              # optional; goes in the run folder name
ingestion: config/ingestion/sydney.yaml
canonical: config/canonical/standard.yaml
split: config/splits/sydney_5fold_blocked.yaml    # or a released .json, to skip Component 03
graphs: config/graphs/gauge_baseline.yaml
model: config/models/hgnn.yaml
```

To run with a different model config, point `model:` at it — or copy the experiment file
under a new name, which leaves the combination itself in git. Without this file the
runner would have to guess which of several files in `config/splits/` or `config/graphs/`
a run meant, and it would guess wrong the day there are two.

The runner does five things, in order: it creates the run folder and refuses if the folder
exists; it copies the experiment file and the five files it points at into that folder; it calls the
existing component programs one after another — `src.canonical.gauge.build`,
`src.study.split`, `src.graph.builder_new`, `src.models.run` — telling each where to
read and where to write; it writes `provenance.json`, the record of what each component read
and wrote; and it appends one row to `experiments/registry.csv`. It makes nothing else:
no new file format, no server, no new model code.

Alongside it, three existing pieces change: each component program accepts its input and
output paths and the fold count from the runner instead of fixed defaults (D07.1);
Component 02's build refuses an existing output folder (D07.6); the gauge fetch stops
keeping stale state between runs (D07.6).

Two words used below. A **fingerprint** (also called a hash, or SHA-256 after the
method) is a short string computed from a file's bytes: the same bytes always give the
same string, and any change to the file gives a different one. The runner records one
for every file a component reads or writes, so a run can later prove "I read exactly this
file", and a rerun can tell when a file has been rewritten since. The **provenance
record** (`provenance.json`) is the per-run file that holds those fingerprints and the
code revision. It is not the per-source *manifest* of ISS-24: a manifest declares what a
data source *is* (units, coordinate system, time step, origin) and is written once per
source; the provenance record says what *this run used* and is written once per run.

The subtask id in brackets is the unit of work.

### FR01 [D07.1] — Paths come from the region and the run folder.

**Makes:** a path argument
on every component program that still lacks one (input directory, output directory, split
file), and a fold count read from the split file. **Rule:** no component writes to a fixed
directory. The ingestion config's `region` key names the region; every component's output goes
under the run folder (or the component cache, FR06). Today the fold count is fixed at five in
`src/graph/builder_new.py:89`, `src/models/contracts.py:69-71`, `src/models/evaluate.py:18`
and `src/models/run.py:99,235`.

### FR02 [D07.2] — One folder per run, named by date and time.

**Makes:** the run folder,
`experiments/<YYYY-MM-DD>_<HHMM>[_<label>]/`, for example
`experiments/2026-09-19_1432_blocked-folds/`. The folder name is the run's id: it sorts
in time order and reads at a glance. An existing folder is refused, never reused (copy
the pattern in `src/models/run.py:205-215`); two runs started in the same minute get the
seconds appended. `report.md` is written first with `Status: RUNNING` and updated to
`DONE` or `FAILED: <reason>` so a killed run is visible.

### FR03 [D07.2] — The config copies and the provenance record.

**Makes:** `<run>/config/`
and `<run>/provenance.json`. Before any component runs, the runner copies every config file
it read into `<run>/config/`, and beside each copy writes the *effective* version — the
file as read, with every default the code fills in written out — so two runs that differ
only in a defaulted key still differ in their copies. `provenance.json` then records, per
component: the fingerprint of that component's effective config; every input file with its
fingerprint (for Component 02, the raw files selected from the cache, taken from
the index rows); every output file with its fingerprint; the git revision and whether the
worktree had uncommitted changes; Python, torch and PyG versions. Reuse `provenance()`
and `file_digest()` from `src/models/run.py:168-202`, extended to every component. Remove the
stale `models/gnn.py` entry there; fingerprint the code under `src/canonical/`,
`src/study/`, `src/graph/` and `src/models/` as well.

### FR04 [D07.2] — The registry.

**Makes:** `experiments/registry.csv`, one row per run,
append-only. Columns: `run` (the folder name), `created_utc`, `label`, `region`,
`git_revision`, `worktree_dirty`, one `<component>_config_fingerprint` and one
`<component>_inputs_fingerprint` column per component (`c02`, `c03`, `c04`, `c06`), `status`, and the headline metrics from
`report.md` (`rmse_mm` and `correlation` per method, mean across folds). This is the
"see all my past runs" table.

### FR05 [D07.2] — Selection is explicit.

**Makes:** the inputs part of the provenance record
for Component 02. The ingestion config states the region, the period and any
station filter. The runner works out which raw files those select (from the index) and
lists them, with fingerprints, in `provenance.json`. Changing the selection is a new run;
the cache is never edited to change what a run sees.

### FR06 [D07.3] — Component cache.

**Backlog. Makes:** `database/cache/<component>/<fingerprint>/`,
where the fingerprint covers the component's effective config and its input
fingerprints. A run whose component fingerprint already exists reuses the cached output and says so in
`provenance.json`, instead of computing it again. This is what makes "same data, new
subset" cheap, and what stops every run folder holding its own gigabyte of graphs.

### FR07 [D07.4] — `list` and `diff`.

**Backlog. Makes:** two subcommands. `list` prints
the registry as a table. `diff <run_a> <run_b>` compares the two runs' effective config
copies and prints every key that differs, plus every component whose input fingerprint
differs. This is also what catches the "edited `config/` for one run and forgot" case:
the next run's copy shows the key.

### FR08 [D07.6] — The standard-table build never replaces existing tables.

**Must do:**
When `python -m src.canonical.gauge.build` is pointed at an output directory that already
holds any of `gauge_hourly.csv`, `stations.csv`, `days.csv` or `report.md`, it stops before
writing anything, names the directory and the file it found, and exits non-zero. The
person picks a fresh directory (or the runner picks one, FR01); the old tables are left
exactly as they were.

**Why:**
Today `build.py:97-136` creates the directory with `exist_ok=True` and writes with plain
`to_csv`, so a second build with different settings overwrites the tables with no
warning — the "I rebuilt the tables and now none of my old runs can be replayed" case
(ISS-28). The graph builder already behaves the required way
(`src/graph/builder_new.py:46-47`).

**Done when:** TEST04 passes.

### FR09 [D07.6] — The fetch does not keep stale state.

**Changes:**
`src/ingestion/gauge/fetch.py` and `src/ingestion/common.py`. When the gauge bounds or
period in the config differ from those recorded beside `gauge/stations.json`, the station
list is fetched again (`fetch.py:110-124` reads the old file back today). The index never
drops a row for a file that is still on disk (`common.py:65-77` rewrites the whole file
each flush; a narrower fetch today drops earlier rows and Component 02 then cannot see
those files).

### FR10 [D07.5] — The golden run.

**Makes:** `tests/data/golden/`: one real, small Sydney
run that reproduces itself byte for byte. Stations chosen by the archived rule in
[../archive/05_GOLDEN_FIXTURE_REQUIREMENTS_CUT.md](../archive/05_GOLDEN_FIXTURE_REQUIREMENTS_CUT.md)
(FR02 there: at most 25 stations around anchor `568163`); period three months. The
fingerprint of its `predictions.csv` is recorded in `tests/data/golden/EXPECTED.md`.

### FR11 — Overrides without editing `config/`.

**Backlog. Makes:** two conveniences.
A flag per component (`--model config/models/hgnn_small.yaml`) that replaces that entry
of the experiment file for one run, so trying another model config does not need a new
experiment file. And an optional overrides file (`--override sweep.yaml`) whose keys
replace the matching keys in the component configs, so a sweep of ten settings does not
mean ten edits to `config/`. The copies in the run folder show the result either way.

### FR12 [D07.7] — The coverage gate.

**Makes:** `pytest-cov` as a development dependency
(`uv add --dev pytest-cov`); a `[tool.coverage]` section in `pyproject.toml` that measures
the production packages and omits inherited review-only code; the two floors from rule
10 of [README.md](README.md) wired into the verification commands — 80 % over the
production packages with `--cov-fail-under=80`, and 90 % over the file-writing modules
listed in that rule with a second `coverage report --include=... --fail-under=90` — so
either floor failing fails the run; and today's baseline number recorded in
`docs/CURRENT_STATE.md`. Nobody knows that number yet; measuring it is the first step,
and if it is under a floor the gap is listed as work, not hidden by lowering the floor.

**Why FR06, FR07 and FR11 are backlog.** FR01–FR05 give the researcher the folder per
run, the config copies and the provenance record this week. The cache, `diff` and
overrides make it faster and nicer; they do not change what is recorded.

## 5. Non-negotiable rules

- Never overwrite. Never delete from the raw cache. A run folder is written once.
- The files under `config/` are the inputs people edit; the run folder holds copies.
  A result is read from its run folder, never from `config/`.
- A run without a provenance record is not a result.
- The registry is append-only. Correcting a row means adding a row.
- No new framework. Reasons, one each: MLflow adds a server and a database for a
  six-person team; DVC would fight the git-tracked released splits; Hydra solves config
  composition, which is not the problem here. The runner is a few hundred lines the team
  can read.
- The runner calls the existing component entry points (`src.canonical.gauge.build`,
  `src.study.split`, `src.graph.builder_new`, `src.models.run`) with paths it chose. It
  does not copy or fork them.

## 6. Outputs

Run folder:

```text
experiments/<YYYY-MM-DD>_<HHMM>_<label>/
  config/            the experiment file and a copy of each of the five files it named,
                     plus each one's effective version with every default filled in
                     (e.g. standard.yaml, standard.effective.yaml)
  provenance.json    per-component inputs, outputs, fingerprints, code revision, versions
  report.md          status; then the model report (RMSE, MAE, correlation per fold)
  log.txt            everything the components printed
  canonical/         gauge_hourly.csv, stations.csv, days.csv, report.md  (or a cache pointer)
  split/             <name>.json, summary, plot
  graphs/            fold-<i>/{train,validation,test}.pt, report.md
  model/             history.csv, best.pt, predictions.csv
```

`experiments/registry.csv` columns are listed in FR04. `provenance.json` fields are listed
in FR03. Read by: every deliverable (B, C, D report from run folders), and by `diff`.

## 7. Testing

| Test | Checks | Subtask |
|---|---|---|
| TEST01 | The golden run, run twice, gives byte-identical `predictions.csv` and the same fingerprints in `provenance.json`. | D07.5 |
| TEST02 | Change one key in one file under `config/` and run again: a new folder; the registry gains one row; that component's effective copy differs in exactly that key; exactly one component's input fingerprint differs. | D07.2 |
| TEST03 | Running into an existing folder is refused with a clear message. | D07.2 |
| TEST04 | A build into a directory that already holds `stations.csv` is refused: the message names the directory and the file, the exit code is non-zero, and the file is byte-identical afterwards. | D07.6 |
| TEST05 | Changing the gauge bounds and rerunning the fetch fetches the station list again (offline, using the fake client in `tests/test_ingestion_gauge.py`). | D07.6 |
| TEST06 | Every raw file `provenance.json` lists exists and its fingerprint matches the index row. | D07.2 |
| TEST07 | The synthetic pipeline fixture (`tests/data/pipeline/make_fixture.py`) runs end to end through the runner with the same 864 prediction rows as `tests/test_pipeline_integration.py`. | D07.1, D07.2 |
| TEST08 | The suite fails when coverage drops below either floor — shown once by running with one test file excluded. `src/experiments/run.py` itself meets the 90 % floor. | D07.7 |

## 8. Documentation requirements

- `docs/experiments/README.md`: how to write an experiment file (five paths and a
  label), which component files to edit, how to run, how to find a run by date, how to
  read `provenance.json`, and (later) how to use `list`, `diff` and overrides.
- `docs/graphs/README.md`: replace the five hand-run commands with the runner command,
  keeping the old commands in a "component by component" section.

## 9. Dependencies and blockers

No scientific decision blocks this deliverable. Issues it addresses: ISS-24 (the per-source
manifest; the per-run provenance record is its counterpart, not a replacement), ISS-28 (02 overwrite), ISS-29
(fetch stale state), ISS-32 (fixed fold count and no region concept). D07.5 needs one
Sydney fetch (about 90 minutes) with the box D08 names in its protocol.

## 10. Current technical findings

What each component records today, so the reader knows what the provenance record adds:

| Component | Where output goes | Second run with different settings | What it records about its inputs |
|---|---|---|---|
| 01 fetch | `database/raw/` | resumes; index rewritten whole | per-file SHA-256 in `index/<fetcher>.csv`; config *path* in the report |
| 02 build | `database/standard/` | **overwrites silently** | config path and a short git revision in `report.md`; no input hashes |
| 02 grids | `database/standard/` | refuses | SHA-256 of its input files in `grid_report.md` |
| 03 split | `config/splits/<name>.json` (git-tracked) | refuses if different | `stations.csv` *path*, seed, counts; not the clustering settings |
| 04 graphs | `database/graphs/<name>/` | refuses | SHA-256 of config, split, seeds, tables — as text in `report.md` |
| 06 model | `experiments/<method>_<time>/` | new folder | effective settings, seed, versions, full git revision, dirty flag, fingerprint (SHA-256) of all 19 inputs, in `config.yaml` |

So the chain of custody is complete from the model back to the graphs, and from the
graphs back to the tables, but breaks at 02 → 01: nothing says which raw files produced a
given `gauge_hourly.csv`.

## 11. This week and backlog

**This week:** D07.1, D07.2, D07.6, D07.7. D07.5 if the Sydney fetch runs.
**Backlog (week 7+):** D07.3, D07.4, FR11 (overrides).

Glossary: *component, run, fingerprint, provenance record, manifest, registry, component cache, region* in [GLOSSARY.md](GLOSSARY.md).
