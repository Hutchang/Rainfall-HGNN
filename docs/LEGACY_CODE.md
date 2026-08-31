# Inherited and transitional code

The integration branch contains both the Greater Sydney transitional baseline and
restored inherited reference code while the component architecture in
`docs/requirements/` is implemented. Presence in the tree does not mean a module is a
supported project capability.

## Transitional entry points retained

| Path | Status | Use |
|---|---|---|
| `train_fused.py` | validated transitional runner | Exercises the 2022 Sydney gauge-only smoke path. New model work belongs behind component 06's adapter. |
| `train_homo.py` | restored inherited reference | Potential homogeneous-model ideas only; not baseline-validated or a supported runner. |
| `train_st.py` | restored inherited reference | Potential temporal-model ideas only; outside the current gauge milestone. |
| `config.yaml` | transitional 2022 smoke profile | Read by inherited modules at import time. New components use scoped configs under `config/`. |

The canonical current science contract is not encoded by these scripts. It is defined
by the component requirements and will use regenerated 2021–2022 BoM data.

## Removed from the integration root

The following inherited files were removed because the audit proved them broken,
orphaned, misleading, machine-specific, or notebook-only:

- all root Jupyter notebooks;
- `analyze.py` (hard-coded path to another user's Optuna database);
- `paramtuning.py` (invalid objective/directions contract);
- inherited multi-source tuning prototypes;
- `run_idw.py` (non-comparable and crashes on missing output directory);
- `run_search.sh` and `run_train.sh` (NUS-specific SLURM paths/environment);
- `train.py` (broken precursor against the current training signature);
- `visualize_error.py` (expects prediction files no code produces).

They remain recoverable from Git history and the preserved inherited-source branch.

## Restored reference areas awaiting classification

The rollback restored `benchmarks/`, `src/radar/`, `src/visualization/`, `src/cml/`,
additional graph modules, and experimental models because several files mix generic
algorithms with regional assumptions. They are retained temporarily for a conservative,
file-by-file review; they are not supported entry points.

- Keep or extract genuinely generic spatial, temporal, model, and diagnostic code.
- Refactor reusable code only after its coordinate, unit, time, and product assumptions
  are explicit and tested for the Australian component contract.
- Remove Singapore/NEA and commercial-microwave-link implementations after confirming
  that no reusable gauge/radar/satellite logic would be lost.
- Do not treat the inherited radar path as Australian radar support. Australian radar
  and satellite work starts from the validated NCI products in Components 01 and 02.

## Source preservation

- `jylow-clone` at `1c2489a` preserves one imported source tree outside the active scope.
- tag `niramay-inherited` at `8368cbf` preserves the inherited Niramay tree.
- `niramay-clone` preserves the inherited Greater Sydney source tree.

Do not copy a deleted entry point back into the active root to start new work. Extract
the useful function behind the relevant component contract and protect it with that
component's tests.
