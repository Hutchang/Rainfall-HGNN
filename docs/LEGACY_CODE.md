# Inherited and transitional code

The integration branch contains inherited Jylow research code while the component
architecture in `docs/requirements/` is implemented. This page distinguishes usable
transitional code from deleted stale entry points.

## Transitional entry points retained

| Path | Status | Use |
|---|---|---|
| `train_fused.py` | validated transitional runner | Reproduces the documented 2022 gauge-only Jylow smoke path. New model work belongs behind component 06's adapter. |
| `train_homo.py` | inherited research variant, not baseline-validated | Reference only for the homogeneous ablation. Do not use for a baseline claim. |
| `train_st.py` | inherited spatio-temporal prototype, not baseline-validated | Reference only for later temporal requirements. It is outside the present baseline. |
| `config.yaml` | transitional 2022 smoke profile | Read by inherited modules at import time. New components use scoped configs under `config/`. |

The canonical current science contract is not encoded by these scripts. It is defined
by the component requirements and will use regenerated 2021–2022 BoM data.

## Removed from the integration root

The following inherited files were removed because the audit proved them broken,
orphaned, misleading, machine-specific, or notebook-only:

- all root Jupyter notebooks;
- `analyze.py` (hard-coded path to another user's Optuna database);
- `paramtuning.py` (invalid objective/directions contract);
- `paramtuning_gauge.py` (name says gauge-only but runs gauge+CML, with inert settings);
- `paramtuning_gauge_cml.py` (import-time training/search prototype);
- `run_idw.py` (non-comparable and crashes on missing output directory);
- `run_search.sh` and `run_train.sh` (NUS-specific SLURM paths/environment);
- `train.py` (broken precursor against the current training signature);
- `visualize_error.py` (expects prediction files no code produces).

They remain recoverable from Git history and the preserved `jylow-clone` branch. Their
audit evidence is in `docs/baseline-reconciliation/AUDIT_JYLOW.md` sections 3 and 6.

## Source preservation

- `jylow-clone` at `1c2489a` preserves the imported Jylow tree.
- tag `niramay-inherited` at `8368cbf` preserves the inherited Niramay tree.
- `niramay-clone` contains the isolated compatibility changes documented by the audit.

Do not copy a deleted entry point back into the active root to start new work. Extract
the useful function behind the relevant component contract and protect it with that
component's tests.

