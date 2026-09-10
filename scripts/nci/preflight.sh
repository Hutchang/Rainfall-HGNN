#!/usr/bin/env bash
# Read-only checks; never installs packages, creates data, or submits jobs.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"

config_file=config/models/hgnn.yaml
check_cuda=false
if [[ ${1:-} == --cuda ]]; then
    check_cuda=true
    shift
fi
if [[ $# -gt 0 ]]; then
    config_file=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    printf 'Usage: bash scripts/nci/preflight.sh [--cuda] [model-config.yaml]\n' >&2
    exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
    printf 'BLOCKED: missing .venv/bin/python. Prepare the locked Linux environment first; see docs/nci/README.md.\n' >&2
    exit 1
fi
if [[ $check_cuda == true && -z ${PBS_JOBID:-} ]]; then
    printf 'BLOCKED: --cuda requires a PBS allocation. Run the ordinary preflight on the login node.\n' >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
if [[ $check_cuda == true ]]; then
    .venv/bin/python -B scripts/nci/check_environment.py --cuda
else
    .venv/bin/python -B scripts/nci/check_environment.py
fi
if [[ ! -f src/models/run.py ]]; then
    printf 'BLOCKED: Component 06 runner src/models/run.py is missing from this checkout.\n' >&2
    exit 1
fi
# Structural/data validation works on the login CPU even if the YAML selects CUDA.
.venv/bin/python -B -m src.models.run "$config_file" --check-inputs --device cpu
printf 'Input and environment preflight passed. No training or job submission was performed.\n'
