#!/usr/bin/env bash
# Submit the Gadi data chain from the login node: grid tables, then graphs, then a
# one-epoch G+R+S timing run. Each job starts only if the one before it succeeded.
#   bash scripts/nci/submit_pipeline.sh            # all three
#   bash scripts/nci/submit_pipeline.sh --no-timing
# Full training runs are submitted by hand once the timing run says how long they take
# (docs/nci/README.md, section 6).
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
timing=true
if [[ ${1:-} == --no-timing ]]; then
    timing=false
    shift
fi
if [[ $# -gt 0 ]]; then
    printf 'Usage: bash scripts/nci/submit_pipeline.sh [--no-timing]\n' >&2
    exit 2
fi
if [[ -n ${PBS_JOBID:-} ]]; then
    printf 'BLOCKED: run this on a Gadi login node, not inside a job.\n' >&2
    exit 1
fi
if ! command -v qsub > /dev/null; then
    printf 'BLOCKED: qsub not found; run this on a Gadi login node.\n' >&2
    exit 1
fi
for file in .venv/bin/python database/standard/gauge_hourly.csv database/standard/stations.csv; do
    if [[ ! -e $file ]]; then
        printf 'BLOCKED: missing %s (docs/nci/README.md, section 6).\n' "$file" >&2
        exit 1
    fi
done
.venv/bin/python -B scripts/nci/pipeline_config.py check-grids config/canonical/grids.yaml

grids=$(qsub scripts/nci/build_grids.pbs)
printf 'grids:  %s\n' "$grids"
graphs=$(qsub -W "depend=afterok:$grids" scripts/nci/build_graphs.pbs)
printf 'graphs: %s (after grids)\n' "$graphs"
if [[ $timing == true ]]; then
    output="experiments/nci/d09-grs-timing-$(date -u +%Y%m%d-%H%M%S)"
    timed=$(qsub -W "depend=afterok:$graphs" -l walltime=04:00:00 \
        -v "NCI_MODEL_CONFIG=config/models/d09_round_robin_grs.yaml,NCI_OUTPUT_DIR=$output,NCI_EPOCHS=1" \
        scripts/nci/train.pbs)
    printf 'timing: %s (after graphs) -> %s\n' "$timed" "$output"
fi
printf 'Watch with: qstat -u %s\n' "$USER"
