#!/bin/bash
# Restartable Slurm array for the complete four-model Thura et al. (2012) fit.
# Never run this command directly on a login node.
#
#   sbatch --array=0-31%8 scripts/thura2012_fit_array.sh tokens.toml 3
#
# Each array task owns one subject's output paths. Requeue a failed task with
# the same array index; no task writes another subject's files.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-fit
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/thura12-fit-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/thura2012_fit_array.sh <config.toml> [n_starts]}"
N_STARTS="${2:-3}"
if [ "$N_STARTS" -lt 3 ]; then
    echo "production mechanistic fit array requires at least three optimizer starts" >&2
    exit 2
fi

mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects"
    exit 0
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname) python=$(./.venv/bin/python --version 2>&1)"
echo "command=meg-tokens --config ${CONFIG} behavior ssm-fit --subjects ${SUBJECT} --model-set mechanistic --n-starts ${N_STARTS} --n-jobs ${SLURM_CPUS_PER_TASK}"

./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-fit \
    --subjects "$SUBJECT" \
    --model-set mechanistic \
    --n-starts "$N_STARTS" \
    --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "finished=$(date --iso-8601=seconds)"
