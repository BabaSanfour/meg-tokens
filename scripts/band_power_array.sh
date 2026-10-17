#!/bin/bash
# Slurm array for single-trial band power: one task per run, stream and band
# (every lock), 2,560 for the cohort. Never run this command directly on a
# login node.
#
#   sbatch --array=0-2559%400 scripts/band_power_array.sh tokens.toml [fsaverage_root]
#
# Run scripts/band_power_prepare.sh first (this array with
# --dependency=afterok on it): every task reads the morph maps it builds.
# Task i takes line i of `meg band-power-tasks`. Each task writes only its own
# part files; requeue a failed task with the same index. Assemble afterwards
# with scripts/band_power_merge_array.sh (submitted with --dependency=afterok
# on this array).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/band-power-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/band_power_array.sh <config.toml> [fsaverage_root]}"
FSAVERAGE_ROOT="${2:-}"

mapfile -t TASKS < <(./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-tasks)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#TASKS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#TASKS[@]} tasks"
    exit 0
fi
read -r SUBJECT RUN CONDITION STREAM BAND <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} run=${RUN} condition=${CONDITION} stream=${STREAM} band=${BAND}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

./.venv/bin/meg-tokens --config "$CONFIG" meg band-power \
    --subjects "$SUBJECT" \
    --run "$RUN" \
    --condition "$CONDITION" \
    --stream "$STREAM" \
    --band "$BAND" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --fsaverage-root "$FSAVERAGE_ROOT"
echo "finished=$(date --iso-8601=seconds)"
