#!/bin/bash
# Slurm array for single-trial band power at every deep source: one task per
# run, stream and band (every lock), 2,560 for the cohort, the tasks of
# `meg band-power-tasks`. Never run this command directly on a login node.
#
#   sbatch --array=0-2559%400 scripts/deep_power_array.sh tokens.toml <deep_root>
#
# <deep_root> is the BIDS-like root in /project that also holds the fsaverage
# power. A task writes its file to the node's local disk ($SLURM_TMPDIR), then
# copies it into place under a temporary name renamed at the end, so a file
# on the shared filesystem is whole or absent. Requeue a failed task with the
# same index.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=deep-power
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --tmp=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/deep-power-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/deep_power_array.sh <config.toml> <deep_root>}"
DEEP_ROOT="${2:?usage: sbatch --array=0-N scripts/deep_power_array.sh <config.toml> <deep_root>}"

mapfile -t TASKS < <(./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-tasks)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#TASKS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#TASKS[@]} tasks"
    exit 0
fi
read -r SUBJECT RUN CONDITION STREAM BAND <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
LOCAL="${SLURM_TMPDIR}/deep-power"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} run=${RUN} condition=${CONDITION} stream=${STREAM} band=${BAND}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

# GNU time logs the task's peak resident memory; sacct's MaxRSS counts page cache.
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" meg deep-power \
    --subjects "$SUBJECT" \
    --run "$RUN" \
    --condition "$CONDITION" \
    --stream "$STREAM" \
    --band "$BAND" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --output-root "$LOCAL"
echo "computed=$(date --iso-8601=seconds)"

(cd "$LOCAL" && find derivatives -type f) | while read -r file; do
    mkdir -p "$DEEP_ROOT/$(dirname "$file")"
    cp "$LOCAL/$file" "$DEEP_ROOT/$file.partial"
    mv "$DEEP_ROOT/$file.partial" "$DEEP_ROOT/$file"
done
echo "finished=$(date --iso-8601=seconds)"
