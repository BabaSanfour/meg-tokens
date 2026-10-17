#!/bin/bash
# Slurm array for single-trial band power: one task per run, stream and band
# (every lock), 2,560 for the cohort. Never run this command directly on a
# login node.
#
#   sbatch --dependency=afterok:<prepare job> --array=0-2559%400 \
#       scripts/band_power_array.sh tokens.toml [fsaverage_root]
#
# Run scripts/band_power_prepare.sh first: every task reads the morph maps it
# builds. Task i takes line i of `meg band-power-tasks`. A task writes its
# files to the node's local disk ($SLURM_TMPDIR), where HDF5's many small
# writes are cheap, then copies each finished file to its place in one
# stream, under a temporary name renamed at the end, so a file on the shared
# filesystems is whole or absent. Requeue a failed task with the same index.
# Assemble afterwards with scripts/band_power_merge_array.sh.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --tmp=20G
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
BIDS_ROOT="$(./.venv/bin/python -c "from meg_tokens.core import ProjectConfig; print(ProjectConfig.from_toml('$CONFIG').bids_root)")"
LOCAL="${SLURM_TMPDIR}/band-power"
LOCAL_REGIONS="${LOCAL}/regions"
LOCAL_FSAVERAGE=""
if [ -n "$FSAVERAGE_ROOT" ]; then
    LOCAL_FSAVERAGE="${LOCAL}/fsaverage"
fi

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
    --output-root "$LOCAL_REGIONS" \
    --fsaverage-root "$LOCAL_FSAVERAGE"
echo "computed=$(date --iso-8601=seconds)"

# Copy each file into place under a temporary name, then rename it.
publish() {
    local from="$1" to="$2"
    (cd "$from" && find derivatives -type f) | while read -r file; do
        mkdir -p "$to/$(dirname "$file")"
        cp "$from/$file" "$to/$file.partial"
        mv "$to/$file.partial" "$to/$file"
    done
}
publish "$LOCAL_REGIONS" "$BIDS_ROOT"
if [ -n "$FSAVERAGE_ROOT" ]; then
    publish "$LOCAL_FSAVERAGE" "$FSAVERAGE_ROOT"
fi
echo "finished=$(date --iso-8601=seconds)"
