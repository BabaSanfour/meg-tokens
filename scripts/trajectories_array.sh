#!/bin/bash
# Slurm array for the PCA trajectories (meg.md §2): one task per analysis,
# stream and lock, 42 for the plan. Never run this command directly on a login
# node.
#
#   sbatch --array=0-29 scripts/trajectories_array.sh tokens.toml
#   sbatch --array=30-41 --mem=96G scripts/trajectories_array.sh tokens.toml
#
# Task i runs analysis i // 6 of ANALYSES on stream (i % 6) // 3 and lock
# i % 3. The fsaverage fields (tasks 30-41) hold every subject's 12 x 2,101 x
# 8,196 means at once, about 53 GB, hence their own memory. Needs the band
# power means (scripts/band_power_means_array.sh) and the event-related fields
# (scripts/erp_array.sh).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=trajectories
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/trajectories-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/trajectories_array.sh <config.toml>}"
# fit measure space
ANALYSES=(
    "grandaverage power HCPMMP1"
    "pooled log10 HCPMMP1"
    "pooled db HCPMMP1"
    "grandaverage erp HCPMMP1"
    "pooled erp HCPMMP1"
    "grandaverage erp fsaverage"
    "pooled erp fsaverage"
)
STREAMS=(dss ica)
LOCKS=(go decision enter)

ANALYSIS=$((SLURM_ARRAY_TASK_ID / 6))
if [ "$ANALYSIS" -ge "${#ANALYSES[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#ANALYSES[@]} analyses x ${#STREAMS[@]} streams x ${#LOCKS[@]} locks"
    exit 0
fi
read -r FIT MEASURE SPACE <<< "${ANALYSES[$ANALYSIS]}"
STREAM="${STREAMS[$(((SLURM_ARRAY_TASK_ID % 6) / 3))]}"
LOCK="${LOCKS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} fit=${FIT} measure=${MEASURE} space=${SPACE} stream=${STREAM} lock=${LOCK}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

./.venv/bin/meg-tokens --config "$CONFIG" analyze trajectories \
    --fit "$FIT" --measure "$MEASURE" --space "$SPACE" \
    --streams "$STREAM" --locks "$LOCK"
echo "finished=$(date --iso-8601=seconds)"
