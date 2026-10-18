#!/bin/bash
# Slurm array for each subject's morph of its deep sources onto fsaverage's
# grid of the same structures (MNE's volume morph, affine then SDR through
# dipy): one task per subject, 32 for the cohort. Never run this command
# directly on a login node.
#
#   sbatch --array=0-31 scripts/deep_morph_array.sh tokens.toml

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=deep-morph
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/deep-morph-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/deep_morph_array.sh <config.toml>}"
mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects"
    exit 0
fi
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"
# GNU time logs the task's peak resident memory; sacct's MaxRSS counts page cache.
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" meg deep-morph --subjects "$SUBJECT"
echo "finished=$(date --iso-8601=seconds)"
