#!/bin/bash
# Slurm array for the event-related fields: one task per subject and stream,
# 64 for the cohort. Never run this command directly on a login node.
#
#   sbatch --array=0-63 scripts/erp_array.sh tokens.toml
#
# Task i takes subject i // 2 in stream i % 2 (dss, then ica). Needs the morph
# maps of scripts/band_power_prepare.sh.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=erp
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/erp-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/erp_array.sh <config.toml>}"
STREAMS=(dss ica)

mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
INDEX=$((SLURM_ARRAY_TASK_ID / ${#STREAMS[@]}))
if [ "$INDEX" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects x ${#STREAMS[@]} streams"
    exit 0
fi
SUBJECT="${SUBJECTS[$INDEX]}"
STREAM="${STREAMS[$((SLURM_ARRAY_TASK_ID % ${#STREAMS[@]}))]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} stream=${STREAM}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

./.venv/bin/meg-tokens --config "$CONFIG" meg erp \
    --subjects "$SUBJECT" \
    --streams "$STREAM"
echo "finished=$(date --iso-8601=seconds)"
