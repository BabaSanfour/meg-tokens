#!/bin/bash
# Slurm array for mean band power per condition and decision-time quintile:
# one task per subject and stream, 64 for the cohort. Never run this command
# directly on a login node.
#
#   sbatch --array=0-63 scripts/band_power_means_array.sh tokens.toml
#
# Task i takes subject i // 2 in stream i % 2 (dss, then ica). Needs the
# assembled band power of scripts/band_power_merge_array.sh.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power-means
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/band-power-means-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/band_power_means_array.sh <config.toml>}"
STREAMS=(dss ica)

# Further options go to the command: `--induced` for the second copy.
EXTRA=("${@:2}")
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

./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-means \
    --subjects "$SUBJECT" \
    --streams "$STREAM" \
    "${EXTRA[@]}"
echo "finished=$(date --iso-8601=seconds)"
