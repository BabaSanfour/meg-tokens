#!/bin/bash
# Slurm array for each subject's magnetometer spectra (meg.md §2, Bands): one
# task per subject and stream, 64 for the cohort. Never run this command
# directly on a login node.
#
#   sbatch --array=0 scripts/sensor_spectra_array.sh tokens.toml            # canary
#   sbatch --array=1-63 --dependency=afterok:<canary> scripts/sensor_spectra_array.sh tokens.toml
#
# Task i takes subject i // 2 and stream i % 2 (dss, then ica). It reads the
# subject's eight press-locked epoch files and writes one small file; size the
# array from the canary's peak (GNU time below). Needs the epochs. The band
# choice follows (scripts/band_choice.sh).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=sensor-spectra
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/sensor-spectra-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-63 scripts/sensor_spectra_array.sh <config.toml> [meg sensor-spectra options]}"
STREAMS=(dss ica)
EXTRA=("${@:2}")
mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
INDEX=$((SLURM_ARRAY_TASK_ID / 2))
if [ "$INDEX" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects x 2 streams"
    exit 0
fi
SUBJECT="${SUBJECTS[$INDEX]}"
STREAM="${STREAMS[$((SLURM_ARRAY_TASK_ID % 2))]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} stream=${STREAM}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

# GNU time logs the task's peak resident memory; sacct's MaxRSS counts page cache.
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" meg sensor-spectra \
    --subjects "$SUBJECT" --streams "$STREAM" "${EXTRA[@]}"
echo "finished=$(date --iso-8601=seconds)"
