#!/bin/bash
# Slurm array for mean deep-source band power per condition and decision-time
# quintile on fsaverage's grid: one task per subject, stream and band (every
# lock), 320 for the cohort. Never run this command directly on a login node.
#
#   sbatch --array=0-319%100 scripts/deep_power_means_array.sh tokens.toml <deep_root>
#
# Task i takes subject i // 10, stream (i % 10) // 5 (dss, then ica) and band
# i % 5. It reads the subject's eight deep-power run files of that band
# (scripts/deep_power_array.sh), morphs each mean with the subject's deep
# morph (scripts/deep_morph_array.sh), and writes one file per lock to the
# node's local disk, then into place under a temporary name.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=deep-power-means
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --tmp=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/deep-power-means-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/deep_power_means_array.sh <config.toml> <deep_root>}"
DEEP_ROOT="${2:?usage: sbatch --array=0-N scripts/deep_power_means_array.sh <config.toml> <deep_root>}"
STREAMS=(dss ica)
BANDS=(theta alpha beta gamma highgamma)

mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
INDEX=$((SLURM_ARRAY_TASK_ID / 10))
if [ "$INDEX" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects x 2 streams x 5 bands"
    exit 0
fi
SUBJECT="${SUBJECTS[$INDEX]}"
STREAM="${STREAMS[$(((SLURM_ARRAY_TASK_ID % 10) / 5))]}"
BAND="${BANDS[$((SLURM_ARRAY_TASK_ID % 5))]}"
LOCAL="${SLURM_TMPDIR}/means"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} stream=${STREAM} band=${BAND}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

# GNU time logs the task's peak resident memory; sacct's MaxRSS counts page cache.
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" meg deep-power-means \
    --subjects "$SUBJECT" \
    --streams "$STREAM" \
    --bands "$BAND" \
    --deep-root "$DEEP_ROOT" \
    --output-root "$LOCAL"
echo "computed=$(date --iso-8601=seconds)"

(cd "$LOCAL" && find derivatives -type f) | while read -r file; do
    mkdir -p "$DEEP_ROOT/$(dirname "$file")"
    cp "$LOCAL/$file" "$DEEP_ROOT/$file.partial"
    mv "$DEEP_ROOT/$file.partial" "$DEEP_ROOT/$file"
done
echo "finished=$(date --iso-8601=seconds)"
