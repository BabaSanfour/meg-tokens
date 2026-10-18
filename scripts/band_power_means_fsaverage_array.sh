#!/bin/bash
# Slurm array for mean fsaverage band power per condition and decision-time
# quintile: one task per subject, stream and band (every lock), 320 for the
# cohort. Never run this command directly on a login node.
#
#   sbatch --array=0-319%100 scripts/band_power_means_fsaverage_array.sh tokens.toml <fsaverage_root>
#
# Task i takes subject i // 10, stream (i % 10) // 5 (dss, then ica) and band
# i % 5. It reads the subject's eight fsaverage run files of that band (about
# 75 GB, which sets its time), keeps the sums in memory (about 9 GB), and
# writes one file per lock (about 1-5 GB) to the node's local disk, then into
# place under a temporary name. Needs the region trial tables
# (scripts/band_power_merge_array.sh).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power-means-fsaverage
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --tmp=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/band-power-means-fsaverage-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/band_power_means_fsaverage_array.sh <config.toml> <fsaverage_root>}"
FSAVERAGE_ROOT="${2:?usage: sbatch --array=0-N scripts/band_power_means_fsaverage_array.sh <config.toml> <fsaverage_root>}"
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
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-means-fsaverage \
    --subjects "$SUBJECT" \
    --streams "$STREAM" \
    --bands "$BAND" \
    --fsaverage-root "$FSAVERAGE_ROOT" \
    --output-root "$LOCAL"
echo "computed=$(date --iso-8601=seconds)"

(cd "$LOCAL" && find derivatives -type f) | while read -r file; do
    mkdir -p "$FSAVERAGE_ROOT/$(dirname "$file")"
    cp "$LOCAL/$file" "$FSAVERAGE_ROOT/$file.partial"
    mv "$FSAVERAGE_ROOT/$file.partial" "$FSAVERAGE_ROOT/$file"
done
echo "finished=$(date --iso-8601=seconds)"
