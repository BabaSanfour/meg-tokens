#!/bin/bash
# Slurm array assembling band power: one task per subject and stream, 64 for
# the cohort, after scripts/band_power_array.sh. Never run this command
# directly on a login node.
#
#   sbatch --dependency=afterok:<band-power array id> --array=0-63 \
#       scripts/band_power_merge_array.sh tokens.toml
#
# Task i assembles subject i // 2 in stream i % 2 (dss, then ica) and, once its
# files are written, removes that subject and stream's region parts. The
# fsaverage parts are the final fsaverage files and stay.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power-merge
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/band-power-merge-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/band_power_merge_array.sh <config.toml>}"
STREAMS=(dss ica)

mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
INDEX=$((SLURM_ARRAY_TASK_ID / ${#STREAMS[@]}))
if [ "$INDEX" -ge "${#SUBJECTS[@]}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} outside ${#SUBJECTS[@]} subjects x ${#STREAMS[@]} streams"
    exit 0
fi
SUBJECT="${SUBJECTS[$INDEX]}"
STREAM="${STREAMS[$((SLURM_ARRAY_TASK_ID % ${#STREAMS[@]}))]}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} stream=${STREAM}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-merge \
    --subjects "$SUBJECT" \
    --streams "$STREAM"

DERIVATIVES="$(./.venv/bin/python -c "from meg_tokens.core import ProjectConfig; print(ProjectConfig.from_toml('$CONFIG').bids_root / 'derivatives')")"
rm -f "$DERIVATIVES/sub-${SUBJECT}/meg/sub-${SUBJECT}"_task-tokens_run-*_proc-"${STREAM}"_space-HCPMMP1_desc-*_power.h5
echo "finished=$(date --iso-8601=seconds)"
