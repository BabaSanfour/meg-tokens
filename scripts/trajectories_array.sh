#!/bin/bash
# Slurm array for the PCA trajectories (meg.md §2): one task per analysis,
# stream and lock, 60 for the plan. Never run this command directly on a login
# node.
#
#   sbatch --array=<tasks> [resources] scripts/trajectories_array.sh tokens.toml [analyze trajectories options]
#
# Options after the config go to every task of the submission: the
# preprint-matched fit is the press lock cut to -1.25 to +1.2 s (tasks 2 and
# 5: `--window -1.25 1.2`), `--with-deep` fits the deep structures beside the
# regions, and band power on fsaverage (tasks 42-59) needs `--fsaverage-root`
# (scripts/band_power_means_fsaverage_array.sh).
#
# Task i runs analysis i // 6 of ANALYSES on stream (i % 6) // 3 and lock
# i % 3 (0 go, 1 decision, 2 enter). Memory follows the analysis and the lock:
# GNU time peaks of the float64 runs on the cohort, halved for float32 (power
# 2.6 GB go, about 6 GB the longer locks; region fields under 1.5 GB;
# fsaverage fields about 13 GB go, about 30 GB the longer locks); each task
# logs its own peak. fir allows about 24 GB per core, so the
# longer fsaverage locks take two, which the covariance's BLAS uses:
#
#   power, go               sbatch --array=0,3,6,9,12,15 --mem=3300M
#   power, decision/enter   sbatch --array=1,2,4,5,7,8,10,11,13,14,16,17 --mem=7680M
#   region fields, go       sbatch --array=18,21,24,27 --mem=1G
#   region fields, longer   sbatch --array=19,20,22,23,25,26,28,29 --mem=2G
#   fsaverage, go           sbatch --array=30,33,36,39 --mem=16G
#   fsaverage, longer       sbatch --array=31,32,34,35,37,38,40,41 --mem=37G --cpus-per-task=2
#   fsaverage power         tasks 42-59: size them from a canary
#
# each followed by `scripts/trajectories_array.sh tokens.toml`. The power tasks
# spend most of their time reading the means files, so their CPU use stays low
# whatever they are given. Needs the band power means
# (scripts/band_power_means_array.sh) and the event-related fields
# (scripts/erp_array.sh).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=trajectories
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3300M
#SBATCH --time=03:00:00
#SBATCH --output=logs/trajectories-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N scripts/trajectories_array.sh <config.toml> [analyze trajectories options]}"
OPTIONS=("${@:2}")
# fit measure space
ANALYSES=(
    "grandaverage power HCPMMP1"
    "pooled log10 HCPMMP1"
    "pooled db HCPMMP1"
    "grandaverage erp HCPMMP1"
    "pooled erp HCPMMP1"
    "grandaverage erp fsaverage"
    "pooled erp fsaverage"
    "grandaverage power fsaverage"
    "pooled log10 fsaverage"
    "pooled db fsaverage"
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
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} fit=${FIT} measure=${MEASURE} space=${SPACE} stream=${STREAM} lock=${LOCK} options=${OPTIONS[*]:-}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') host=$(hostname)"

# GNU time logs the task's peak resident memory; sacct's MaxRSS counts page cache.
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" analyze trajectories \
    --fit "$FIT" --measure "$MEASURE" --space "$SPACE" \
    --streams "$STREAM" --locks "$LOCK" "${OPTIONS[@]}"
echo "finished=$(date --iso-8601=seconds)"
