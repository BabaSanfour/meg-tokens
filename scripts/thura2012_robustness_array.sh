#!/bin/bash
# Restartable subject × configuration robustness array. Six configurations
# and one task per selected subject keep each task well below the 12-hour
# limit and make failed cells independently resubmittable.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-robust-array
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/thura12-robust-%A_%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-191%8 scripts/thura2012_robustness_array.sh <config.toml> [n_starts]}"
N_STARTS="${2:-1}"
mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
CONFIGURATIONS=(baseline tau_100ms tau_300ms solver_20ms post_horizon_evidence_zero expanded_bounds)
SUBJECT_INDEX=$((SLURM_ARRAY_TASK_ID / ${#CONFIGURATIONS[@]}))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % ${#CONFIGURATIONS[@]}))
if [ "$SUBJECT_INDEX" -ge "${#SUBJECTS[@]}" ]; then exit 0; fi
SUBJECT="${SUBJECTS[$SUBJECT_INDEX]}"
ROBUSTNESS_CONFIGURATION="${CONFIGURATIONS[$CONFIG_INDEX]}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT} configuration=${ROBUSTNESS_CONFIGURATION}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') python=$(./.venv/bin/python --version 2>&1)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-robustness \
    --subjects "$SUBJECT" \
    --configuration "$ROBUSTNESS_CONFIGURATION" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --n-starts "$N_STARTS"
echo "finished=$(date --iso-8601=seconds)"
