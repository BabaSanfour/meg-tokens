#!/bin/bash
# Restartable, repetition-indexed recovery array. Each task writes only
# ssmparameterrecoveryrepNNN/ssmmodelrecoveryrepNNN and is independent of
# subject-level fit outputs. Submit, for example:
#   sbatch --array=0-11%4 scripts/thura2012_recovery_array.sh tokens.toml 12

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-recovery
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thura12-recovery-%A_%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-11%4 scripts/thura2012_recovery_array.sh <config.toml> [repetitions]}"
REPETITIONS="${2:-12}"
if [ "$REPETITIONS" -lt 8 ]; then
    echo "recovery requires at least 8 truth repetitions" >&2
    exit 2
fi
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} repetition=${SLURM_ARRAY_TASK_ID}/${REPETITIONS}"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') python=$(./.venv/bin/python --version 2>&1)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-recovery \
    --repetitions "$REPETITIONS" \
    --repetition-index "$SLURM_ARRAY_TASK_ID" \
    --n-starts 2
echo "finished=$(date --iso-8601=seconds)"
