#!/bin/bash
# Validate and merge all repetition-scoped recovery derivatives after the
# repetition array. No fitting occurs in this merge job.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-recovery-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thura12-recovery-merge-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/thura2012_recovery_aggregate.sh <config.toml> [repetitions]}"
REPETITIONS="${2:-12}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-recovery-aggregate \
    --repetitions "$REPETITIONS"
echo "finished=$(date --iso-8601=seconds)"
