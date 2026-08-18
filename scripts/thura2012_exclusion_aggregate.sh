#!/bin/bash
# Merge the subject × exclusion-rule sensitivity refits after the array.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-excl-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thura12-excl-merge-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/thura2012_exclusion_aggregate.sh <config.toml>}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-exclusion-robustness-aggregate
echo "finished=$(date --iso-8601=seconds)"
