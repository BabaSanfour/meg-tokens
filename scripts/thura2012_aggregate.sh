#!/bin/bash
# Aggregate completed subject fits before evaluation. This is intentionally a
# separate restartable Slurm step so a partial array never masquerades as a
# complete group derivative.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thura12-aggregate-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/thura2012_aggregate.sh <config.toml> }"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') python=$(./.venv/bin/python --version 2>&1)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-aggregate
echo "finished=$(date --iso-8601=seconds)"
