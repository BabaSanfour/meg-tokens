#!/bin/bash
# Merge subject-scoped held-out/matched/distribution diagnostics after the
# evaluation array. This validates that every selected subject produced all
# required tables before writing group derivatives.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-eval-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thura12-eval-merge-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/thura2012_evaluate_aggregate.sh <config.toml>}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') python=$(./.venv/bin/python --version 2>&1)"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-evaluate-aggregate --folds 3
echo "finished=$(date --iso-8601=seconds)"
