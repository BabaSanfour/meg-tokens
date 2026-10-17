#!/bin/bash
# Build every subject's morph map with fsaverage, one after another, before
# scripts/band_power_array.sh: MNE writes a missing map on first use, and the
# array's tasks would write and read the same map at once. Never run this
# command directly on a login node.
#
#   sbatch scripts/band_power_prepare.sh tokens.toml

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-power-prepare
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/band-power-prepare-%j.out

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/band_power_prepare.sh <config.toml>}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} commit=$(git rev-parse HEAD)"
./.venv/bin/meg-tokens --config "$CONFIG" meg band-power-prepare
echo "finished=$(date --iso-8601=seconds)"
