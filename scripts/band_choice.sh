#!/bin/bash
# The cohort's band choice from spectral peaks (the preprint's Fig S1, and its
# checks), both streams, then its figures. One job, after every subject's
# spectra (scripts/sensor_spectra_array.sh). Never run this command directly
# on a login node.
#
#   sbatch scripts/band_choice.sh tokens.toml               # fixed aperiodic, the preprint's
#   sbatch scripts/band_choice.sh tokens.toml --aperiodic knee
#
# About 1.5 minutes and 0.3 GB per stream (GNU time below).

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=band-choice
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --time=00:30:00
#SBATCH --output=logs/band-choice-%j.out

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/band_choice.sh <config.toml> [--aperiodic knee]}"
EXTRA=("${@:2}")
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';')"
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" analyze band-choice "${EXTRA[@]}"
/usr/bin/time -v ./.venv/bin/meg-tokens --config "$CONFIG" report band-choice "${EXTRA[@]}"
echo "finished=$(date --iso-8601=seconds)"
