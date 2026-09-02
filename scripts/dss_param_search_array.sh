#!/bin/bash
# Restartable Slurm array for the DSS parameter search: one task per subject,
# fifty configurations per task. Never run this on a login node.
#
#   sbatch --array=0-31%16 scripts/dss_param_search_array.sh <bids_root> <output_root>
#
# Each array index owns one subject's output file, so a failed task is requeued
# with the same index and no task can write another subject's result.
#
# One core, deliberately. Threading was measured twice on this workload -- BLAS
# threads 1/2/4 gave 243/241/254 s locally, and four cores on a compute node
# bought 1.3%, inside the run-to-run spread. It is memory-bandwidth bound.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=dss-search
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dss-search-%A_%a.out

set -euo pipefail

BIDS_ROOT="${1:?usage: sbatch --array=0-N scripts/dss_param_search_array.sh <bids_root> <output_root>}"
OUTPUT_ROOT="${2:?usage: sbatch --array=0-N scripts/dss_param_search_array.sh <bids_root> <output_root>}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON="${DSS_SEARCH_PYTHON:-./.venv/bin/python}"

# `uv` does not exist on the cluster and `mne-denoise` is not in the shared
# venv, so the pinned checkout is carried beside this one rather than installed
# into an environment other work depends on. Harmless where it is installed:
# the path is only prepended when the directory is actually there.
DEPS="${DSS_SEARCH_DEPS:-.deps/mne-denoise}"
if [ -d "${DEPS}" ]; then
    export PYTHONPATH="${DEPS}:${PYTHONPATH:-}"
fi

echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "commit=${DSS_SEARCH_SOURCE_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)} host=$(hostname) python=$(${PYTHON} --version 2>&1)"
echo "command=runner --bids-root ${BIDS_ROOT} --output-root ${OUTPUT_ROOT} --index ${SLURM_ARRAY_TASK_ID}"

"${PYTHON}" -m scripts.qc.dss_param_search.runner \
    --bids-root "${BIDS_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --index "${SLURM_ARRAY_TASK_ID}"

echo "finished=$(date --iso-8601=seconds)"
