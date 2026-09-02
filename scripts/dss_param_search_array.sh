#!/bin/bash
# Restartable Slurm array for the DSS parameter search: one task per subject,
# fifty configurations per task. Never run this on a login node.
#
#   sbatch --array=0-31%16 scripts/dss_param_search_array.sh \
#       <bids_root> <output_root> [python]
#
# `python` defaults to this checkout's own `./.venv/bin/python`, matching every
# other launcher here. Name one explicitly when running from a checkout that has
# no venv of its own -- the search is meant to be runnable from a clean tree
# beside a working one, and borrowing an interpreter is how that is done.
#
# Each array index owns one subject's output file, so a failed task is requeued
# with the same index and no task can write another subject's result.
#
# One core, deliberately. Threading was measured twice on this workload -- BLAS
# threads 1/2/4 gave 243/241/254 s locally, and four cores on a compute node
# bought 1.3%, inside the run-to-run spread. It is memory-bandwidth bound.
#
# Resources are measured, not assumed: job 57785083 ran the full 50-configuration
# grid on H01 from Lustre in 9:11 at a 10.58 GB peak. 16 GB and one hour are that
# with headroom for a longer recording.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=dss-search
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dss-search-%A_%a.out

set -euo pipefail

USAGE="usage: sbatch --array=0-N scripts/dss_param_search_array.sh <bids_root> <output_root> [python]"
BIDS_ROOT="${1:?$USAGE}"
OUTPUT_ROOT="${2:?$USAGE}"
PYTHON="${3:-${DSS_SEARCH_PYTHON:-./.venv/bin/python}}"

# Fail here, naming the interpreter, rather than thirty-two tasks into the array
# with "./.venv/bin/python: No such file or directory" in each of their logs.
if [ ! -x "${PYTHON}" ]; then
    echo "no interpreter at ${PYTHON}; pass one as the third argument. ${USAGE}" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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
