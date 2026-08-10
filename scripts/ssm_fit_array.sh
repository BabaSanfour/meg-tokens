#!/bin/bash
# Fit the Tier sequential-sampling models as a SLURM array job,
# one subject per array task.
#
#   sbatch --array=0-31 scripts/ssm_fit_array.sh tokens.toml
#   meg-tokens --config tokens.toml behavior characterization   # aggregate
#
# Each task fits one subject and writes one table to
# derivatives/sub-<ID>/beh/sub-<ID>_task-tokens_desc-ssmcomparison_beh.tsv.
# Tasks never share an output path, so any one of them can fail and be
# requeued on its own with --array=<index>.
#
# Four CPUs per task. One subject is six independent fits -- three conditions
# (all, fast, slow) by two models (ddm, urgency) -- but they are very unequal:
# the urgency fit takes about ten times the integrator's. Wall time is set by
# the single longest fit, the pooled 'all' urgency cell, so cores past four buy
# no wall time and only sit idle.
#
# Set the --array upper bound from the subject count (zero-indexed):
#   meg-tokens --config tokens.toml behavior subjects | wc -l

#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=meg-tokens-ssm
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ssm-%A_%a.out

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/ssm_fit_array.sh <config.toml>}"

# The same subject list, in the same order, that the aggregation step expects.
mapfile -t SUBJECTS < <(meg-tokens --config "$CONFIG" behavior subjects)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} is past the end of the subject list" \
         "(${#SUBJECTS[@]} subjects)."
    exit 0
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: fitting ${SUBJECT}"

meg-tokens --config "$CONFIG" behavior ssm-fit \
    --subjects "$SUBJECT" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-4}"
