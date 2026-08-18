#!/bin/bash
# Per-subject held-out/predictive/matched diagnostics. Run after
# thura2012_aggregate.sh, then submit thura2012_evaluate_aggregate.sh to
# validate and merge these outputs. Outputs are subject-scoped, so tasks can
# run in parallel without path collisions. Recovery is a separate
# repetition-indexed array followed by thura2012_recovery_aggregate.sh; it is
# never run in this evaluation array.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-eval-array
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Held-out fitting is serial within a subject (36 model/condition/fold cells).
# Observed production throughput was 11--18 cells in two hours, so four hours
# cannot accommodate the slower subjects.  Twelve hours gives a defensible
# margin without changing the statistical analysis.
#SBATCH --time=12:00:00
#SBATCH --output=logs/thura12-eval-%A_%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-N scripts/thura2012_evaluate_array.sh <config.toml>}"
mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then exit 0; fi
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} subject=${SUBJECT} commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';')"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-evaluate \
    --subjects "$SUBJECT" --folds 3 --n-starts 1 --recovery-repetitions 0
echo "finished=$(date --iso-8601=seconds)"
