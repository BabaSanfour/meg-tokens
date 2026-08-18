#!/bin/bash
# Deprecated recovery merge compatibility wrapper. The expensive recovery simulations
# must be submitted with thura2012_recovery_array.sh first; this job only
# validates/merges their repetition-scoped derivatives. Held-out evaluation is
# handled by thura2012_evaluate_array.sh plus thura2012_evaluate_aggregate.sh.
# Production recovery should use thura2012_recovery_array.sh followed by
# thura2012_recovery_aggregate.sh directly.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thura12-eval-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/thura2012_evaluate.sh <config.toml> [recovery_repetitions]}"
RECOVERY="${2:-12}"
if [ "$RECOVERY" -lt 12 ]; then
    echo "deprecated recovery merge requires at least 12 truth repetitions" >&2
    exit 2
fi

echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} host=$(hostname)"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';') python=$(./.venv/bin/python --version 2>&1)"
echo "command=./.venv/bin/meg-tokens --config ${CONFIG} behavior ssm-recovery-aggregate --repetitions ${RECOVERY}"

./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-recovery-aggregate \
    --repetitions "$RECOVERY"

echo "finished=$(date --iso-8601=seconds)"
