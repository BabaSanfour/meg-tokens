#!/bin/bash
# Subject × strict complete-log/alignment sensitivity refits. Run only after
# the primary mechanistic fit array is complete; one task per subject writes a
# unique derivative. The audit records whether the two source masks are equal.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-excl
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thura12-excl-%A_%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-N scripts/thura2012_exclusion_array.sh <config.toml>}"
mapfile -t SUBJECTS < <(./.venv/bin/meg-tokens --config "$CONFIG" behavior subjects)
RULES=(complete_token_log_alignment)
INDEX="${SLURM_ARRAY_TASK_ID}"
SUBJECT_INDEX="$INDEX"
RULE_INDEX=0
if [ "$SUBJECT_INDEX" -ge "${#SUBJECTS[@]}" ]; then exit 0; fi
SUBJECT="${SUBJECTS[$SUBJECT_INDEX]}"
RULE="${RULES[$RULE_INDEX]}"
echo "started=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} subject=${SUBJECT} rule=${RULE} commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | tr '\n' ';')"
./.venv/bin/meg-tokens --config "$CONFIG" behavior ssm-exclusion-robustness \
    --subjects "$SUBJECT" --rule "$RULE" --n-jobs "${SLURM_CPUS_PER_TASK:-1}" --n-starts 2
echo "finished=$(date --iso-8601=seconds)"
