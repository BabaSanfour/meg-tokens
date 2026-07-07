#!/bin/bash
#SBATCH --job-name=meg_gold
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

module load python/3.7.0

# ==============================================================================
# Golden Reference Validation Job
# ==============================================================================
# Required environment:
#   GOLDEN_CONFIG=/path/to/golden_validation.json
#   GOLDEN_OUT_TSV=/path/to/validation_report.tsv
#
# Optional environment:
#   ALLOW_FAILURES=1
# ==============================================================================

: "${GOLDEN_CONFIG:?Set GOLDEN_CONFIG to a validation config JSON file}"
: "${GOLDEN_OUT_TSV:?Set GOLDEN_OUT_TSV to the validation report TSV path}"

EXTRA_ARGS=()
if [ "${ALLOW_FAILURES:-0}" = "1" ]; then
    EXTRA_ARGS+=(--allow_failures)
fi

python -m meg_tokens.utils.batch_validate_golden \
    --config "$GOLDEN_CONFIG" \
    --out_tsv "$GOLDEN_OUT_TSV" \
    "${EXTRA_ARGS[@]}"
