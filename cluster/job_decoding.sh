#!/bin/bash
#SBATCH --job-name=meg_decoding
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load python/3.7.0

# ROI decoding over Stage 6 ERP derivatives.
#
# Required:
#   export TOKENS_BIDS=/path/to/tokens-bids
#
# Usage:
#   sbatch cluster/job_decoding.sh Label_1-lh
#   sbatch cluster/job_decoding.sh all

ROI=$1
TOKENS_BIDS=${TOKENS_BIDS:?Set TOKENS_BIDS to the BIDS derivatives root}
CONDITIONS=${CONDITIONS:-"Fast Slow"}
ALIGN_TO=${ALIGN_TO:-go}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}
PARC=${PARC:-HCPMMP1}
PERMUTATIONS=${PERMUTATIONS:-0}

if [ -z "$ROI" ]; then
    echo "Usage: sbatch cluster/job_decoding.sh <roi|all>"
    echo "Set TOKENS_BIDS, CONDITIONS, ALIGN_TO, SOURCE_METHOD, PARC, and PERMUTATIONS as needed."
    exit 1
fi

CMD=(python -m meg_tokens.utils.batch_decoding_roi
    --feature_dir "$TOKENS_BIDS"
    --out_dir "$TOKENS_BIDS"
    --conditions $CONDITIONS
    --align_to "$ALIGN_TO"
    --source_method "$SOURCE_METHOD"
    --parc "$PARC"
    --permutations "$PERMUTATIONS"
    --n_jobs "$SLURM_CPUS_PER_TASK")

if [ "$ROI" != "all" ]; then
    CMD+=(--roi "$ROI")
fi

echo "Starting decoding job for ROI: $ROI"
echo "Executing: ${CMD[*]}"
"${CMD[@]}"
echo "Decoding job finished."
