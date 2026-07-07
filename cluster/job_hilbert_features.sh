#!/bin/bash
#SBATCH --job-name=meg_hilbert
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified Hilbert Feature Array Job
# Extracts Stage 11 PAC/CFC-ready phase, amplitude, power, and sigfilt features.
# ==============================================================================
# Required environment:
#   TOKENS_BIDS=/path/to/tokens-bids
#   HILBERT_BANDS="theta alpha beta"
# Optional environment:
#   CONDITIONS="Fast Slow"
#   ALIGN_TO=go
#   SOURCE_METHOD=dSPM
#   PARC=HCPMMP1
#   HILBERT_FEATURES="amplitude phase power sigfilt"
#
# Usage:
#   sbatch cluster/job_hilbert_features.sh
# ==============================================================================

if [ -z "$TOKENS_BIDS" ]; then
    echo "TOKENS_BIDS must point to the BIDS derivatives root."
    exit 1
fi

if [ -z "$HILBERT_BANDS" ]; then
    echo "HILBERT_BANDS must list known bands or name=fmin,fmax entries."
    exit 1
fi

SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)
CONDITIONS=${CONDITIONS:-"Fast Slow"}
ALIGN_TO=${ALIGN_TO:-go}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}
PARC=${PARC:-HCPMMP1}
HILBERT_FEATURES=${HILBERT_FEATURES:-"amplitude phase power sigfilt"}

echo "Starting Hilbert feature job for Subject: $SUBJECT"

python -m meg_tokens.utils.batch_hilbert_features \
    --feature_dir "$TOKENS_BIDS" \
    --out_dir "$TOKENS_BIDS" \
    --subjects "$SUBJECT" \
    --conditions $CONDITIONS \
    --align_to "$ALIGN_TO" \
    --source_method "$SOURCE_METHOD" \
    --parc "$PARC" \
    --bands $HILBERT_BANDS \
    --features $HILBERT_FEATURES \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "Hilbert feature job finished."
