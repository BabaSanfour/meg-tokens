#!/bin/bash
#SBATCH --job-name=meg_pac
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified PAC/CFC Array Job
# Computes modulation-index PAC from Stage 11 Hilbert feature derivatives.
# ==============================================================================
# Required environment:
#   TOKENS_BIDS=/path/to/tokens-bids
#   PAC_PHASE_BANDS="theta"
#   PAC_AMPLITUDE_BANDS="gamma_low gamma_high"
# Optional environment:
#   CONDITIONS="Fast Slow"
#   ALIGN_TO=go
#   SOURCE_METHOD=dSPM
#   PARC=HCPMMP1
#   PAC_N_BINS=18
#   PAC_TIME_WINDOW="0.0 1.5"
#
# Usage:
#   sbatch cluster/job_pac_cfc.sh
# ==============================================================================

if [ -z "$TOKENS_BIDS" ]; then
    echo "TOKENS_BIDS must point to the BIDS derivatives root."
    exit 1
fi

if [ -z "$PAC_PHASE_BANDS" ]; then
    echo "PAC_PHASE_BANDS must list low-frequency phase bands."
    exit 1
fi

if [ -z "$PAC_AMPLITUDE_BANDS" ]; then
    echo "PAC_AMPLITUDE_BANDS must list high-frequency amplitude bands."
    exit 1
fi

SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)
CONDITIONS=${CONDITIONS:-"Fast Slow"}
ALIGN_TO=${ALIGN_TO:-go}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}
PARC=${PARC:-HCPMMP1}
PAC_N_BINS=${PAC_N_BINS:-18}

CMD=(python -m meg_tokens.utils.batch_pac_cfc
    --feature_dir "$TOKENS_BIDS"
    --out_dir "$TOKENS_BIDS"
    --subjects "$SUBJECT"
    --conditions $CONDITIONS
    --phase_bands $PAC_PHASE_BANDS
    --amplitude_bands $PAC_AMPLITUDE_BANDS
    --align_to "$ALIGN_TO"
    --source_method "$SOURCE_METHOD"
    --parc "$PARC"
    --n_bins "$PAC_N_BINS")

if [ -n "$PAC_TIME_WINDOW" ]; then
    CMD+=(--time_window $PAC_TIME_WINDOW)
fi

echo "Starting PAC/CFC job for Subject: $SUBJECT"
echo "Executing: ${CMD[*]}"
"${CMD[@]}"
echo "PAC/CFC job finished."
