#!/bin/bash
#SBATCH --job-name=meg_psd
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified PSD + Specparam Array Job
# ==============================================================================
# Required environment:
#   TOKENS_BIDS=/path/to/tokens-bids
# Optional environment:
#   CONDITION=Fast
#   ALIGN_TO=go
#   PSD_METHOD=welch
#   PSD_FMIN=1.0
#   PSD_FMAX=100.0
#   PSD_N_FFT=2048
#   PSD_N_OVERLAP=150
#   NO_SPECPARAM=0
#
# Usage:
#   sbatch cluster/job_psd_specparam.sh
# ==============================================================================

if [ -z "$TOKENS_BIDS" ]; then
    echo "TOKENS_BIDS must point to the BIDS derivatives root."
    exit 1
fi

SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)
CONDITION=${CONDITION:-}
ALIGN_TO=${ALIGN_TO:-go}
PSD_METHOD=${PSD_METHOD:-welch}
PSD_FMIN=${PSD_FMIN:-1.0}
PSD_FMAX=${PSD_FMAX:-100.0}
PSD_N_FFT=${PSD_N_FFT:-2048}
PSD_N_OVERLAP=${PSD_N_OVERLAP:-150}
NO_SPECPARAM=${NO_SPECPARAM:-0}

CMD=(python -m meg_tokens.utils.batch_psd_fooof
    --epochs_dir "$TOKENS_BIDS"
    --out_dir "$TOKENS_BIDS"
    --subjects "$SUBJECT"
    --align_to "$ALIGN_TO"
    --method "$PSD_METHOD"
    --fmin "$PSD_FMIN"
    --fmax "$PSD_FMAX"
    --n_fft "$PSD_N_FFT"
    --n_overlap "$PSD_N_OVERLAP"
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}")

if [ -n "$CONDITION" ]; then
    CMD+=(--condition "$CONDITION")
fi

if [ "$NO_SPECPARAM" = "1" ]; then
    CMD+=(--no_specparam)
fi

echo "Starting PSD/specparam job for Subject: $SUBJECT"
echo "Executing: ${CMD[*]}"
"${CMD[@]}"
echo "PSD/specparam job finished."
