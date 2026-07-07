#!/bin/bash
#SBATCH --job-name=meg_src
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified Source Extraction Array Job
# Replaces: job_sources.sh AND run_all_sources.sh
# ==============================================================================
# Required environment:
#   TOKENS_RAW_DIR=/path/to/raw-or-noise
#   TOKENS_EPOCHS_DIR=/path/to/tokens-bids
#   TOKENS_TRANS_DIR=/path/to/trans-files
#   TOKENS_SUBJECTS_DIR=/path/to/freesurfer-subjects
#   TOKENS_BIDS=/path/to/tokens-bids
#
# Optional environment:
#   RUN=Slow1
#   ALIGN_TO=go
#   SOURCE_METHOD=dSPM
#   SPACING=oct6
#   STAGES="cov bem src fwd inv apply"
#   VOLUME_LABELS="Left-Putamen Right-Putamen"
#   VOLUME_POS=5.0
#
# Usage:
# sbatch cluster/job_sources.sh
# ==============================================================================

# Pad the SLURM_ARRAY_TASK_ID with a leading zero (e.g., 1 -> H01, 32 -> H32)
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)
RUN=${RUN:-Slow1}
ALIGN_TO=${ALIGN_TO:-go}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}
SPACING=${SPACING:-oct6}
STAGES=${STAGES:-"cov bem src fwd inv apply"}
VOLUME_POS=${VOLUME_POS:-5.0}

: "${TOKENS_RAW_DIR:?Set TOKENS_RAW_DIR to the raw/noise data root}"
: "${TOKENS_EPOCHS_DIR:?Set TOKENS_EPOCHS_DIR to the Stage 2 derivatives root}"
: "${TOKENS_TRANS_DIR:?Set TOKENS_TRANS_DIR to the MEG-MRI trans file root}"
: "${TOKENS_SUBJECTS_DIR:?Set TOKENS_SUBJECTS_DIR to the FreeSurfer subjects directory}"
: "${TOKENS_BIDS:?Set TOKENS_BIDS to the derivatives root}"

EXTRA_ARGS=()
if [ -n "${VOLUME_LABELS:-}" ]; then
    EXTRA_ARGS+=(--volume_labels $VOLUME_LABELS --volume_pos "$VOLUME_POS")
fi

echo "Starting Source Extraction Job for Subject: $SUBJECT"

# Execute the modern batch sources module for this subject
python -m meg_tokens.utils.batch_sources \
    --subjects "$SUBJECT" \
    --raw_dir "$TOKENS_RAW_DIR" \
    --epochs_dir "$TOKENS_EPOCHS_DIR" \
    --trans_dir "$TOKENS_TRANS_DIR" \
    --subjects_dir "$TOKENS_SUBJECTS_DIR" \
    --out_dir "$TOKENS_BIDS" \
    --run "$RUN" \
    --align_to "$ALIGN_TO" \
    --method "$SOURCE_METHOD" \
    --spacing "$SPACING" \
    --stages $STAGES \
    "${EXTRA_ARGS[@]}"

echo "Source Extraction Job Finished."
