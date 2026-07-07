#!/bin/bash
#SBATCH --job-name=meg_conn
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified Connectivity Array Job
# Replaces: job_conn.sh AND run_all_conn.sh
# ==============================================================================
# Required environment:
#   TOKENS_BIDS=/path/to/tokens-bids
# Optional environment:
#   CONDITIONS="Fast Slow"
#   ALIGN_TO=enter
#   SOURCE_METHOD=dSPM
#   PARC=HCPMMP1
#   CONNECTIVITY_METHOD=imcoh
#   CONNECTIVITY_BANDS="delta theta alpha beta"
#   BEFORE_WINDOW="0.7 1.4"
#   AFTER_WINDOW="1.6 2.3"
#
# Usage:
#   sbatch cluster/job_connectivity.sh
# ==============================================================================

if [ -z "$TOKENS_BIDS" ]; then
    echo "TOKENS_BIDS must point to the BIDS derivatives root."
    exit 1
fi

# Pad the SLURM_ARRAY_TASK_ID with a leading zero (e.g., 1 -> H01, 32 -> H32)
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)
CONDITIONS=${CONDITIONS:-"Fast Slow"}
ALIGN_TO=${ALIGN_TO:-enter}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}
PARC=${PARC:-HCPMMP1}
CONNECTIVITY_METHOD=${CONNECTIVITY_METHOD:-imcoh}
CONNECTIVITY_BANDS=${CONNECTIVITY_BANDS:-"delta theta alpha beta"}
BEFORE_WINDOW=${BEFORE_WINDOW:-"0.7 1.4"}
AFTER_WINDOW=${AFTER_WINDOW:-"1.6 2.3"}

echo "Starting Connectivity Job for Subject: $SUBJECT"

# Execute the modern batch connectivity module for this subject
python -m meg_tokens.utils.batch_connectivity \
    --feature_dir "$TOKENS_BIDS" \
    --out_dir "$TOKENS_BIDS" \
    --subjects "$SUBJECT" \
    --conditions $CONDITIONS \
    --align_to "$ALIGN_TO" \
    --source_method "$SOURCE_METHOD" \
    --parc "$PARC" \
    --method "$CONNECTIVITY_METHOD" \
    --bands $CONNECTIVITY_BANDS \
    --before_window $BEFORE_WINDOW \
    --after_window $AFTER_WINDOW \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "Connectivity Job Finished."
