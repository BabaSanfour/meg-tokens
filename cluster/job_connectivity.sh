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
# Usage:
# sbatch cluster/job_connectivity.sh
# ==============================================================================

# Pad the SLURM_ARRAY_TASK_ID with a leading zero (e.g., 1 -> H01, 32 -> H32)
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)

echo "Starting Connectivity Job for Subject: $SUBJECT"

# Execute the modern batch connectivity module for this subject
python -m meg_tokens.utils.batch_connectivity \
    --subjects "$SUBJECT" \
    --parc HCPMMP1 \
    --method imcoh \
    --data_dir ./data/epochs_enter/ \
    --out_dir ./ROI_imcoh/

echo "Connectivity Job Finished."
