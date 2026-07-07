#!/bin/bash
#SBATCH --job-name=meg_epochs
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified Epoch Extraction & Preprocessing Array Job
# Replaces obsolete: job_resample_raw.sh AND run_all_resample_raw.sh
# ==============================================================================
# Usage:
# sbatch cluster/job_epochs.sh
# ==============================================================================

# Pad the SLURM_ARRAY_TASK_ID with a leading zero
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)

echo "Starting Epochs Extraction for Subject: $SUBJECT"

# Execute the modern batch epochs module which inherently handles preprocessing
python -m meg_tokens.utils.batch_epochs \
    --subjects "$SUBJECT" \
    --raw_dir ./data/raw/ \
    --behavior_dir ./data/behavior/ \
    --out_dir ./data/epochs/ \
    --align_to go

echo "Epochs Job Finished."
