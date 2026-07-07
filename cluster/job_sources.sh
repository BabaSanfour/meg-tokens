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
# Usage:
# sbatch cluster/job_sources.sh
# ==============================================================================

# Pad the SLURM_ARRAY_TASK_ID with a leading zero (e.g., 1 -> H01, 32 -> H32)
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)

echo "Starting Source Extraction Job for Subject: $SUBJECT"

# Execute the modern batch sources module for this subject
python -m meg_tokens.utils.batch_sources \
    --subjects "$SUBJECT"

echo "Source Extraction Job Finished."
