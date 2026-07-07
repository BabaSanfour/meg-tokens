#!/bin/bash
#SBATCH --job-name=meg_power
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-32

module load python/3.7.0

# ==============================================================================
# Unified Time-Frequency Power Array Job
# Replaces: job_power.sh AND run_all_power.sh
# ==============================================================================
# Usage:
# sbatch cluster/job_power.sh
# ==============================================================================

# Pad the SLURM_ARRAY_TASK_ID with a leading zero (e.g., 1 -> H01, 32 -> H32)
SUBJECT=$(printf "H%02d" $SLURM_ARRAY_TASK_ID)

echo "Starting Time-Frequency Power Job for Subject: $SUBJECT"

# Execute the modern batch time-frequency module for this subject
python -m meg_tokens.utils.batch_time_frequency \
    --subjects "$SUBJECT" \
    --method hilbert \
    --align_to go \
    --n_jobs 8

echo "Power Job Finished."
