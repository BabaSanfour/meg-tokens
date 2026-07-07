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
TOKENS_BIDS=${TOKENS_BIDS:?Set TOKENS_BIDS to the BIDS derivatives root}
RUN_LABEL=${RUN_LABEL:-Slow1}
ALIGN_TO=${ALIGN_TO:-go}
SOURCE_METHOD=${SOURCE_METHOD:-dSPM}

echo "Starting Time-Frequency Power Job for Subject: $SUBJECT"

# Execute the modern batch time-frequency module for this subject
python -m meg_tokens.utils.batch_time_frequency \
    --source_dir "$TOKENS_BIDS" \
    --out_dir "$TOKENS_BIDS" \
    --subjects "$SUBJECT" \
    --run "$RUN_LABEL" \
    --method hilbert \
    --align_to "$ALIGN_TO" \
    --source_method "$SOURCE_METHOD" \
    --n_jobs 8

echo "Power Job Finished."
