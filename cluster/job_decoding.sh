#!/bin/bash
#SBATCH --job-name=meg_decoding
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load python/3.7.0

# ==============================================================================
# Unified ROI Decoding Job
# Replaces: 
#   - job_classif_baseline_enter.sh
#   - job_classif_baseline_go.sh
#   - job_classif_fast_vs_slow_enter.sh
#   - job_classif_fast_vs_slow_go.sh
# ==============================================================================
# Usage Example: 
# sbatch job_decoding.sh "17Networks_LH_SomMotA_1-lh" "./data/epochs_enter/" "speed in ['Fast', 'Slow']"
# ==============================================================================

ROI=$1
DATA_DIR=$2
BEHAVIOR_FILTER=$3

if [ -z "$ROI" ] || [ -z "$DATA_DIR" ]; then
    echo "Usage: sbatch job_decoding.sh <roi> <data_dir> [behavior_filter]"
    echo ""
    echo "HOW TO RUN THE 4 LEGACY VERSIONS:"
    echo "---------------------------------"
    echo "1. Baseline vs Enter (job_classif_baseline_enter):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_enter/\""
    echo ""
    echo "2. Baseline vs Go (job_classif_baseline_go):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_go/\""
    echo ""
    echo "3. Fast vs Slow on Enter (job_classif_fast_vs_slow_enter):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_enter/\" \"speed in ['Fast', 'Slow']\""
    echo ""
    echo "4. Fast vs Slow on Go (job_classif_fast_vs_slow_go):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_go/\" \"speed in ['Fast', 'Slow']\""
    echo ""
    echo "5. Left vs Right Choice on Enter (job_classif_lh_rh_enter):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_enter/\" \"nChoiceMade in [1, 2]\""
    echo ""
    echo "6. Left vs Right Choice on Go (job_classif_lh_rh_go):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_go/\" \"nChoiceMade in [1, 2]\""
    echo ""
    echo "7. Sensory Evidence on Enter (job_classif_sensory_evidence_enter):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_enter/\" \"sTrialClass in [1, 2, 3]\""
    echo ""
    echo "8. Sensory Evidence Enter ALL SOURCES (job_classif_sensory_evidence_enter_all_sources):"
    echo "   sbatch job_decoding.sh \"all\" \"./data/epochs_enter/\" \"sTrialClass in [1, 2, 3]\""
    echo ""
    echo "9. Sensory Evidence on Go (job_classif_sensory_evidence_go):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_go/\" \"sTrialClass in [1, 2, 3]\""
    echo ""
    echo "10. Sensory Evidence Go ALL SOURCES (job_classif_sensory_evidence_go_all_sources):"
    echo "   sbatch job_decoding.sh \"all\" \"./data/epochs_go/\" \"sTrialClass in [1, 2, 3]\""
    echo ""
    echo "11. Trial Class on Enter (job_classif_trial_class_enter):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_enter/\""
    echo ""
    echo "12. Trial Class on Go (job_classif_trial_class_go):"
    echo "   sbatch job_decoding.sh \"17Networks_LH_SomMotA_1-lh\" \"./data/epochs_go/\""
    exit 1
fi

echo "Starting Unified Decoding Job for ROI: $ROI"
echo "Data Directory: $DATA_DIR"
echo "Behavior Filter: $BEHAVIOR_FILTER"

# Construct the base python command using the modernized pipeline
CMD="python -m meg_tokens.utils.batch_decoding_roi --parcellation HCPMMP1 --roi \"$ROI\" --data_dir \"$DATA_DIR\""

# Append the optional dynamic behavior filter if provided
if [ ! -z "$BEHAVIOR_FILTER" ]; then
    CMD="$CMD --behavior_filter \"$BEHAVIOR_FILTER\""
fi

echo "Executing: $CMD"
eval $CMD

echo "Decoding Job Finished."
