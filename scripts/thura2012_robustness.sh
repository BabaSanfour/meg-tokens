#!/bin/bash
# Deprecated safety wrapper. The complete six-configuration × 32-subject
# grid must not be submitted as one 12-hour job. Use
# thura2012_robustness_array.sh followed by
# thura2012_robustness_aggregate.sh instead.

#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=meg-thura12-robust
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thura12-robust-%j.out

set -euo pipefail
echo "Do not submit this deprecated monolithic wrapper; use scripts/thura2012_robustness_array.sh" >&2
exit 2
