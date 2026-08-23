#!/bin/bash
# Render the whole behavioral figure battery from the group derivatives.
#
#   sbatch scripts/behavior_report.sh
#
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=behavior-report
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/behavior-report-%j.out
set -euo pipefail
cd "$HOME/meg-tokens"
echo "node: $(hostname)"

# fc30670 had a broken Lustre mount and could not read the package, which made
# the run fail with a misleading ModuleNotFoundError. Check the filesystem is
# healthy before spending the allocation on a bad node.
GUARD="$HOME/meg-tokens/meg_tokens/behavior/trials.py"
if [ ! -r "$GUARD" ]; then
    echo "FATAL: cannot read $GUARD on $(hostname) -- Lustre mount is broken" >&2
    exit 1
fi

./.venv/bin/meg-tokens --config tokens.toml report behavior --figures all
echo "DONE"
