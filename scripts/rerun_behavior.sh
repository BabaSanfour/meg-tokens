#!/bin/bash
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=behavior_rerun
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/behavior_rerun_%j.out
set -euo pipefail
cd "$HOME/meg-tokens"
echo "node: $(hostname)"

# fc30670 had a broken Lustre mount and could not read this file, which made
# the run fail with a misleading ModuleNotFoundError. Check the filesystem is
# actually healthy before spending an hour on a bad node.
GUARD="$HOME/meg-tokens/meg_tokens/behavior/trials.py"
if [ ! -r "$GUARD" ]; then
    echo "FATAL: cannot read $GUARD on $(hostname) -- Lustre mount is broken"
    exit 1
fi
echo "filesystem check: OK"

./.venv/bin/meg-tokens --config tokens.toml behavior analyze
echo "--- analyze done ---"
./.venv/bin/meg-tokens --config tokens.toml behavior characterization
echo "DONE"
