#!/bin/bash
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --job-name=behavior_figs
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/behavior_figs_%j.out
set -euo pipefail
cd "$HOME/meg-tokens"
echo "node: $(hostname)"
GUARD="$HOME/meg-tokens/meg_tokens/behavior/trials.py"
if [ ! -r "$GUARD" ]; then
    echo "FATAL: Lustre mount broken on $(hostname)"
    exit 1
fi
# F26 needs --neural-metrics and aborts the no-argument run, so name the rest.
KEYS=$(./.venv/bin/meg-tokens --config tokens.toml report behavior --list-figures \
       | awk '{print $1}' | tr "\n" " ")
echo "rendering: $KEYS"
./.venv/bin/meg-tokens --config tokens.toml report behavior --figures $KEYS
echo "DONE"
