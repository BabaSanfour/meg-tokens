#!/bin/bash
# Archive the raw MEG (.ds), MRI (IRM), and TDMS source data into single
# per-directory tarballs, as a SLURM job. Collapsing ~30,000 files into 3
# matters on its own, since scratch quota here is enforced by file count,
# not bytes.
#
#   sbatch scripts/archive_raw_data.sh [data_root]
#
# Each archive is verified (tar -tzf) before the script moves on. Nothing
# is deleted -- that is a separate, manual step once the archives are
# confirmed good.

#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=meg-tokens-archive
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/archive-%j.out

set -euo pipefail

DATA_ROOT="${1:-/scratch/hamza97/meg-tokens}"
CORES="${SLURM_CPUS_PER_TASK:-8}"

cd "$DATA_ROOT"

for dir in raw IRM tdms; do
    if [ ! -d "$dir" ]; then
        echo "Skipping $dir: not found under $DATA_ROOT"
        continue
    fi
    archive="${dir}.tar.gz"
    if [ -f "$archive" ]; then
        echo "Skipping $dir: $archive already exists"
        continue
    fi

    echo "Archiving $dir -> $archive ($(date))"
    # Write under a .partial name so a killed/timed-out job never leaves
    # something that looks like a finished archive at the real path.
    tar -cf - "$dir" | pigz -p "$CORES" > "$archive.partial"
    mv "$archive.partial" "$archive"

    echo "Verifying $archive ($(date))"
    tar -tzf "$archive" > /dev/null

    echo "OK: $archive ($(du -sh "$archive" | cut -f1)) ($(date))"
done

echo "All archives complete."
