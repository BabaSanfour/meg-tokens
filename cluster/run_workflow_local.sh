#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_CONFIG=${1:-workflow/config.yaml}
shift || true

exec snakemake \
    --snakefile workflow/Snakefile \
    --configfile "$WORKFLOW_CONFIG" \
    --profile workflow/profiles/local \
    "$@"
