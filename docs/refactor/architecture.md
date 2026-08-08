# Refactor Architecture

## Status

This document defines the final package boundaries after the post-replication
refactor. Scientific behavior and the derivative contract remain covered by
unit, workflow, traceability, and golden-reference tests.

The canonical records are:

- `docs/data_contract.md` for persisted data and metadata
- `docs/legacy_traceability.md` for legacy analysis coverage
- `docs/refactor/migration_map.md` for final module ownership

## Target Layers

Dependencies must point down this list and never back up:

1. `meg_tokens.core`: identifiers, immutable configuration, and result models
2. `meg_tokens.io`: paths, discovery, serialization, and contract validation
3. Domain packages: `behavior`, `meg`, `features`, `analysis`, and `reports`
4. `meg_tokens.workflows`: filesystem-aware orchestration for one processing
   unit or one group analysis
5. `meg_tokens.cli`: argument parsing and calls into workflows
6. `workflow/`: Snakemake rules and execution profiles

The CLI must not contain scientific calculations or derivative naming logic.
Domain modules must not import workflows or CLI modules. Cluster scripts invoke
Snakemake, whose rules invoke the installed CLI; neither layer implements
analysis behavior.

The Snakefile invokes only `meg-tokens` commands. Local and Slurm profiles
change execution resources, not scientific parameters or derivative names.
Shared source models are subject-level dependencies; run-level inverse and
source-estimate rules consume them without rebuilding them concurrently.

## Stable Interfaces

The Python workflow API will use typed configuration rather than long argument
lists:

```python
from meg_tokens.core import ProjectConfig, RunSpec
from meg_tokens.workflows import preprocess_run

result = preprocess_run(project, run, settings)
```

The installed command will expose the same workflows:

```text
meg-tokens --config tokens.toml meg preprocess --subject H01 --run Slow1
meg-tokens --config tokens.toml meg epoch --subjects H01 --alignment go
meg-tokens --config tokens.toml features power --subjects H01 --run Slow1 --bands alpha
meg-tokens --config tokens.toml analyze decoding --conditions Fast Slow
```

Each workflow returns a result containing its declared inputs, outputs, and
effective configuration. Batch selection and dependency scheduling remain
outside domain functions.

## Persistence Rules

This structural refactor does not rename existing derivatives.

- MNE objects remain in MNE-native formats.
- Analysis tensors remain `.npy` plus JSON sidecars.
- Tables remain TSV plus JSON sidecars where required.
- Named dimensions may be represented as `xarray.DataArray` in memory, but the
  existing on-disk representation remains stable.
- A future schema change requires a versioned migration and compatibility
  reader; it must not be hidden inside a package move.

All derivative construction and discovery will move to one `DerivativeLayout`.
Where a path is valid BIDS, that implementation may delegate to
`mne_bids.BIDSPath`. Project-specific analysis suffixes remain explicit layout
methods.

## Dependency Boundary

Subject and run entities live in `core`. There are no `utils` modules and no
forwarding shims; the boundary test enforces the complete dependency
direction for `core`, domain packages, workflows, reports, and CLI (zero
`utils` imports anywhere).

## Phase Gate

Every refactor phase must pass:

```text
python -m pytest -q
python -m compileall -q meg_tokens
bash -n cluster/*.sh
```

It must also preserve the no-generated-data and no-MAT/HDF5 production scans
used by the existing hardening tests. Data-dependent stages require a recorded
real-subject smoke run or golden comparison when the project data are mounted.
