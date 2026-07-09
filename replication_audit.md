# Final Replication and Refactor Audit

Date: 2026-07-08

The production pipeline now replicates the scientific behavior of the legacy
MEG Tokens scripts with modern MNE calls, typed workflow configuration, and
BIDS-style derivatives. It is not a line-by-line port.

## Scope

- MEG Tokens behavior, MEG, MRI-based source reconstruction, features,
  statistics, decoding, PCA/dPCA, trajectories, connectivity, PAC, and reports
  are in scope.
- iEEG-only scripts are out of scope.
- Simulation/model-only scripts are out of scope.
- Production stages require real persisted inputs. They do not generate mock,
  random, simulated, or demonstration project data.
- MNE objects use native FIF/STC formats. Analysis tensors use `.npy` plus JSON
  sidecars; tables use TSV plus JSON sidecars. New `.mat` and HDF5 outputs are
  prohibited.

## Stage Coverage

| Stage | Final implementation |
| --- | --- |
| Behavior ingestion and metrics | `behavior.tdms`, `behavior.metrics`, `workflows.behavior` |
| Filtering, ICA, epoching | `meg.preprocessing`, `meg.epoching`, `workflows.preprocessing` |
| Source models and trial estimates | `meg.sources`, `workflows.sources` |
| ERP, all-source, volume extraction | `features.erp`, `workflows.erp` |
| Power, PSD, specparam | `features.time_frequency`, `workflows.power`, `workflows.spectral` |
| Hilbert and PAC/CFC | `features.time_frequency`, `features.pac`, corresponding workflows |
| Connectivity | `features.connectivity`, `workflows.connectivity` |
| Group and lateralized statistics | `analysis.statistics`, `workflows.statistics` |
| Time-resolved and sensor decoding | `analysis.decoding`, decoding workflows |
| PCA/dPCA and trajectories | `analysis.decomposition`, `workflows.decomposition` |
| Figures and behavior correlations | `reports` |
| Real-reference validation | `validation.golden`, `meg-tokens validate golden` |
| Local and Slurm scheduling | `workflow/Snakefile` and profiles |

## MATLAB nmData Equivalence

The PCA path preserves the relevant `@nmData` behavior:

1. Build condition observations from real ERP or power derivatives.
2. Average by subject by default, with trial averaging available explicitly.
3. Fit one shared PCA basis over condition-by-time observations.
4. Project each condition trajectory through the shared loading axes.
5. Preserve padded invalid times as `NaN`.
6. Save trajectories, loadings, variance, fit scores, and observation tables
   with named dimensions and provenance.

The dPCA path uses real trial metadata to form marginalization cells and
requires the optional external `dPCA` package.

## Final Architecture

- `core`: identifiers, settings, project configuration, workflow results
- `io`: derivative layout, discovery, array/table/xarray adapters
- `behavior`, `meg`, `features`, `analysis`: numerical and MNE domain logic
- `workflows`: persisted-input/persisted-output stages
- `reports`: figures and report tables
- `cli`: the only command parser
- `workflow`: Snakemake dependency scheduling

The transitional `meg_tokens.utils` package and forwarding modules have been
removed.

## Verification

- 189 tests pass.
- Python compilation and shell syntax checks pass.
- A one-subject Snakemake dry-run resolves the complete 21-job DAG.
- Package-boundary tests enforce the final dependency direction.
- Production scans reject generated-data paths and MATLAB/HDF5 writers.
- `docs/legacy_traceability.md` covers every archived executable exactly once:
  253 implemented, 4 out of scope, and 20 archival-only rows.

Real-data numerical validation remains configuration-driven because the large
MEG/MRI dataset is not stored in Git:

```bash
meg-tokens --config tokens.toml validate golden \
  --comparison-config /path/to/golden_validation.json \
  --out-tsv /path/to/validation.tsv
```
