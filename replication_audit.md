# Legacy Replication Audit

Date: 2026-07-07

This audit reflects the staged refactor through Stage 12 and the strict legacy
traceability matrix. The target is behavioral/scientific equivalence with modern
MNE-Python and BIDS-style derivatives, not a line-by-line copy of the 2022
scripts.

## Scope Decisions

- MEG Tokens analyses are in scope.
- iEEG-only notebooks are out of scope.
- The old neural-space modeling-only MATLAB file is out of scope and should not
  be ported into the production pipeline.
- New production outputs use MNE-native files, `.tsv` tables with JSON
  sidecars, or `.npy` arrays with JSON sidecars.
- `.mat`, `.h5`, HDF5, and undocumented pickle outputs are not accepted as new
  primary outputs.
- Analysis code must require real staged inputs and fail clearly when inputs are
  absent.

## Current Status

Current local verification:

```text
114 passed
```

| Stage | Status | Modern replacement |
| :--- | :--- | :--- |
| 1. TDMS behavior parsing | Implemented with golden-validation support for old parser exports | `meg_tokens.utils.batch_processor`, `batch_validate_golden.py` |
| 2. Preprocessing and epoching | Implemented for the modern raw-FIF and strict behavior-sync path; manual ICA review and problem-event recovery still need project-specific validation | `batch_preprocess.py`, `batch_epochs.py` |
| 3. Behavioral analysis | Implemented for core DT, accuracy, trial-class, correct/error, and post-error summaries, with golden-validation support for historical report checks | `meg_tokens.behavior`, `batch_plot_behavior.py`, `batch_validate_golden.py` |
| 4. Source reconstruction | Implemented for MNE-native covariance, BEM/source/forward/inverse, trial STC export, manifests, and mixed surface+volume source spaces | `batch_sources.py` |
| 5. Source power | Implemented for Stage 3 source-estimate manifests and sidecar-backed band-power arrays | `batch_time_frequency.py` |
| 5b. PSD/aperiodic modeling | Implemented for staged Epochs derivatives with sidecar-backed PSD arrays and specparam tables | `batch_psd_fooof.py` |
| 6. ERP/parcellation | Implemented for trial-level parcellated, all-source, and volume source-coordinate time courses with aligned `erptrials.tsv` | `batch_erp_parcellation.py` |
| 7. Group statistics | Implemented for paired subject-level contrasts over Stage 6 arrays | `batch_group_statistics.py` |
| 8. Statistical reports and behavior correlations | Implemented for stats summaries, label figures, and optional behavior correlations over real derivatives | `batch_plot_statistics.py` |
| 9. Decoding | Implemented for ERP and power derivatives, ROI and lateralized wrappers, and permutation outputs | `batch_decoding.py`, `batch_decoding_roi.py`, `batch_decoding_lateralized.py` |
| 9.5. PCA/dPCA trajectories | PCA implemented against the MATLAB `@nmData` behavior; optional dPCA tensor creation implemented but external `dPCA` remains optional | `batch_dpca.py`, `meg_tokens.meg.pca` |
| 10. Functional connectivity | Implemented for before/after band connectivity, circle stats, and seed maps from Stage 5 ROI time courses | `batch_connectivity.py`, connectivity plotters |
| 11. Hilbert features for PAC/CFC | Implemented for Brainpipe-style `sigfilt`, amplitude, power, and phase extraction from real Stage 5 derivatives | `batch_hilbert_features.py` |
| 12. PAC/CFC statistics | Implemented for Tort-style phase-amplitude modulation index over Stage 11 derivatives | `batch_pac_cfc.py`, `meg_tokens.meg.pac` |
| Golden validation | Implemented as config-driven array/table comparison against frozen real-reference outputs | `batch_validate_golden.py`, `meg_tokens.validation` |
| Legacy traceability | Implemented as one row per legacy executable with zero production-scope partial rows and pytest coverage | `docs/legacy_traceability.md`, `tests/test_legacy_traceability.py` |

## Stage 5b Detail

Legacy evidence:

- `archive/replicated/DDM_scripts/scripts_new/0000_FOOF_AND_PSD.ipynb`
  computes PSDs and fits aperiodic/periodic spectral models.

Modern replication:

- `meg_tokens.meg.time_frequency.compute_psd` uses `Epochs.compute_psd` from
  MNE-Python.
- `meg_tokens.meg.time_frequency.fit_specparam` uses the declared `specparam`
  dependency. `fit_fooof` remains only as a compatibility wrapper.
- `meg_tokens.utils.batch_psd_fooof` now consumes Stage 2 Epochs FIF derivatives
  and writes `.npy` PSD arrays plus `.tsv` specparam parameter and peak tables,
  all with JSON sidecars.

Outputs are saved as:

```text
sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_psd.npy
sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_specparam.tsv
sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_specparampeaks.tsv
```

with PSD dimensions:

```text
channel x frequency
```

PSD validation is now handled through `batch_validate_golden.py` once frozen
real-reference tables are available.

## Stage 11 Detail

Legacy evidence:

- `archive/replicated/DDM_analysis_scripts/Untitled.ipynb` imports Brainpipe CFC
  helpers and computes band `power`, `amplitude`, and `sigfilt` arrays.

Modern replication:

- `meg_tokens.meg.time_frequency.compute_hilbert_band_features` uses MNE
  filtering plus SciPy Hilbert transforms.
- `meg_tokens.utils.batch_hilbert_features` consumes Stage 5 ERP/parcellation
  derivatives shaped `trial x label x time`.
- Outputs are saved as:

```text
sub-H01_task-tokens_run-1_desc-fast-go-dSPM-HCPMMP1-alpha-amplitude_hilbertfeature.npy
```

with dimensions:

```text
trial x feature x time
```

and sidecar metadata for input derivative, band bounds, sample rate, feature,
condition, run, alignment, source method, parcellation, and labels.

## Stage 12 Detail

Modern replication:

- `meg_tokens.meg.pac.modulation_index` computes Tort-style modulation index
  from low-frequency phase and high-frequency amplitude arrays.
- `meg_tokens.utils.batch_pac_cfc` consumes Stage 11 derivatives shaped
  `trial x feature x time` and writes PAC/CFC arrays with dimensions:

```text
phase_band x amplitude_band x feature
```

Outputs are saved as:

```text
sub-H01_task-tokens_run-1_desc-fast-go-dSPM-HCPMMP1-theta-to-gamma-low-modulation-index_pac.npy
```

PAC/CFC numerical validation is now handled through `batch_validate_golden.py`
once frozen real-reference arrays are available. Additional PAC estimators can
be added later only if the scientific analysis plan requires them.

## Deep, Volume, And All-Source Detail

Modern replication:

- `meg_tokens.utils.batch_sources --volume_labels ...` creates explicit mixed
  surface+volume source spaces with `desc-<spacing>-mixed_src.fif` filenames and
  sidecars recording requested aseg labels.
- `meg_tokens.utils.batch_erp_parcellation --feature_space all_source` writes
  all-source ERP arrays with dimensions `trial x source x time`.
- `meg_tokens.utils.batch_erp_parcellation --feature_space volume` writes volume
  source-coordinate arrays from MNE volume or mixed source estimates.
- Downstream generic feature loaders use explicit `source` and `orientation`
  coordinates, so decoding and PCA can consume these variants without MATLAB
  reshaping exports.

## Legacy Traceability Detail

`docs/legacy_traceability.md` now covers every legacy `.ipynb`, `.py`, `.m`,
and `.sh` file under `archive/replicated`.

Current matrix counts:

```text
implemented: 253
partial: 0
out_of_scope: 4
archive: 20
```

`tests/test_legacy_traceability.py` inventories the archive and verifies that
each executable appears exactly once, that row statuses are limited to the
approved vocabulary, and that the summary counts match the matrix.

## External Validation Step

### Golden Validation

The current tests validate shapes, metadata, missing-input behavior, sidecars,
and command contracts. They do not prove numerical equivalence to old outputs
until frozen real-reference exports are supplied.

Required next steps:

1. Select two or three representative real MEG subjects and export frozen legacy
   reference `.npy`, `.tsv`, or `.csv` outputs.
2. Write a `golden_validation.json` config listing the modern/reference pairs,
   comparison kind, sort columns, selected columns, and numeric tolerances.
3. Record expected subject exclusions, run exclusions, event overrides, and
   condition definitions in a versioned config file.
4. Run `python -m meg_tokens.utils.batch_validate_golden --config ... --out_tsv ...`
   and commit the report or the checked validation config, depending on data
   governance constraints.

## Definition Of Done For A Legacy Script

A script should be marked implemented only when:

1. It has a documented modern command/function or an explicit out-of-scope note.
2. The modern path consumes the same conceptual real inputs.
3. Subject/run inclusion and condition logic are represented.
4. The estimator or a documented modern equivalent is used.
5. Outputs have documented paths, dimensions, coordinates, and sidecars.
6. Missing inputs fail clearly.
7. A test or validation note exists.

By this standard, the main MEG pipeline is implemented through PSD,
connectivity, Hilbert feature extraction, PAC/CFC modulation-index statistics,
deep/volume/all-source source-coordinate variants, golden-validation tooling,
and strict traceability. The remaining task is operational: run the validation
tool against frozen real-reference exports from the project data.
