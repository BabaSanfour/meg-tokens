# MEG Tokens Data Contract

This project targets behavioral replication of the legacy scripts using modern
MNE-Python, BIDS/MNE-BIDS-style derivatives, and explicit analysis tensors.

## Core Rules

- Analysis code must use real project inputs only.
- Package code, notebooks, documentation, examples, and cluster scripts must not
  generate fake MEG, source, behavior, decoding, PCA, or connectivity data.
- Missing inputs should raise a clear error that names the required file.
- Modern MNE/scientific-Python APIs are preferred over step-by-step ports when
  they preserve the same scientific behavior.

## Storage

- Raw, cleaned raw, epochs, forward models, inverse operators, source spaces,
  morphs, labels, and source estimates should use MNE-native formats under a
  BIDS derivatives root.
- Parsed behavior runs should be `.tsv` tables under
  `derivatives/meg-tokens/sub-<ID>/beh/` with JSON sidecars.
- Analysis tensors should use `.npy` plus a JSON sidecar, or xarray/Zarr where
  labeled coordinates are essential.
- New primary outputs should not be `.mat`, `.h5`, or undocumented pickle files.

## Array Sidecars

Every `.npy` derivative written by the refactored pipeline should have a JSON
sidecar containing:

- `shape`
- `dtype`
- `dims`
- `coords`
- `metadata`

Downstream stages should validate the sidecar before consuming an array.

## Behavior Tables

Stage 1 writes one table per TDMS run using:

```text
derivatives/meg-tokens/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv
```

Required trial columns include:

- `subject`
- `condition`
- `run`
- `source_file`
- `nTrialIndex`
- `sTrialClass`
- `nChoiceMade`
- `nCorrectChoice`
- `tGO`
- `tEnterTarget`
- `tTrialEnd`
- `rawRT`
- `isCorrect`

Subject labels are normalized to `H01` style. The behavior parser validates
sequential trial indices and basic event ordering before writing a table.

## Preprocessed Raw Files

Stage 2 preprocessing writes cleaned or filtered raw FIF files under:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_proc-filt_desc-slow_raw.fif
```

The JSON sidecar records the subject, condition, run, sampling frequency,
channel count, and preprocessing label.

## Epochs And Events

Stage 2 epoching consumes:

- cleaned/filtered raw FIF derivatives
- Stage 1 behavior TSV derivatives

It writes:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go_epo.fif
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go_events.tsv
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go_eve.eve
```

The epoch builder is strict by default: MEG event counts and behavior rows must
match. Truncation is available only through an explicit diagnostic option in the
library API and should not be used for production replication.

## Source Reconstruction

Stage 3 consumes:

- Stage 2 epoch FIF derivatives
- empty-room/noise recordings
- FreeSurfer subject directories
- MEG-MRI `-trans.fif` files for forward/inverse/source-estimate stages

It writes MNE-native derivatives such as:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_desc-noise_cov.fif
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_desc-singlelayer_bem.fif
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_space-subject_desc-oct6_src.fif
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go_fwd.fif
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM_inv.fif
```

Mixed surface+volume source spaces are explicit and do not overwrite
cortical-only derivatives. When `batch_sources.py` is run with
`--volume_labels`, the source-space derivative uses:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_space-subject_desc-oct6-mixed_src.fif
```

The source-space sidecar records the surface spacing, requested aseg volume
labels, and volume grid spacing in millimeters.

Trial source estimates are written with MNE `SourceEstimate.save(ftype="stc")`
using one extensionless base path per trial, plus a manifest consumed by later
stages:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM_stcmanifest.tsv
```

Each model, inverse, source estimate base, and manifest gets a JSON sidecar.
Source estimates that MNE can only persist through disallowed container formats
must be converted to scalar `.stc` estimates before export.

## Source-Space Power Extraction

Stage 4 consumes Stage 3 source-estimate manifests:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM_stcmanifest.tsv
```

It writes one array derivative per frequency band:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-hilbert-alpha_power.npy
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-hilbert-alpha_power.json
```

The power array dimensions are:

```text
trial x source x time
```

Vector estimates, if explicitly supported by a future source-export path, add an
orientation dimension:

```text
trial x source x orientation x time
```

The sidecar records the source manifest path, source method, power method,
frequency band, sample-rate, sliding-window parameters, baseline settings,
trial coordinates, time coordinates, and source vertices.

## PSD And Spectral Parameterization

Stage 5b consumes Stage 2 Epochs FIF derivatives:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go_epo.fif
```

It computes run-level mean PSDs across epochs and writes:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_psd.npy
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_psd.json
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_specparam.tsv
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-welch-1to100hz_specparampeaks.tsv
```

The PSD dimensions are:

```text
channel x frequency
```

The PSD sidecar records the input Epochs file, subject, run, condition,
alignment, PSD method, frequency bounds, FFT settings, number of epochs, channel
names, and frequency coordinates.

The `specparam.tsv` table stores one row per channel with aperiodic parameters,
fit metrics, and the number of fitted peaks. The `specparampeaks.tsv` table
stores one row per fitted periodic peak with center frequency, power, and
bandwidth. This replaces the old FOOOF runtime dependency with the declared
`specparam` dependency while keeping the legacy command name as a compatibility
entry point.

## ERP Slicing And Parcellation

Stage 5 consumes:

- Stage 3 source-estimate manifests
- Stage 1 behavior TSV derivatives
- FreeSurfer annotation labels for the requested parcellation

It writes a trial-level parcellated source time-course array plus a trial
metadata table:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erp.npy
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erp.json
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erptrials.tsv
```

The array dimensions are:

```text
trial x label x time
```

Vector data add a component dimension:

```text
trial x component x label x time
```

Go-aligned outputs keep the legacy behavior of cutting each trial before the
response and padding the rest of the fixed-length window with `NaN`. Trials
that fail the minimum reaction-time rule or exceed the fixed window are excluded
from the array and listed only by absence from the `erptrials.tsv` output.

The same command can write source-coordinate variants when legacy all-source or
deep/volume analyses are required:

```bash
python -m meg_tokens.utils.batch_erp_parcellation --feature_space all_source ...
python -m meg_tokens.utils.batch_erp_parcellation --feature_space volume ...
```

All-source output uses:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-slow-go-dSPM-all-source_erp.npy
```

with dimensions:

```text
trial x source x time
```

Vector estimates add orientation:

```text
trial x source x orientation x time
```

Volume output uses `desc-...-volume_erp.npy` with the same source-coordinate
dimensions, selecting volume groups from MNE volume or mixed source estimates.
The sidecar records `feature_space`, source vertices, source labels, and the
trial metadata table. Downstream generic feature loaders consume these arrays
through their explicit dimensions and coordinates.

## Group Statistics

Stage 6 consumes Stage 5 ERP arrays. For each subject and condition it averages
trials within each run, then averages runs within the same condition. It then
runs a paired subject-level contrast:

```text
condition_1 - condition_2
```

It writes:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_tstat.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_pvalue.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_contrast.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_h0.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_sigwindows.tsv
```

`tstat` and `pvalue` use the same feature dimensions as the subject-level ERP
mean, usually:

```text
label x time
```

`contrast` keeps the subject dimension:

```text
subject x label x time
```

Features that are not finite for all subjects, including late padded `NaN`
regions, are retained as `NaN` in the statistical outputs and excluded from the
permutation test.

## Statistical Plotting And Reporting

Stage 7 consumes Stage 6 group-statistics derivatives. It writes a label-level
summary table and selected figures:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_statsummary.tsv
derivatives/meg-tokens/sub-group/fig/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1-label000_stattimecourse.png
derivatives/meg-tokens/sub-group/fig/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1-label000_stattimecourse.json
```

The summary table includes each label's first significant time point, peak
latency, peak statistic, minimum p-value, and number of significant windows.

When behavior correlation is explicitly enabled, Stage 7 also consumes Stage 1
behavior TSV derivatives and writes:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_statcorrelations.tsv
derivatives/meg-tokens/sub-group/fig/sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1-label000_statcorrelation.png
```

The correlation uses each subject's mean `rawRT` from the requested conditions
and the subject-level peak latency from the selected contrast label.

## Decoding

Stage 8 consumes either Stage 6 ERP arrays or Stage 4 source-power arrays and
builds a trial-level matrix:

```text
trial x feature x time
```

For ERP inputs, features are labels, component-label combinations, selected
ROIs, or left-minus-right label pairs. For power inputs, features are source
points or source-orientation combinations.

It writes:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decoding.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decodingsplits.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decodingthreshold.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decodingpermutations.npy
derivatives/meg-tokens/sub-group/fig/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1_decodingplot.png
```

`decoding.npy` stores mean cross-validation accuracy over time. `decodingsplits`
stores split-level accuracy, and permutation outputs are present only when
requested. Time points containing non-finite padded values are excluded from
classifier fitting and retained as `NaN` in the decoding outputs.

## PCA Trajectories

Stage 9 consumes Stage 4 power arrays or Stage 5 ERP arrays. It builds
condition means, fits PCA over condition-by-time samples, and writes:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcacondmeans.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcaloadings.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcavariance.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcafitscores.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcafitsamples.tsv
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcaobservations.tsv
```

The primary trajectory dimensions are:

```text
condition x component x time
```

Loadings use:

```text
feature x component
```

Sidecars record source derivative paths, feature coordinates, time coordinates,
subject/condition observations, PCA fit range, transform, and whether
trajectories were projected with the nmData-style raw projection or centered
scikit-learn projection.

## Functional Connectivity

Stage 10 consumes Stage 5 ERP/parcellation arrays:

```text
trial x label x time
```

It computes band-averaged spectral connectivity in explicit before/after time
windows and writes one derivative per subject/run/condition:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-enter-dSPM-HCPMMP1-imcoh_connectivity.npy
```

The connectivity array dimensions are:

```text
window x band x node_from x node_to
```

The sidecar records node labels, band names and bounds, input ERP derivative,
method, mode, sampling rate, and before/after windows in seconds.

Group connectivity plotting consumes those derivatives, averages repeated runs
within each subject, and writes active-minus-baseline statistics:

```text
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-alpha-connectivity_connectivityadjacency.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-alpha-connectivity_connectivitytstat.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-alpha-connectivity_connectivitypvalue.npy
derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-fast-alpha-connectivity_connectivityh0.npy
```

Seed connectivity outputs use:

```text
node
```

and store the seed name/index, node labels, subjects, and input connectivity
derivatives in the sidecar.

## Hilbert Features For PAC/CFC

Stage 11 consumes Stage 5 ERP/parcellation arrays:

```text
trial x label x time
```

It extracts band-filtered signal (`sigfilt`), Hilbert amplitude, Hilbert power,
and phase from real staged derivatives. It writes one derivative per
subject/run/condition/band/feature:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-dSPM-HCPMMP1-alpha-amplitude_hilbertfeature.npy
```

The array dimensions are:

```text
trial x feature x time
```

The sidecar records selected labels, band bounds, inferred or supplied sample
rate, input ERP derivative, alignment, source method, parcellation, feature
name, and time coordinates. This stage replaces the legacy Brainpipe
amplitude/power/sigfilt export behavior without `.mat` files.

## PAC/CFC Modulation Index

Stage 12 consumes Stage 11 low-frequency phase and high-frequency amplitude
derivatives:

```text
trial x feature x time
```

It computes Tort-style phase-amplitude modulation index for each requested
phase-band/amplitude-band pair and writes one derivative per
subject/run/condition:

```text
derivatives/meg-tokens/sub-H01/meg/sub-H01_task-tokens_run-1_desc-fast-go-dSPM-HCPMMP1-theta-to-gamma-low+gamma-high-modulation-index_pac.npy
```

The PAC array dimensions are:

```text
phase_band x amplitude_band x feature
```

The sidecar records the input phase and amplitude derivatives, node/feature
labels, alignment, source method, parcellation, phase bins, optional time
window, and exact phase/amplitude band coordinates. Missing Stage 11 derivatives
raise a clear input error rather than recomputing or fabricating features.

## Golden Validation

Real-subject validation is run from a JSON comparison config:

```bash
python -m meg_tokens.utils.batch_validate_golden \
  --config /path/to/golden_validation.json \
  --out_tsv /path/to/tokens-bids/derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-golden-validation_validation.tsv
```

The config contains a non-empty `comparisons` list. Each item names a modern
derivative, a frozen real-reference derivative, and a comparison kind:

```json
{
  "comparisons": [
    {
      "name": "H01_slow1_behavior",
      "kind": "table",
      "modern": "/path/to/modern.tsv",
      "reference": "/path/to/reference.tsv",
      "sort_by": ["nTrialIndex"],
      "columns": ["nTrialIndex", "sTrialClass", "rawRT", "isCorrect"]
    },
    {
      "name": "H01_slow1_all_source_erp",
      "kind": "array",
      "modern": "/path/to/modern.npy",
      "reference": "/path/to/reference.npy",
      "atol": 1e-8,
      "rtol": 1e-5,
      "compare_sidecar_keys": ["dims", "coords"]
    }
  ]
}
```

Missing files raise a clear input error. Mismatches are written to the report
and cause the command to exit nonzero unless `--allow_failures` is supplied.

## Path Convention

Use `meg_tokens.io.derivative_path` for BIDS-derivatives-style paths:

```python
from meg_tokens.io import derivative_path

path = derivative_path(
    "/data/tokens-bids",
    pipeline="meg-tokens",
    subject="H01",
    task="tokens",
    description="goAlphaFast",
    suffix="power",
    extension=".npy",
)
```

This yields a path under:

```text
/data/tokens-bids/derivatives/meg-tokens/sub-H01/meg/
```

Condition, alignment, band, parcellation, ROI, estimator, and units should also
be stored in the JSON sidecar so downstream stages do not rely only on filenames.
