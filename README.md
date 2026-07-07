# MEG Tokens Task Analysis (DDM Project)

This repository contains the analysis scripts and a refactored Python library for investigating decision-making dynamics using the **Tokens Task** paired with Magnetoencephalography (MEG). iEEG notebooks in the archive are out of scope for this refactor.

## 🗂️ Project Structure

*   **`meg_tokens/`**: Main Python package with refactored, clean production code.
    *   `behavior/`: Modules for parsing behavior logs, calculating reaction times, and plotting performance.
    *   `meg/`: Modules for neural data preprocessing, ICA, and source localization.
    *   `utils/`: Helpers for reading TDMS files and I/O.
*   **`tests/`**: Unit tests.
*   **`docs/`**: Data contracts and refactor notes, including the strict one-row-per-file legacy mapping in [`docs/legacy_traceability.md`](docs/legacy_traceability.md).
*   **`pyproject.toml`**: Metadata and dependency configuration for the python package.
*   **`archive/`**: Contains the raw, unorganized scripts copied from the external drives:
    *   `DDM_scripts/`: Python/Jupyter notebooks (`scripts_new/`) and Matlab scripts (`matlab_scripts/`) copied from the `DDM_scripts` partition.
    *   `DDM_analysis_scripts/`: Jupyter notebooks copied from the `DDM/scripts/` partition.

## 🚀 Pipeline Execution Flow & Module Map

The analysis pipeline is designed to be executed sequentially from raw data ingestion to group-level parcellation and export. 

| Step | Stage | Module | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Behavioral Log Parsing** | [`tdms_parser.py`](meg_tokens/utils/tdms_parser.py) | Parses raw LabVIEW `.tdms` logs into BIDS-derivatives-style behavior TSV tables with JSON sidecars. |
| **2** | **Behavioral Metrics Extraction** | [`analysis.py`](meg_tokens/behavior/analysis.py) | Computes choice RTs, accuracy, difficulty levels, and formats behavioral variables. |
| **3** | **Behavioral Performance Plotting** | [`plotting.py`](meg_tokens/behavior/plotting.py) | Renders performance diagnostics, psychometric response curves, and RT distributions. |
| **4** | **MEG Preprocessing & Filtering** | [`preprocessing.py`](meg_tokens/meg/preprocessing.py) | Loads raw CTF MEG, applies filters, performs ICA decomposition, and coregisters head points. |
| **5** | **Epoch Extraction & Event Alignment** | [`epochs_builder.py`](meg_tokens/utils/epochs_builder.py) | Segments MEG data into trial-by-trial epochs aligned to triggers, filtered by behavior. |
| **6** | **Pipeline Automation & Batching** | [`batch_processor.py`](meg_tokens/utils/batch_processor.py) | Automates steps 1-5 in batch across blocks, runs, and subjects. |
| **7** | **Neural Source Localization** | [`sources.py`](meg_tokens/meg/sources.py) | Computes noise covariance, BEM models, sets up source spaces, and applies minimum-norm inverses. |
| **8** | **Source-Space Time-Frequency Power** | [`time_frequency.py`](meg_tokens/meg/time_frequency.py) | Extracts spectrograms (theta, alpha, beta, gamma) from source estimates using Morlet/multitapers. |
| **9** | **ERP Slicing, Parcellation, & Export** | [`erp.py`](meg_tokens/meg/erp.py) | Truncates trial waveforms, pads with NaNs, parcellates using cortical atlases, and exports `.npy` arrays with JSON sidecars. |
| **10** | **Group Statistics & Permutations** | [`stats.py`](meg_tokens/meg/stats.py) | Computes non-parametric permutation t-tests and spatio-temporal cluster significance. |

## 💾 Data Locations

> [!NOTE]
> The raw data files are large and are stored on external drives rather than tracked in this repository.

*   **Raw MEG Brain Recordings**
    *   `[Hamza Drive] /media/karim/Hamza/DDM-tthiery/`
    *   Contains raw CTF MEG datasets (`.ds` folders), digitized head shapes, and fiducial photos.
*   **Behavioral Logs (TDMS)**
    *   `[cc197cfe-12fc-4d55-b0a8-4f52a93ef003 Drive] /media/karim/cc197cfe-12fc-4d55-b0a8-4f52a93ef003/DDM/tdms/`
    *   Contains LabVIEW behavioral event logs for all 32 subjects (`H1` to `H32`).
*   **Parsed Behavioral Derivatives**
    *   Choose a BIDS derivatives root, for example `/path/to/tokens-bids/`.
    *   Stage 1 writes tables such as `derivatives/meg-tokens/sub-H01/beh/sub-H01_task-tokens_run-1_desc-slow_beh.tsv`.

---
*Note: This repository was refactored and organized starting 2026-06-25.*

## 💻 Running the Batch Pipelines (CLI)

The legacy scripts for extracting data, slicing ERPs, computing sources, and extracting power have all been unified into parameterized batch scripts located in `meg_tokens/utils/`. You can run them directly from your terminal.

### Stage 1: TDMS Behavioral Extraction
Extracts trial-by-trial logs from the raw LabVIEW directories into behavior TSV derivatives with JSON sidecars.
```bash
python -m meg_tokens.utils.batch_processor --input_dir /path/to/tdms/ --output_dir /path/to/tokens-bids/
```

### Stage 2: Epoch Extraction & Alignment
Filters raw continuous MEG data and slices it into event-locked trial epochs. Epoching consumes Stage 1 behavior TSV derivatives and cleaned/filtered raw FIF derivatives.
```bash
python -m meg_tokens.utils.batch_preprocess --raw_path /path/to/H01Slow1.ds --out_dir /path/to/tokens-bids/ --subject H01 --run Slow1
```

```bash
python -m meg_tokens.utils.batch_epochs --align_to go --subjects H01 H02 --raw_dir /path/to/tokens-bids/ --behavior_dir /path/to/tokens-bids/ --out_dir /path/to/tokens-bids/
```

### Stage 3: Behavioral Distributions & Metrics Plotting
Generates behavioral diagnostic plots, including decision time probability densities and brain-behavior scatterplots.
```bash
python -m meg_tokens.utils.batch_plot_behavior --subjects H01 H02 H03
```

### Stage 4: Neural Source Localization
Builds noise covariance, BEM, source-space, forward, inverse, and trial source-estimate derivatives for each subject.
```bash
python -m meg_tokens.utils.batch_sources \
  --subjects H01 H02 \
  --raw_dir /path/to/raw-or-noise/ \
  --epochs_dir /path/to/tokens-bids/ \
  --trans_dir /path/to/trans-files/ \
  --subjects_dir /path/to/freesurfer-subjects/ \
  --out_dir /path/to/tokens-bids/ \
  --run Slow1 \
  --align_to go \
  --spacing oct6
```

For legacy deep/volume analyses, request a mixed surface+volume source space
with FreeSurfer aseg labels. This writes a distinct
`desc-oct6-mixed_src.fif` derivative.

```bash
python -m meg_tokens.utils.batch_sources \
  --subjects H01 \
  --raw_dir /path/to/raw-or-noise/ \
  --epochs_dir /path/to/tokens-bids/ \
  --trans_dir /path/to/trans-files/ \
  --subjects_dir /path/to/freesurfer-subjects/ \
  --out_dir /path/to/tokens-bids/ \
  --run Slow1 \
  --align_to go \
  --spacing oct6 \
  --volume_labels Left-Putamen Right-Putamen Left-Caudate Right-Caudate
```

### Stage 5: Time-Frequency Power Extraction
Extracts source-space frequency-band power from the Stage 3 source-estimate manifest using sliding-window Hilbert, Morlet, or multitaper transforms.
```bash
python -m meg_tokens.utils.batch_time_frequency \
  --source_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --subjects H01 H02 \
  --run Slow1 \
  --align_to go \
  --source_method dSPM \
  --method hilbert \
  --bands alpha beta gamma_low \
  --width 400 \
  --step 110
```

### Stage 5b: Power Spectral Density & Specparam Modeling
Computes Welch or multitaper PSD on Stage 2 Epochs FIF derivatives and fits `specparam` models to separate periodic and aperiodic spectral structure. The command name keeps the old FOOOF label for compatibility, but the implementation uses `specparam` and writes `.npy`/`.tsv` derivatives with JSON sidecars.

```bash
python -m meg_tokens.utils.batch_psd_fooof \
  --epochs_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --subjects H01 H02 H03 \
  --condition Fast \
  --align_to go \
  --method welch \
  --fmin 1.0 \
  --fmax 100.0
```

### Stage 6: ERP Slicing & Parcellation
Slices Stage 3 source estimates relative to task events, pads Go-aligned trials before response, parcellates into cortical atlases, and writes trial-level `.npy` arrays with aligned trial metadata.
*(Note: This natively replicates legacy scripts like `08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py` that extracted condition-specific Trial Types arrays).*
```bash
python -m meg_tokens.utils.batch_erp_parcellation \
  --source_dir /path/to/tokens-bids/ \
  --behavior_dir /path/to/tokens-bids/ \
  --subjects_dir /path/to/freesurfer-subjects/ \
  --out_dir /path/to/tokens-bids/ \
  --subjects H01 H02 \
  --run Slow1 \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --max_duration_samples 400 \
  --min_rt_ms 100.0
```

All-source and volume source-coordinate exports use the same alignment and
trial metadata contract:

```bash
python -m meg_tokens.utils.batch_erp_parcellation \
  --source_dir /path/to/tokens-bids/ \
  --behavior_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --subjects H01 H02 \
  --run Slow1 \
  --align_to go \
  --source_method dSPM \
  --feature_space all_source
```

Use `--feature_space volume` for source estimates produced from mixed or volume
source spaces.

### Stage 7: Group-Level Statistics (Permutation T-Tests)
Runs a paired subject-level permutation t-test on Stage 6 parcellated ERP derivatives.
```bash
python -m meg_tokens.utils.batch_group_statistics \
  --erp_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --perms 1000
```

### Stage 7: Statistical Plotting & Correlations
Generates summary tables and selected label time-course figures from Stage 7 group-statistics derivatives. Optional behavior correlations read Stage 1 behavior TSV derivatives.
```bash
python -m meg_tokens.utils.batch_plot_statistics \
  --stats_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --top_n 8
```

```bash
python -m meg_tokens.utils.batch_plot_statistics \
  --stats_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --behavior_dir /path/to/tokens-bids/ \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --labels Label_1-lh Label_2-rh \
  --correlate_behavior
```

> **Tip:** You can append `--help` or `-h` to any of these commands to view all available path override arguments.

### Stage 8: Brain-Behavior Correlations
(Integrated directly into Stage 7 execution output. Automatically generated alongside stats).

### Stage 9: Time-Resolved MVPA Decoding (Classification)
Runs time-resolved Linear Discriminant Analysis over Stage 6 ERP/parcellation derivatives or Stage 4 source-power derivatives. Outputs are `.npy` arrays with JSON sidecars plus a decoding time-course figure.

```bash
python -m meg_tokens.utils.batch_decoding \
  --feature_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --feature_source erp \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --permutations 100
```

```bash
python -m meg_tokens.utils.batch_decoding \
  --feature_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --feature_source power \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --band alpha \
  --permutations 100
```

Trial-metadata decoding, such as sensory-evidence classes inside Fast/Slow runs, uses the Stage 6 `erptrials.tsv` metadata:

```bash
python -m meg_tokens.utils.batch_decoding \
  --feature_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --feature_source erp \
  --input_conditions Fast Slow \
  --conditions Easy Ambiguous Misleading \
  --class_column sTrialClass \
  --class_values 1 2 3 \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1
```

ROI and lateralized ROI wrappers call the same decoding engine:

```bash
python -m meg_tokens.utils.batch_decoding_roi \
  --feature_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --conditions Fast Slow \
  --roi Label_1-lh

python -m meg_tokens.utils.batch_decoding_lateralized \
  --feature_dir /path/to/tokens-bids/ \
  --out_dir /path/to/tokens-bids/ \
  --conditions Fast Slow
```

### Stage 9.5: PCA Trajectories and Loadings
This replaces the legacy `@nmData` MATLAB PCA trajectory framework (`Neural_space_Thomas_*.m`, `Neural_space_AL_all_sources.m`) and the PCA/LDA plotting notebooks.

The replicated behavior is: load real Stage 4 power or Stage 5 ERP derivatives, optionally select labels/ROIs, average trials into subject-level condition observations by default, fit PCA over condition-by-time samples, project condition means onto shared loadings, and save `.npy`/`.tsv` outputs with JSON sidecars.

```bash
python -m meg_tokens.utils.batch_dpca \
  --analysis pca \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --feature_source erp \
  --conditions Fast Slow \
  --subjects H01 H02 H03 \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --fit_time_range -1.25 0.70 \
  --n_components 20
```

For Stage 4 power derivatives:

```bash
python -m meg_tokens.utils.batch_dpca \
  --analysis pca \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --feature_source power \
  --conditions Correct Error \
  --band theta \
  --align_to feedback \
  --n_components 15
```

Useful options: `--labels`, `--lateralize`, `--average_unit trial`, `--transform sqrt` for non-negative power, and `--project_centered` if you explicitly want centered scikit-learn projections instead of the nmData-style raw projection.

The PCA stage writes `*_pcatrajectory.npy`, `*_pcaloadings.npy`, `*_pcavariance.npy`, `*_pcacondmeans.npy`, `*_pcafitscores.npy`, `*_pcafitsamples.tsv`, and `*_pcaobservations.tsv`.

```bash
python -m meg_tokens.utils.batch_plot_pca_trajectory \
  --timecourse_path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy \
  --out_dir /path/to/figures/pca_trajectory \
  --components 1 2 3

python -m meg_tokens.utils.batch_plot_component_timecourse \
  --timecourse_path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy \
  --out_dir /path/to/figures/pca_timecourses \
  --components 3

python -m meg_tokens.utils.batch_plot_pca_variance \
  --variance_path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcavariance.npy \
  --out_dir /path/to/figures/pca_variance
```

For source-space power PCA, loading sidecars carry source vertices when Stage 4 provided them:

```bash
python -m meg_tokens.utils.batch_plot_pca_loadings \
  --loadings_path /path/to/sub-group_task-tokens_desc-correct-vs-error-power-feedback-dSPM-theta-pca_pcaloadings.npy \
  --out_dir /path/to/figures/pca_loadings \
  --subjects_dir /path/to/freesurfer/subjects
```

### Stage 9.6: Optional Demixed PCA
`batch_dpca.py --analysis dpca` builds demixed-PCA tensors from real ERP derivatives and their `erptrials.tsv` sidecars. This mode requires the optional Python `dPCA` package.

```bash
python -m meg_tokens.utils.batch_dpca \
  --analysis dpca \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --conditions Fast Slow \
  --marginalize_cols sTrialClass nChoiceMade nCorrectChoice \
  --n_components 20

python -m meg_tokens.utils.batch_plot_dpca \
  --dpca_dir /path/to/tokens-bids \
  --out_dir /path/to/figures/dpca \
  --n_components 3
```

### Stage 10: Functional Connectivity
Extracts spectral connectivity between Stage 5 parcellated source time courses. This replaces `08_SRC_Connectivity.py` and `08_SRC_Connectivity_all2ROI.py` without writing full vertex-to-vertex matrices.

```bash
python -m meg_tokens.utils.batch_connectivity \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --align_to enter \
  --source_method dSPM \
  --parc HCPMMP1 \
  --method imcoh \
  --bands delta theta alpha beta \
  --before_window 0.7 1.4 \
  --after_window 1.6 2.3
```

#### Example 1: Circular Connectivity (Chord Diagrams) (Replicates `08_Plot_connectivity_circle.ipynb`)
To calculate subject-level permutation tests on active-minus-baseline connectivity and plot significant edges:

```bash
python -m meg_tokens.utils.batch_plot_connectivity_circle \
  --data_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --condition Fast \
  --band alpha \
  --p_threshold 0.05 \
  --perms 1000
```

#### Example 2: Seed-Based Spatial Connectivity Maps (Replicates `08_Seed_based_connectivity_final.ipynb`)
To extract the connectivity profile from one seed ROI to all other ROIs:

```bash
python -m meg_tokens.utils.batch_plot_seed_connectivity \
  --data_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --condition Fast \
  --band alpha \
  --seed_roi 17Networks_LH_SomMotA_1-lh \
  --p_threshold 0.05 \
  --perms 1000
```

### Stage 11: Hilbert Features for PAC/CFC
Extracts band-filtered signal, Hilbert amplitude, Hilbert power, and phase from Stage 5 parcellated source time courses. This modernizes the Brainpipe amplitude/power/sigfilt extraction visible in the legacy CFC notebooks and writes `.npy` arrays with JSON sidecars.

```bash
python -m meg_tokens.utils.batch_hilbert_features \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --bands theta alpha beta gamma_low \
  --features amplitude phase power sigfilt
```

### Stage 12: PAC/CFC Modulation Index
Computes final phase-amplitude coupling statistics from Stage 11 low-frequency phase and high-frequency amplitude derivatives.

```bash
python -m meg_tokens.utils.batch_pac_cfc \
  --feature_dir /path/to/tokens-bids \
  --out_dir /path/to/tokens-bids \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --phase_bands theta \
  --amplitude_bands gamma_low gamma_high \
  --align_to go \
  --source_method dSPM \
  --parc HCPMMP1 \
  --n_bins 18 \
  --time_window 0.0 1.5
```

### Golden Validation
Compares modern derivatives against frozen real-reference `.npy`, `.tsv`, or
`.csv` outputs. The command writes a validation report and exits nonzero if any
comparison fails.

```bash
python -m meg_tokens.utils.batch_validate_golden \
  --config /path/to/golden_validation.json \
  --out_tsv /path/to/tokens-bids/derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-golden-validation_validation.tsv
```

> ## SLURM Cluster Job Submission (`cluster/`)
> The `cluster/` directory contains modernized `.sh` wrappers designed to submit jobs to a SLURM high-performance computing cluster. 
> 
> Instead of having dozens of hard-coded shell scripts, we use unified parameterized scripts. For example, all `job_classif_*.sh` decoding scripts are now fully encapsulated by **`cluster/job_decoding.sh`**:
> 
> ```bash
> export TOKENS_BIDS=/path/to/tokens-bids
> export CONDITIONS="Fast Slow"
> export ALIGN_TO=go
> sbatch cluster/job_decoding.sh Label_1-lh
> ```
> If you run `sbatch cluster/job_decoding.sh` without arguments, it prints the required environment variables.
> 
> ### Submitting All ROIs Automatically
> The `cluster/submit_all_rois.sh` master script Replaces all 12 legacy `run_all_classif_*.sh` scripts. It iterates through all 360 HCPMMP1 brain regions and launches a parallel SLURM job for each region.
> 
> ```bash
> export TOKENS_BIDS=/path/to/tokens-bids
> export CONDITIONS="Fast Slow"
> ./cluster/submit_all_rois.sh
> ```
>
> ### SLURM Array Jobs (32 Subjects)
> We have replaced all the legacy subject-looping scripts (`run_all_power`, `run_all_sources`, `run_all_conn`, `run_all_resample_raw`) with extremely efficient SLURM Array Jobs. They will automatically parallelize execution across all 32 subjects (H01-H32):
> 
> ```bash
> export TOKENS_BIDS=/path/to/tokens-bids
> export HILBERT_BANDS="theta alpha beta"
> sbatch cluster/job_epochs.sh        # Replaces job_resample_raw.sh
> sbatch cluster/job_sources.sh       # Replaces job_sources.sh
> sbatch cluster/job_power.sh         # Replaces job_power.sh
> sbatch cluster/job_psd_specparam.sh
> sbatch cluster/job_connectivity.sh  # Replaces job_conn.sh
> sbatch cluster/job_hilbert_features.sh
> sbatch cluster/job_pac_cfc.sh
> sbatch cluster/job_golden_validation.sh
> ```
