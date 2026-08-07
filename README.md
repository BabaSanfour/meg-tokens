# MEG Tokens Task Analysis (DDM Project)

This repository contains the analysis scripts and a refactored Python library for investigating decision-making dynamics using the **Tokens Task** paired with Magnetoencephalography (MEG). iEEG notebooks in the archive are out of scope for this refactor.

## Project Structure

*   **`meg_tokens/`**: Main Python package with refactored, clean production code.
    *   `core/`: Canonical subject/run identifiers, project configuration, and workflow result models.
    *   `behavior/`: TDMS parsing and behavioral metrics.
    *   `meg/`: Modules for neural data preprocessing, ICA, and source localization.
    *   `features/`: ERP, power, spectral, Hilbert, PAC, and connectivity estimators.
    *   `analysis/`: Statistics, decoding, PCA, and dPCA estimators.
    *   `workflows/`: Filesystem-aware processing and analysis stages.
    *   `reports/`: Figure and report-table generation.
    *   `cli/`: The unified `meg-tokens` command.
    *   `io/`: Central derivative layout and `.npy`/JSON, xarray, and table contracts.
*   **`workflow/`**: Snakemake DAG and local/Slurm profiles.
*   **`tests/`**: Unit tests.
*   **`docs/`**: Data contracts and refactor notes, including the strict one-row-per-file legacy mapping in [`docs/legacy_traceability.md`](docs/legacy_traceability.md) and the post-replication architecture contract in [`docs/refactor/architecture.md`](docs/refactor/architecture.md).
*   **`pyproject.toml`**: Metadata and dependency configuration for the python package.
*   **`archive/`**: Contains the raw, unorganized scripts copied from the external drives:
    *   `DDM_scripts/`: Python/Jupyter notebooks (`scripts_new/`) and Matlab scripts (`matlab_scripts/`) copied from the `DDM_scripts` partition.
    *   `DDM_analysis_scripts/`: Jupyter notebooks copied from the `DDM/scripts/` partition.

## Pipeline Execution Flow & Module Map

The analysis pipeline is designed to be executed sequentially from raw data ingestion to group-level parcellation and export. 

| Step | Stage | Module | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Behavioral Log Parsing** | [`tdms.py`](meg_tokens/behavior/tdms.py) | Parses raw LabVIEW `.tdms` logs into BIDS-derivatives-style behavior TSV tables with JSON sidecars. |
| **2** | **Behavioral Metrics Extraction** | [`metrics.py`](meg_tokens/behavior/metrics.py) | Computes choice RTs, accuracy, difficulty levels, and behavioral summaries. |
| **3** | **Behavioral Reporting** | [`behavior.py`](meg_tokens/reports/behavior.py) | Renders performance diagnostics and RT distributions. |
| **4** | **MEG Preprocessing & Filtering** | [`preprocessing.py`](meg_tokens/meg/preprocessing.py) | Loads raw CTF MEG, applies filters, performs ICA decomposition, and coregisters head points. |
| **5** | **Epoch Extraction & Event Alignment** | [`epoching.py`](meg_tokens/meg/epoching.py) | Segments MEG data into trial epochs aligned to triggers and behavior. |
| **6** | **Workflow Orchestration** | [`Snakefile`](workflow/Snakefile) | Schedules subjects, runs, shared source models, features, analyses, and reports. |
| **7** | **Neural Source Localization** | [`sources.py`](meg_tokens/meg/sources.py) | Computes noise covariance, BEM models, sets up source spaces, and applies minimum-norm inverses. |
| **8** | **Source-Space Time-Frequency Power** | [`time_frequency.py`](meg_tokens/features/time_frequency.py) | Extracts band power from source estimates using modern MNE calls. |
| **9** | **ERP Slicing, Parcellation, & Export** | [`erp.py`](meg_tokens/features/erp.py) | Aligns, pads, parcellates, and exports labeled `.npy` arrays with JSON sidecars. |
| **10** | **Group Analyses** | [`analysis/`](meg_tokens/analysis) | Provides permutation statistics, decoding, PCA, and dPCA estimators. |

## 💾 Data Locations

> [!NOTE]
> The raw data files are large and are stored outside this repository.

All project data lives under one convention data root, `meg-tokens`:

*   **`raw/`** — Raw MEG brain recordings.
    *   Contains raw CTF MEG datasets (`.ds` folders), digitized head shapes, and fiducial photos.
*   **`tdms/`** — Raw behavioral logs (TDMS).
    *   Contains LabVIEW behavioral event logs for all 32 subjects (`H1` to `H32`).
*   **`BIDS/`** — Parsed and derived outputs (where the pipeline writes).

Set `data_root` once in a TOML file based on 
[`config/tokens.toml.template`](config/tokens.toml.template) and `raw_meg_root`, `behavior_root`,
and `bids_root` default to `data_root/raw`, `data_root/tdms`, and `data_root/BIDS` respectively
— no per-field paths to keep in sync. Any of the three can still be overridden individually 
(e.g. if raw MEG stays on a separate external drive) by uncommenting the matching line in the 
template. Relative paths are resolved from the configuration file location.

---
*Note: This repository was refactored and organized starting 2026-06-25.*

## Running the Pipeline (CLI)

The installed `meg-tokens` command is the supported execution API. All stages
also have callable workflow functions under `meg_tokens.workflows`.

### Stage 1: TDMS Behavioral Extraction
Extracts trial-by-trial logs from the raw LabVIEW directories into behavior TSV derivatives with JSON sidecars.
```bash
meg-tokens --config tokens.toml behavior ingest
```

Compute subject summaries, paired group statistics, and the trial-level
MEG-joinable feature table from those staged TSV files:

```bash
meg-tokens --config tokens.toml behavior analyze
```

Re-run the source-log success-probability/SPD validation and print its summary:

```bash
meg-tokens --config tokens.toml behavior qc
```


### Stage 2: Epoch Extraction & Alignment
Filters raw continuous MEG data and slices it into event-locked trial epochs. Epoching consumes Stage 1 behavior TSV derivatives and cleaned/filtered raw FIF derivatives.
```bash
meg-tokens --config tokens.toml meg preprocess \
  --raw-path /path/to/H01Slow1.ds \
  --subject H01 \
  --run Slow1
```

```bash
meg-tokens --config tokens.toml meg epoch \
  --alignment go \
  --subjects H01 H02
```

### Stage 3: Behavioral Distributions & Metrics Plotting
Generates behavioral diagnostic plots, including decision time probability densities and brain-behavior scatterplots.
```bash
meg-tokens --config tokens.toml report behavior --subjects H01 H02 H03
```

### Stage 4: Neural Source Localization
Builds noise covariance, BEM, source-space, forward, inverse, and trial source-estimate derivatives for each subject.
```bash
meg-tokens --config tokens.toml meg source \
  --subjects H01 H02 \
  --run Slow1 \
  --alignment go \
  --spacing oct6
```

For legacy deep/volume analyses, request a mixed surface+volume source space
with FreeSurfer aseg labels. This writes a distinct
`desc-oct6-mixed_src.fif` derivative.

```bash
meg-tokens --config tokens.toml meg source \
  --subjects H01 \
  --run Slow1 \
  --alignment go \
  --spacing oct6 \
  --volume-labels Left-Putamen Right-Putamen Left-Caudate Right-Caudate
```

### Stage 5: Time-Frequency Power Extraction
Extracts source-space frequency-band power from the Stage 3 source-estimate manifest using sliding-window Hilbert, Morlet, or multitaper transforms.
```bash
meg-tokens --config tokens.toml features power \
  --subjects H01 H02 \
  --run Slow1 \
  --alignment go \
  --source-method dSPM \
  --method hilbert \
  --bands alpha beta gamma_low \
  --width 400 \
  --step 110
```

### Stage 5b: Power Spectral Density & Specparam Modeling
Computes Welch or multitaper PSD on Stage 2 Epochs FIF derivatives and fits `specparam` models to separate periodic and aperiodic spectral structure, writing `.npy`/`.tsv` derivatives with JSON sidecars.

```bash
meg-tokens --config tokens.toml features spectral \
  --subjects H01 H02 H03 \
  --condition Fast \
  --alignment go \
  --method welch \
  --fmin 1.0 \
  --fmax 100.0
```

### Stage 6: ERP Slicing & Parcellation
Slices Stage 3 source estimates relative to task events, pads Go-aligned trials before response, parcellates into cortical atlases, and writes trial-level `.npy` arrays with aligned trial metadata.
*(Note: This natively replicates legacy scripts like `08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py` that extracted condition-specific Trial Types arrays).*
```bash
meg-tokens --config tokens.toml features erp \
  --subjects H01 H02 \
  --run Slow1 \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --max-duration-samples 400 \
  --min-rt-ms 100.0
```

All-source and volume source-coordinate exports use the same alignment and
trial metadata contract:

```bash
meg-tokens --config tokens.toml features erp \
  --subjects H01 H02 \
  --run Slow1 \
  --alignment go \
  --source-method dSPM \
  --feature-space all_source
```

Use `--feature-space volume` for source estimates produced from mixed or volume
source spaces.

### Stage 7: Group-Level Statistics (Permutation T-Tests)
Runs a paired subject-level permutation t-test on Stage 6 parcellated ERP derivatives.
```bash
meg-tokens --config tokens.toml analyze statistics \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --permutations 1000
```

For a one-condition hemisphere test, homologous parcellation labels are
subtracted left-minus-right before the group permutation test:

```bash
meg-tokens --config tokens.toml analyze lateralization \
  --condition Easy \
  --subjects H01 H02 H03 \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --permutations 1000
```

### Stage 7: Statistical Plotting & Correlations
Generates summary tables and selected label time-course figures from Stage 7 group-statistics derivatives. Optional behavior correlations read Stage 1 behavior TSV derivatives.
```bash
meg-tokens --config tokens.toml report statistics \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --top-n 8
```

```bash
meg-tokens --config tokens.toml report statistics \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --labels Label_1-lh Label_2-rh \
  --correlate-behavior
```

> **Tip:** You can append `--help` or `-h` to any of these commands to view all available path override arguments.

### Stage 8: Brain-Behavior Correlations
(Integrated directly into Stage 7 execution output. Automatically generated alongside stats).

### Stage 9: Time-Resolved MVPA Decoding (Classification)
Runs time-resolved Linear Discriminant Analysis over Stage 6 ERP/parcellation derivatives or Stage 4 source-power derivatives. Outputs are `.npy` arrays with JSON sidecars plus a decoding time-course figure.

```bash
meg-tokens --config tokens.toml analyze decoding \
  --feature-source erp \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --permutations 100
```

```bash
meg-tokens --config tokens.toml analyze decoding \
  --feature-source power \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --band alpha \
  --permutations 100
```

Trial-metadata decoding, such as sensory-evidence classes inside Fast/Slow runs, uses the Stage 6 `erptrials.tsv` metadata:

```bash
meg-tokens --config tokens.toml analyze decoding \
  --feature-source erp \
  --input-conditions Fast Slow \
  --conditions Easy Ambiguous Misleading \
  --class-column sTrialClass \
  --class-values 1 2 3 \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1
```

ROI and lateralized analyses are configurations of the same decoding workflow:

```bash
meg-tokens --config tokens.toml analyze decoding \
  --conditions Fast Slow \
  --roi Label_1-lh

meg-tokens --config tokens.toml analyze decoding \
  --conditions Fast Slow \
  --lateralize
```

### Stage 9.5: PCA Trajectories and Loadings
This replaces the legacy `@nmData` MATLAB PCA trajectory framework (`Neural_space_Thomas_*.m`, `Neural_space_AL_all_sources.m`) and the PCA/LDA plotting notebooks.

The replicated behavior is: load real Stage 4 power or Stage 5 ERP derivatives, optionally select labels/ROIs, average trials into subject-level condition observations by default, fit PCA over condition-by-time samples, project condition means onto shared loadings, and save `.npy`/`.tsv` outputs with JSON sidecars.

```bash
meg-tokens --config tokens.toml analyze decomposition \
  --analysis pca \
  --feature-source erp \
  --conditions Fast Slow \
  --subjects H01 H02 H03 \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --fit-time-range -1.25 0.70 \
  --n-components 20
```

For Stage 4 power derivatives:

```bash
meg-tokens --config tokens.toml analyze decomposition \
  --analysis pca \
  --feature-source power \
  --conditions Correct Error \
  --band theta \
  --alignment feedback \
  --n-components 15
```

Useful options: `--labels`, `--lateralize`, `--average-unit trial`, `--transform sqrt` for non-negative power, and `--project-centered` if you explicitly want centered scikit-learn projections instead of the nmData-style raw projection.

The PCA stage writes `*_pcatrajectory.npy`, `*_pcaloadings.npy`, `*_pcavariance.npy`, `*_pcacondmeans.npy`, `*_pcafitscores.npy`, `*_pcafitsamples.tsv`, and `*_pcaobservations.tsv`.

```bash
meg-tokens --config tokens.toml report pca-trajectory \
  --timecourse-path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy \
  --components 1 2 3

meg-tokens --config tokens.toml report pca-timecourse \
  --timecourse-path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcatrajectory.npy \
  --components 3

meg-tokens --config tokens.toml report pca-variance \
  --variance-path /path/to/sub-group_task-tokens_desc-fast-vs-slow-erp-go-dSPM-HCPMMP1-pca_pcavariance.npy
```

For source-space power PCA, loading sidecars carry source vertices when Stage 4 provided them:

```bash
meg-tokens --config tokens.toml report pca-loadings \
  --loadings-path /path/to/sub-group_task-tokens_desc-correct-vs-error-power-feedback-dSPM-theta-pca_pcaloadings.npy \
  --subjects-dir /path/to/freesurfer/subjects
```

### Stage 9.6: Optional Demixed PCA
`meg-tokens analyze decomposition --analysis dpca` builds demixed-PCA tensors from real ERP derivatives and their `erptrials.tsv` sidecars. This mode requires the optional Python `dPCA` package.

```bash
meg-tokens --config tokens.toml analyze decomposition \
  --analysis dpca \
  --conditions Fast Slow \
  --marginalize-cols sTrialClass nChoiceMade nCorrectChoice \
  --n-components 20

meg-tokens --config tokens.toml report dpca \
  --dpca-root /path/to/tokens-bids \
  --n-components 3
```

### Stage 10: Functional Connectivity
Extracts spectral connectivity between Stage 5 parcellated source time courses. This replaces `08_SRC_Connectivity.py` and `08_SRC_Connectivity_all2ROI.py` without writing full vertex-to-vertex matrices.

```bash
meg-tokens --config tokens.toml features connectivity \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --alignment enter \
  --source-method dSPM \
  --parc HCPMMP1 \
  --method imcoh \
  --bands delta theta alpha beta \
  --before-window 0.7 1.4 \
  --after-window 1.6 2.3
```

#### Example 1: Circular Connectivity (Chord Diagrams) (Replicates `08_Plot_connectivity_circle.ipynb`)
To calculate subject-level permutation tests on active-minus-baseline connectivity and plot significant edges:

```bash
meg-tokens --config tokens.toml report connectivity \
  --condition Fast \
  --band alpha \
  --p-threshold 0.05 \
  --permutations 1000
```

#### Example 2: Seed-Based Spatial Connectivity Maps (Replicates `08_Seed_based_connectivity_final.ipynb`)
To extract the connectivity profile from one seed ROI to all other ROIs:

```bash
meg-tokens --config tokens.toml report seed-connectivity \
  --condition Fast \
  --band alpha \
  --seed-roi 17Networks_LH_SomMotA_1-lh \
  --p-threshold 0.05 \
  --permutations 1000
```

### Stage 11: Hilbert Features for PAC/CFC
Extracts band-filtered signal, Hilbert amplitude, Hilbert power, and phase from Stage 5 parcellated source time courses. This modernizes the Brainpipe amplitude/power/sigfilt extraction visible in the legacy CFC notebooks and writes `.npy` arrays with JSON sidecars.

```bash
meg-tokens --config tokens.toml features hilbert \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --bands theta alpha beta gamma_low \
  --feature-types amplitude phase power sigfilt
```

### Stage 12: PAC/CFC Modulation Index
Computes final phase-amplitude coupling statistics from Stage 11 low-frequency phase and high-frequency amplitude derivatives.

```bash
meg-tokens --config tokens.toml features pac \
  --subjects H01 H02 H03 \
  --conditions Fast Slow \
  --phase-bands theta \
  --amplitude-bands gamma_low gamma_high \
  --alignment go \
  --source-method dSPM \
  --parc HCPMMP1 \
  --n-bins 18 \
  --time-window 0.0 1.5
```

### Golden Validation
Compares modern derivatives against frozen real-reference `.npy`, `.tsv`, or
`.csv` outputs. The command writes a validation report and exits nonzero if any
comparison fails.

```bash
meg-tokens --config tokens.toml validate golden \
  --comparison-config /path/to/golden_validation.json \
  --out-tsv /path/to/tokens-bids/derivatives/meg-tokens/sub-group/meg/sub-group_task-tokens_desc-golden-validation_validation.tsv
```

## Workflow Orchestration

Snakemake owns local and Slurm execution. Edit `workflow/config.yaml` to list
the real subjects, runs, raw-data template, and analysis settings; keep path
roots in the referenced `config/tokens.toml`.

```bash
pip install -e '.[workflow]'

# Inspect the complete DAG without running data.
snakemake --snakefile workflow/Snakefile \
  --configfile workflow/config.yaml \
  --profile workflow/profiles/local \
  --dry-run

# Run locally.
./cluster/run_workflow_local.sh workflow/config.yaml

# Submit jobs through the Snakemake Slurm executor.
./cluster/submit_workflow.sh workflow/config.yaml
```

Shared covariance, BEM, and source-space models are separate subject-level
rules. Run-level forward/inverse/source estimates depend on those models, so
parallel execution does not rebuild or overwrite shared files.
