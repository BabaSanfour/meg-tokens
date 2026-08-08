# MEG Tokens Task Analysis (DDM Project)

Analysis library for decision-making dynamics on the **Tokens Task** with MEG. `archive/` iEEG notebooks are out of scope.

## Project Structure

*   **`meg_tokens/`**: package.
    *   `core/`: subject/run identifiers, project config, workflow result models.
    *   `behavior/`: TDMS parsing, behavioral metrics.
    *   `meg/`: preprocessing, ICA, source localization.
    *   `features/`: ERP, power, spectral, Hilbert, PAC, connectivity estimators.
    *   `analysis/`: statistics, decoding, PCA, dPCA estimators.
    *   `workflows/`: filesystem-aware processing/analysis stages.
    *   `reports/`: figures, report tables.
    *   `cli/`: the `meg-tokens` command.
    *   `io/`: derivative layout, `.npy`/JSON/xarray/table contracts.
*   **`workflow/`**: Snakemake DAG, local/Slurm profiles.
*   **`tests/`**: unit tests.
*   **`docs/`**: data contracts, refactor notes -- [`legacy_traceability.md`](docs/legacy_traceability.md) (old-script mapping), [`refactor/architecture.md`](docs/refactor/architecture.md) (architecture contract).
*   **`pyproject.toml`**: package metadata/deps.
*   **`archive/`**: raw scripts copied from the external drives -- `DDM_scripts/` (`scripts_new/` Python/Jupyter + `matlab_scripts/`), `DDM_analysis_scripts/` (Jupyter, from `DDM/scripts/`).

## Pipeline Execution Flow & Module Map

Module map, in pipeline order. Stage numbers live only under [Running the Pipeline](#running-the-pipeline-cli), not here.

| Stage | Module | Purpose |
| :--- | :--- | :--- |
| **Raw BIDSification** | [`raw_staging.py`](meg_tokens/meg/raw_staging.py), [`meg_bids.py`](meg_tokens/meg/meg_bids.py), [`anat_bids.py`](meg_tokens/meg/anat_bids.py), [`tdms_bids.py`](meg_tokens/behavior/tdms_bids.py) | Identifies which raw session is which run, then stages the whole raw layer: `BIDS/sub-*/{meg,beh,anat}` and `sub-emptyroom`. |
| **Behavioral Log Parsing** | [`classification.py`](meg_tokens/behavior/classification.py), [`schema.py`](meg_tokens/behavior/schema.py), [`behavior_ingest.py`](meg_tokens/workflows/behavior_ingest.py) | Reads the staged raw behavior, infers trial classes, validates the Stage 1 contract, and writes behavior derivatives. |
| **Behavioral Metrics Extraction** | [`features.py`](meg_tokens/behavior/features.py) and [`performance.py`](meg_tokens/behavior/analyses/performance.py) | Computes choice RTs, accuracy, difficulty levels, and behavioral summaries. |
| **Behavioral Characterization Analyses** | [`behavior_characterization.py`](meg_tokens/workflows/behavior_characterization.py) and [`behavior/analyses/`](meg_tokens/behavior/analyses) | Runs the full `docs/behavior_analysis_roadmap.md` battery over the trial-feature table: distributions, design/session effects, evidence, criterion decline, individual differences, and more. |
| **Behavioral Reporting** | [`behavior.py`](meg_tokens/reports/behavior.py) | Renders performance diagnostics and RT distributions. |
| **MEG Preprocessing & Filtering** | [`preprocessing.py`](meg_tokens/meg/preprocessing.py) | Loads raw CTF MEG, applies filters, performs ICA decomposition, and coregisters head points. |
| **Epoch Extraction & Event Alignment** | [`epoching.py`](meg_tokens/meg/epoching.py) | Segments MEG data into trial epochs aligned to triggers and behavior. |
| **Neural Source Localization** | [`sources.py`](meg_tokens/meg/sources.py) | Computes noise covariance, BEM models, sets up source spaces, and applies minimum-norm inverses. |
| **Source-Space Time-Frequency Power** | [`time_frequency.py`](meg_tokens/features/time_frequency.py) | Extracts band power from source estimates using modern MNE calls. |
| **ERP Slicing, Parcellation, & Export** | [`erp.py`](meg_tokens/features/erp.py) | Aligns, pads, parcellates, and exports labeled `.npy` arrays with JSON sidecars. |
| **Group Analyses** | [`analysis/`](meg_tokens/analysis) | Provides permutation statistics, decoding, PCA, and dPCA estimators. |
| **Workflow Orchestration** | [`Snakefile`](workflow/Snakefile) | Schedules subjects, runs, shared source models, features, analyses, and reports. |

## 💾 Data Locations

> [!NOTE]
> Raw data lives outside this repo.

Data root `meg-tokens`:

*   **`raw/`** — raw CTF sessions (`.ds`), `NOISE_noise_*.ds` empty-room, headshapes, fiducial
    photos. Read-only; only copied into `BIDS/` (`meg-tokens meg stage-raw`).
*   **`tdms/`** — LabVIEW behavioral logs, all 32 subjects (`H1`-`H32`).
*   **`BIDS/`** — Stage 0's raw layer (`sub-*/meg`, `sub-*/beh`, `sub-*/anat`,
    `sub-emptyroom`) + `BIDS/derivatives/`.
*   **`IRM/`** — FreeSurfer `recon-all` output (30/32; H07, H10 missing). Default
    `subjects_dir` (`data_root/IRM`); override in the project TOML.

`data_root` set once in a TOML ([`config/tokens.toml.template`](config/tokens.toml.template)),
passed via `--config`; `raw_meg_root`/`behavior_root`/`bids_root`/`subjects_dir` derive from it.
Relative paths resolve from the config file's location.

---
*Refactored/organized starting 2026-06-25.*

## Running the Pipeline (CLI)

`meg-tokens` is the supported execution API; every stage also has a callable workflow
function under `meg_tokens.workflows`.

### Stage 0: Raw BIDSification
Plans the raw layer per subject (MEG runs, empty-room, headshape, anat) into a reviewable
manifest. Reads `raw/`, `tdms/`, `subjects_dir`; no Stage 1 dependency; never touches `BIDS/`:
```bash
meg-tokens --config tokens.toml meg stage-raw --subjects H01 H02
```

Matching: **inter-trial-interval fingerprint** — logged `nInitialTime` gaps vs. real MEG
trial-start pulse gaps, the same physical intervals on two unrelated clocks. Accepted only
within an error threshold and a margin over the runner-up, with no session claimed twice;
otherwise `review`.

Resolve a flagged row: edit the manifest's `source_path`/`action`, re-run
`apply-raw-staging` (applies the manifest as saved, never recomputes). Applying copies staged
rows into `BIDS/sub-*/{meg,beh,anat}` and `BIDS/sub-emptyroom`; originals under `raw/`,
`tdms/`, `IRM/` are untouched:
```bash
meg-tokens --config tokens.toml meg apply-raw-staging --subjects H01 H02
```

### Stage 1: Behavioral Log Parsing
Builds trial tables from `BIDS/sub-*/beh/` (trial-class inference + validation). Requires
Stage 0.
```bash
meg-tokens --config tokens.toml behavior ingest
```

### Stage 2: Behavioral Metrics Extraction
Computes subject summaries, paired group statistics, and the trial-level
MEG-joinable feature table from Stage 1's staged TSV files:

```bash
meg-tokens --config tokens.toml behavior analyze
```

### Stage 2b: Behavioral Characterization Analyses
Runs the full battery of behavioral analyses from `docs/behavior_analysis_roadmap.md`
over Stage 2's trial-feature table — distributions, condition × class effects, session
drift, lapses, continuous evidence, criterion decline and urgency, reverse
correlation, conditional accuracy, trial history, and individual differences.
Each writes its own group derivative; results on the current dataset are in
`docs/behavior_roadmap_results.md`. Requires Stage 2 to have run first:

```bash
meg-tokens --config tokens.toml behavior characterization
```

**Behavior QC** (not a pipeline stage — a standalone validation utility):
re-run the source-log success-probability/SPD validation and print its summary:

```bash
meg-tokens --config tokens.toml behavior qc
```


### Stage 3: MEG Preprocessing & Filtering
Loads raw CTF MEG, applies filters, and optionally runs ICA:
```bash
meg-tokens --config tokens.toml meg preprocess \
  --raw-path /path/to/H01Slow1.ds \
  --subject H01 \
  --run Slow1
```

### Stage 4: Epoch Extraction & Event Alignment
Slices Stage 3's filtered continuous MEG into event-locked trial epochs, aligned to Stage 1 behavior TSV derivatives:
```bash
meg-tokens --config tokens.toml meg epoch \
  --alignment go \
  --subjects H01 H02
```

### Stage 5: Behavioral Distributions & Metrics Plotting
Generates behavioral diagnostic plots, including decision time probability densities and brain-behavior scatterplots.
```bash
meg-tokens --config tokens.toml report behavior --subjects H01 H02 H03
```

### Stage 6: Neural Source Localization
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

### Stage 7: Time-Frequency Power Extraction
Extracts source-space frequency-band power from the Stage 6 source-estimate manifest using sliding-window Hilbert, Morlet, or multitaper transforms.
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

### Stage 7b: Power Spectral Density & Specparam Modeling
Computes Welch or multitaper PSD on Stage 4 Epochs FIF derivatives and fits `specparam` models to separate periodic and aperiodic spectral structure, writing `.npy`/`.tsv` derivatives with JSON sidecars.

```bash
meg-tokens --config tokens.toml features spectral \
  --subjects H01 H02 H03 \
  --condition Fast \
  --alignment go \
  --method welch \
  --fmin 1.0 \
  --fmax 100.0
```

### Stage 8: ERP Slicing & Parcellation
Slices Stage 6 source estimates relative to task events, pads Go-aligned trials before response, parcellates into cortical atlases, and writes trial-level `.npy` arrays with aligned trial metadata.
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

### Stage 9: Group-Level Statistics (Permutation T-Tests)
Runs a paired subject-level permutation t-test on Stage 8 parcellated ERP derivatives.
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

### Stage 10: Statistical Plotting & Correlations
Generates summary tables and selected label time-course figures from Stage 9 group-statistics derivatives. Optional behavior correlations read Stage 1 behavior TSV derivatives.
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

### Stage 11: Brain-Behavior Correlations
(Integrated directly into Stage 10 execution output. Automatically generated alongside stats).

### Stage 12: Time-Resolved MVPA Decoding (Classification)
Runs time-resolved Linear Discriminant Analysis over Stage 8 ERP/parcellation derivatives or Stage 7 source-power derivatives. Outputs are `.npy` arrays with JSON sidecars plus a decoding time-course figure.

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

Trial-metadata decoding, such as sensory-evidence classes inside Fast/Slow runs, uses the Stage 8 `erptrials.tsv` metadata:

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

### Stage 12.5: PCA Trajectories and Loadings
This replaces the legacy `@nmData` MATLAB PCA trajectory framework (`Neural_space_Thomas_*.m`, `Neural_space_AL_all_sources.m`) and the PCA/LDA plotting notebooks.

The replicated behavior is: load real Stage 7 power or Stage 8 ERP derivatives, optionally select labels/ROIs, average trials into subject-level condition observations by default, fit PCA over condition-by-time samples, project condition means onto shared loadings, and save `.npy`/`.tsv` outputs with JSON sidecars.

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

For Stage 7 power derivatives:

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

For source-space power PCA, loading sidecars carry source vertices when Stage 6 provided them:

```bash
meg-tokens --config tokens.toml report pca-loadings \
  --loadings-path /path/to/sub-group_task-tokens_desc-correct-vs-error-power-feedback-dSPM-theta-pca_pcaloadings.npy \
  --subjects-dir /path/to/freesurfer/subjects
```

### Stage 12.6: Optional Demixed PCA
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

### Stage 13: Functional Connectivity
Extracts spectral connectivity between Stage 8 parcellated source time courses. This replaces `08_SRC_Connectivity.py` and `08_SRC_Connectivity_all2ROI.py` without writing full vertex-to-vertex matrices.

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

### Stage 14: Hilbert Features for PAC/CFC
Extracts band-filtered signal, Hilbert amplitude, Hilbert power, and phase from Stage 8 parcellated source time courses. This modernizes the Brainpipe amplitude/power/sigfilt extraction visible in the legacy CFC notebooks and writes `.npy` arrays with JSON sidecars.

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

### Stage 15: PAC/CFC Modulation Index
Computes final phase-amplitude coupling statistics from Stage 14 low-frequency phase and high-frequency amplitude derivatives.

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
  --out-tsv /path/to/tokens-bids/derivatives/sub-group/meg/sub-group_task-tokens_desc-golden-validation_validation.tsv
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
