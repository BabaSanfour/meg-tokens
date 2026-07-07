# MEG/iEEG Tokens Task Analysis (DDM Project)

This repository contains the analysis scripts and a clean, refactored Python library for investigating decision-making dynamics using the **Tokens Task** paired with Magnetoencephalography (MEG) and Intracranial EEG (iEEG).

## 🗂️ Project Structure

*   **`meg_tokens/`**: Main Python package with refactored, clean production code.
    *   `behavior/`: Modules for parsing behavior logs, calculating reaction times, and plotting performance.
    *   `meg/`: Modules for neural data preprocessing, ICA, and source localization.
    *   `utils/`: Helpers for reading TDMS files and I/O.
*   **`tests/`**: Unit tests.
*   **`pyproject.toml`**: Metadata and dependency configuration for the python package.
*   **`archive/`**: Contains the raw, unorganized scripts copied from the external drives:
    *   `DDM_scripts/`: Python/Jupyter notebooks (`scripts_new/`) and Matlab scripts (`matlab_scripts/`) copied from the `DDM_scripts` partition.
    *   `DDM_analysis_scripts/`: Jupyter notebooks copied from the `DDM/scripts/` partition.

## 🚀 Pipeline Execution Flow & Module Map

The analysis pipeline is designed to be executed sequentially from raw data ingestion to group-level parcellation and export. 

| Step | Stage | Module | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Behavioral Log Parsing** | [`tdms_parser.py`](meg_tokens/utils/tdms_parser.py) | Parses raw LabVIEW `.tdms` logs into tabular CSV format. |
| **2** | **Behavioral Metrics Extraction** | [`analysis.py`](meg_tokens/behavior/analysis.py) | Computes choice RTs, accuracy, difficulty levels, and formats behavioral variables. |
| **3** | **Behavioral Performance Plotting** | [`plotting.py`](meg_tokens/behavior/plotting.py) | Renders performance diagnostics, psychometric response curves, and RT distributions. |
| **4** | **MEG Preprocessing & Filtering** | [`preprocessing.py`](meg_tokens/meg/preprocessing.py) | Loads raw CTF MEG, applies filters, performs ICA decomposition, and coregisters head points. |
| **5** | **Epoch Extraction & Event Alignment** | [`epochs_builder.py`](meg_tokens/utils/epochs_builder.py) | Segments MEG data into trial-by-trial epochs aligned to triggers, filtered by behavior. |
| **6** | **Pipeline Automation & Batching** | [`batch_processor.py`](meg_tokens/utils/batch_processor.py) | Automates steps 1-5 in batch across blocks, runs, and subjects. |
| **7** | **Neural Source Localization** | [`sources.py`](meg_tokens/meg/sources.py) | Computes noise covariance, BEM models, sets up source spaces, and applies minimum-norm inverses. |
| **8** | **Source-Space Time-Frequency Power** | [`time_frequency.py`](meg_tokens/meg/time_frequency.py) | Extracts spectrograms (theta, alpha, beta, gamma) from source estimates using Morlet/multitapers. |
| **9** | **ERP Slicing, Parcellation, & Export** | [`erp.py`](meg_tokens/meg/erp.py) | Truncates trial waveforms, pads with NaNs, parcellates using cortical atlases, and exports to Numpy/MATLAB. |
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
*   **Behavioral Dataframes (CSVs)**
    *   `[cc197cfe-12fc-4d55-b0a8-4f52a93ef003 Drive] /media/karim/cc197cfe-12fc-4d55-b0a8-4f52a93ef003/DDM/dataframes/`
    *   Contains extracted behavior variables exported as CSV files.

---
*Note: This repository was refactored and organized starting 2026-06-25.*

## 💻 Running the Batch Pipelines (CLI)

The legacy scripts for extracting data, slicing ERPs, computing sources, and extracting power have all been unified into parameterized batch scripts located in `meg_tokens/utils/`. You can run them directly from your terminal.

### Stage 1: TDMS Behavioral Extraction
Extracts trial-by-trial logs from the raw LabVIEW directories into clean CSVs.
```bash
python -m meg_tokens.utils.batch_processor --input_dir /path/to/tdms/ --output_dir /path/to/dataframes/
```

### Stage 2: Epoch Extraction & Alignment
Filters raw continuous MEG data and slices it into event-locked trial epochs.
```bash
python -m meg_tokens.utils.batch_epochs --align_to go --subjects H01 H02
```

### Stage 3: Behavioral Distributions & Metrics Plotting
Generates behavioral diagnostic plots, including decision time probability densities and brain-behavior scatterplots.
```bash
python -m meg_tokens.utils.batch_plot_behavior --subjects H01 H02 H03
```

### Stage 4: Neural Source Localization
Builds BEM solutions, noise covariances, and source spaces for each subject.
```bash
python -m meg_tokens.utils.batch_sources --subjects H01 H02 --spacing oct6
```

### Stage 5: Time-Frequency Power Extraction
Extracts frequency-band power spectrograms from trial epochs using sliding-window Hilbert or Morlet transforms.
```bash
python -m meg_tokens.utils.batch_time_frequency --method hilbert --subjects H01 H02
```

### Stage 5: Power Spectral Density & FOOOF Modeling
Computes Welch or Multitaper PSD on epochs and fits the FOOOF model to separate periodic and aperiodic components.
```bash
python -m meg_tokens.utils.batch_psd_fooof --subjects H01 H02 H03 --method welch --fmin 1.0 --fmax 100.0
```

### Stage 6: ERP Slicing & Parcellation
Slices source estimates relative to task events, pads boundaries, and parcellates into cortical atlases.
*(Note: This natively replicates legacy scripts like `08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py` that extracted condition-specific Trial Types arrays).*
```bash
python -m meg_tokens.utils.batch_erp_parcellation --align_to go --min_rt_ms 100.0
```

### Stage 7: Group-Level Statistics (Permutation T-Tests)
Runs a permutation t-test on the parcellated time courses comparing two conditions.
```bash
python -m meg_tokens.utils.batch_group_statistics --conditions Fast Slow --perms 1000
```

### Stage 7: Statistical Plotting & Correlations
Generates brain topoplots of the significant t-values, lateralized timecourses (Contra vs Ipsi), and correlates neural peak latencies with behavior. You can run this for any contrast (e.g. Fast/Slow, Correct/Error, or 3+ Trial Types).
```bash
# Example 1: Fast vs Slow trials (Brain-Behavior correlations)
python -m meg_tokens.utils.batch_plot_statistics --conditions Fast Slow

# Example 2: Contralateral minus Ipsilateral motor preparation (Left vs Right choices)
python -m meg_tokens.utils.batch_plot_statistics --conditions Left Right --lateralized

# Example 3: Multiple conditions (Trial Types: Easy vs Ambiguous vs Misleading)
python -m meg_tokens.utils.batch_plot_statistics --conditions Easy Ambiguous Misleading
```

> **Tip:** You can append `--help` or `-h` to any of these commands to view all available path override arguments.

### Stage 8: Brain-Behavior Correlations
(Integrated directly into Stage 7 execution output. Automatically generated alongside stats).

### Stage 9: Time-Resolved MVPA Decoding (Classification)
Runs native Python inter-subject Machine Learning (Linear Discriminant Analysis) over time using a Sliding Estimator to decode conditions. This replaces all legacy MATLAB LDA scripts and disjointed Python compute scripts.

```bash
# 1. Decode Fast vs Slow trials (Replaces compute_classif_fast_vs_slow_*)
python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment enter
python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment go

# 2. Decode Sensory Evidence Parametrically (Replaces compute_classif_sensory_evidence_*)
python -m meg_tokens.utils.batch_decoding --conditions Easy Ambiguous Misleading --alignment enter
python -m meg_tokens.utils.batch_decoding --conditions Easy Ambiguous Misleading --alignment go

# 3. Decode Left vs Right Choice (Replaces compute_classif_lh_rh_* & _trial_class_*)
python -m meg_tokens.utils.batch_decoding --conditions Left Right --alignment enter
python -m meg_tokens.utils.batch_decoding --conditions Left Right --alignment go

# 4. Decode Baseline vs Task Timecourse (Replaces compute_classif_baseline_*)
python -m meg_tokens.utils.batch_decoding --compare_to_baseline --alignment enter
python -m meg_tokens.utils.batch_decoding --compare_to_baseline --alignment go

> **Flexibility: Sensor vs. Source Space & Powerbands**
> This unified pipeline is entirely agnostic to both the spatial and frequency domains. You can easily switch between them by pointing the `--data_dir` flag to the corresponding extracted features.
> 
> ```bash
> # Example A: Decode on Source Space (Cortical Parcellations)
> python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment enter --data_dir /path/to/source_space_data/
> 
> # Example B: Decode on Sensor Space (Replaces *Classif_in_time*.ipynb)
> python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment enter --data_dir /path/to/sensor_space_data/
> 
> # Example C: Decode on Raw Broadband signal
> python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment enter --data_dir /path/to/broadband_signal/
> 
> # Example D: Decode on isolated Alpha Powerband
> python -m meg_tokens.utils.batch_decoding --conditions Fast Slow --alignment enter --data_dir /path/to/alpha_power/
> 
> # Example E: Spatial Searchlight MVPA (Replaces Da_grp_topoplot*.ipynb & Classif_group_topoplot*.ipynb)
> # Trains independent univariate LDA classifiers on every sensor and renders a predictive topomap
> python -m meg_tokens.utils.batch_decoding_topoplot --conditions Fast Slow --permutations 99
> python -m meg_tokens.utils.batch_decoding_topoplot --conditions Left Right --permutations 99
> python -m meg_tokens.utils.batch_decoding_topoplot --conditions Correct Error --permutations 99
> python -m meg_tokens.utils.batch_decoding_topoplot --conditions Easy Difficult --permutations 99
> # Example F: Multi-Class Decoding (Replicates 08_Decoding_SRC_POWER_arrange_data.ipynb)
> # Supports any number of conditions natively (e.g. 3-way classifier for Trial Types)
> python -m meg_tokens.utils.batch_decoding --conditions Easy Ambiguous Misleading --permutations 100
> 
> # Example G: Decoding with Permutation Thresholds (Replicates 08_Decoding_SRC_POWER_Baseline_figures.ipynb)
> # Appending --permutations calculates and plots a rigorous p<0.05 threshold line on the accuracy graph
> python -m meg_tokens.utils.batch_decoding --compare_to_baseline --permutations 100
> 
> # Example H: Decoding Sensory Evidence (Replicates 08_Decoding_SRC_POWER_Sensory_evidence_figures.ipynb)
> # Passes the Sensory Evidence conditions into the same time-resolved pipeline
> python -m meg_tokens.utils.batch_decoding --conditions Easy Ambiguous Misleading --alignment enter --permutations 100
> 
> # Example I: Spatial Searchlight MVPA on Lateralized Sources (Replicates 09_Test_decoding_local.ipynb)
> # Computes (Left - Right) hemisphere activity and runs vertex-by-vertex time-resolved classification
> # This dynamically replaces the heavy cluster-based lateralized spatial decoding scripts.
> python -m meg_tokens.utils.batch_decoding_lateralized --conditions Easy Ambiguous Misleading --permutations 100
> 
> # Example J: Automated Decoding Latency (Onset) Bar Charts (Replicates 09_1st_moment... notebooks)
> # Extracts the first timepoint where accuracy exceeds the permutation threshold and plots it.
> python -m meg_tokens.utils.batch_plot_decoding_onset \
>     --scores decoding_scores_Easy.npy decoding_scores_Ambiguous.npy decoding_scores_Misleading.npy \
>     --thresholds decoding_threshold_Easy.npy decoding_threshold_Ambiguous.npy decoding_threshold_Misleading.npy \
>     --names Easy Ambiguous Misleading \
>     --time_offset -1000 --time_step 50
> # Example K: Time-Resolved ROI MVPA (Replicates 091_Stats_SRC_POWER... scripts)
> # Extracts data from a predefined parcellation (e.g. HCPMMP1), computes ROI lateralization, 
> # and runs time-resolved decoding individually per ROI.
> python -m meg_tokens.utils.batch_decoding_roi --parcellation HCPMMP1 --conditions Easy Ambiguous Misleading --permutations 100
> ```
> 
> > [!TIP]
> > **Dynamic Behavior Filtering:**
> > All decoding and stats batch scripts (`batch_stats_lateralized.py`, `batch_decoding_lateralized.py`, `batch_decoding_roi.py`) now support a `--behavior_filter` argument. This fully replaces the legacy `091_` "arrange_data" scripts by allowing you to dynamically slice your data using pandas queries directly on the MNE `Epochs.metadata`!
> >
> > Example 2: Run ROI decoding on Post-Error Slowing trials (requires 'previous_error' column in metadata):
> > ```bash
> > python -m meg_tokens.utils.batch_decoding_roi --parcellation HCPMMP1 --behavior_filter "previous_error == True"
> > ```
> 
> # Example L: Deep Volume Source Extraction (Replicates 091_...-DEEP.ipynb)
> # Extracts deep brain volume sources (vertices > 8196) and exports them to MATLAB
> python -m meg_tokens.utils.batch_extract_deep_sources --condition Easy
> ```
> 
> # Example M: ERP (Time-Domain) ROI Decoding (Replicates 091_...-ERP.ipynb)
> # By default, batch_decoding_roi loads 'epo.fif' (raw time-domain ERP data). 
> # You do not need a special flag to replicate the ERP script—just run standard ROI decoding!
> python -m meg_tokens.utils.batch_decoding_roi --parcellation HCPMMP1 --conditions Easy Ambiguous Misleading
> ```
> 
> **Permutation Testing:** You can append `--permutations 100` to any of the above commands to automatically shuffle the labels and calculate a rigorous `p<0.05` significance threshold overlaid on your decoding accuracy plots.

### Stage 9.5: Visualizing PCA / LDA Models (Replaces all 14 `00_Matlab_loadings_*` variants)
The legacy pipeline had 14 different `00_Matlab_loadings_brain_transfer_*.ipynb` notebooks hardcoded for specific conditions (e.g. Fast/Slow, Correct/Error, Go/Enter) and spatial ROIs. 
These have all been unified into Python scripts that read native `.npy` arrays. Below are 4 examples directly replicating the core legacy notebooks:

#### Example 1: PCA Variance Scree Plot (Replicates `00_Matlab_loadings_brain.ipynb`)
To visualize the cumulative variance explained by your principal components (extracted from `sklearn.decomposition.PCA().explained_variance_ratio_`):
```bash
python -m meg_tokens.utils.batch_plot_pca_variance \
    --variance_path /path/to/pca_variance_ratio.npy \
    --out_dir ./figures/pca_loadings/
```

#### Example 2: Temporal Components for Fast vs Slow (Replicates `...-LDA-DEEP.ipynb`, `...-LDA.ipynb`, `...-LDA_PLOT-Copy1.ipynb`, `...-LDA_PLOT.ipynb`, & `..._Confidence_Interval.ipynb`)
To plot the temporal dynamics (2D line plots) of your extracted PCA/LDA components for Fast vs Slow conditions:
```bash
python -m meg_tokens.utils.batch_plot_component_timecourse \
    --timecourse_path /path/to/component_timecourses.npy \
    --conditions Fast Slow \
    --components 3 \
    --out_dir ./figures/temporal_components/fast_slow/
```

#### Example 3: Temporal Components for Correct vs Error (Replicates `...-LDA-Error_Correct.ipynb` & `...-LDA_PLOT-Correct_Error.ipynb`)
To plot the temporal dynamics for Correct vs Error conditions:
```bash
python -m meg_tokens.utils.batch_plot_component_timecourse \
    --timecourse_path /path/to/component_timecourses.npy \
    --conditions Correct Error \
    --components 3 \
    --out_dir ./figures/temporal_components/correct_error/
```

#### Example 4: 3D Spatial Maps for ROIs (Replicates `...-LDA-ROIs.ipynb` & `...-LDA_PLOT-Subregions_ROIs.ipynb`)
To map spatial arrays (e.g., PCA loadings or classifier weights constrained to specific ROIs) onto an MNE PyVista 3D template brain:
```bash
python -m meg_tokens.utils.batch_plot_pca_loadings \
    --loadings_path /path/to/extracted_roi_spatial_weights.npy \
    --out_dir ./figures/spatial_weights/rois/
```

#### Example 5: 3D Spatial Maps for ERPs (Replicates `...-LDA_PLOT-ERP.ipynb`)
Because the spatial plotting script is agnostic to the data type, you can use the exact same tool to plot pure Event-Related Potential (ERP) spatial arrays (instead of PCA/LDA weights) onto the 3D brain:
```bash
python -m meg_tokens.utils.batch_plot_pca_loadings \
    --loadings_path /path/to/erp_spatial_activation.npy \
    --out_dir ./figures/erp_spatial_maps/
```

#### Example 6: 2D Heatmap of PCA Variance by ROI (Replicates `...-LDA_PLOT-DEEP-ROIs.ipynb`)
To plot a Seaborn heatmap visualizing how much variance each Principal Component explains across multiple specific Regions of Interest:
```bash
python -m meg_tokens.utils.batch_plot_pca_heatmap \
    --data_path /path/to/pca_roi_variance.npy \
    --rois Pallidum Caudate Putamen Amygdala Thalamus-Proper Cerebellum-Cortex Brain-Stem \
    --out_dir ./figures/pca_heatmaps/
```

#### Example 7: 3D State Space Trajectories & Loadings
> **Replaces:** Legacy `@nmData` MATLAB PCA trajectory framework (all `Neural_space_Thomas_*.m` variations)
> 
> Plot 3D Neural Trajectories and their spatial variance loadings across the brain. Our modern Python pipeline uses command-line arguments to replace what used to be a dozen copy-pasted MATLAB scripts.
> 
> ```bash
> # Replicates `Neural_space_Thomas_all_sources_correct_error.m` (Full Surface Space)
> python -m meg_tokens.utils.batch_dpca --conditions Correct Error
> python -m meg_tokens.utils.batch_plot_pca_trajectory --conditions Correct Error
> 
> # Replicates `Neural_space_Thomas_all_sources_ERP.m` and `all_sources.m` (ROI Constrained Surface)
> python -m meg_tokens.utils.batch_dpca --conditions enter --rois "DorsoLateral Prefrontal Cortex"
> 
> # Replicates `Neural_space_Thomas_all_sources_DEEP.m` and `deep_fast_slow.m` (Volumetric Subcortical Structures)
> python -m meg_tokens.utils.batch_dpca --conditions all_fast all_slow --rois Brain-Stem --volume
> ```
> 
> <details>
> <summary><b>List of 44 Regions of Interest (ROIs) analyzed by legacy scripts:</b></summary>
> 
> These are the 44 specific HCPMMP1 combined ROIs (22 per hemisphere) historically targeted by the legacy MATLAB scripts. You can pass any of these exact strings into the `--rois` argument.
> 
> *   `???` (Unassigned)
> *   `Anterior Cingulate and Medial Prefrontal Cortex`
> *   `Auditory Association Cortex`
> *   `Dorsal Stream Visual Cortex`
> *   `DorsoLateral Prefrontal Cortex`
> *   `Early Auditory Cortex`
> *   `Early Visual Cortex`
> *   `Inferior Frontal Cortex`
> *   `Inferior Parietal Cortex`
> *   `Insular and Frontal Opercular Cortex`
> *   `Lateral Temporal Cortex`
> *   `MT+ Complex and Neighboring Visual Areas`
> *   `Medial Temporal Cortex`
> *   `Orbital and Polar Frontal Cortex`
> *   `Paracentral Lobular and Mid Cingulate Cortex`
> *   `Posterior Cingulate Cortex`
> *   `Posterior Opercular Cortex`
> *   `Premotor Cortex`
> *   `Primary Visual Cortex (V1)`
> *   `Somatosensory and Motor Cortex`
> *   `Superior Parietal Cortex`
> *   `Temporo-Parieto-Occipital Junction`
> *   `Ventral Stream Visual Cortex`
> *(Note: The legacy scripts appended `-lh` and `-rh` to these to evaluate the 44 separate hemispheres).*
> </details>
> 
> ```bash
> # Standard Trajectory Plotting
> python -m meg_tokens.utils.batch_plot_pca_trajectory \
>     --timecourse_path /path/to/component_timecourses.npy \
>     --conditions Go Enter \
>     --out_dir ./figures/pca_trajectories/
> ```

#### Example 8: Thresholding and Exporting Active Regions (Replicates `00_Plot_brains_MNE_and_CSV.ipynb`)
To plot spatial arrays onto the brain while thresholding out weak activations (e.g., only keeping the top 5% most active vertices) and exporting those surviving regions to a `.csv` report:
```bash
python -m meg_tokens.utils.batch_plot_pca_loadings \
    --loadings_path /path/to/extracted_spatial_weights.npy \
    --threshold_percentile 95 \
    --export_csv \
    --out_dir ./figures/spatial_weights/thresholded/
```

### Stage 10: Functional Connectivity
Extracts source-space functional connectivity (e.g., Imaginary Coherence, Phase Locking Value) between cortical regions (ROIs) across time windows and frequency bands.

*(Note: This unified script replaces the legacy `08_SRC_Connectivity.py` and `08_SRC_Connectivity_all2ROI.py` by extracting ROI time-courses natively before computing the connectivity matrices, vastly improving memory and CPU performance).*

```bash
python -m meg_tokens.utils.batch_connectivity \
    --conditions Fast1 Slow1 \
    --parc aparc.a2009s \
    --out_dir ./connectivity_results/

# You can also change the connectivity metric using the --method flag
# For example, using Weighted Phase Lag Index (wpli2_debiased):
python -m meg_tokens.utils.batch_connectivity \
    --conditions Fast1 Slow1 \
    --method wpli2_debiased \
    --out_dir ./connectivity_results/
```

#### Example 1: Circular Connectivity (Chord Diagrams) (Replicates `08_Plot_connectivity_circle.ipynb`)
To calculate permutation t-tests on the functional connectivity matrices and plot the significant edges as a circular chord diagram:
```bash
python -m meg_tokens.utils.batch_plot_connectivity_circle \
    --data_dir ./connectivity_results/ \
    --condition Fast1 \
    --band alpha \
    --p_threshold 0.05 \
    --perms 1000
```

#### Example 2: Seed-Based Spatial Connectivity Maps (Replicates `08_Seed_based_connectivity_final.ipynb`)
To extract the connectivity profile from a single target "seed" region to the rest of the brain and map it onto a 3D cortical surface:
```bash
# Step 1: Extract the 1D spatial connectivity vector for the seed
python -m meg_tokens.utils.batch_plot_seed_connectivity \
    --seed_roi 17Networks_LH_SomMotA_1-lh \
    --condition Fast1 \
    --out_dir ./figures/connectivity_seeds/

# Step 2: Render that 1D vector onto the 3D PyVista brain
python -m meg_tokens.utils.batch_plot_pca_loadings \
    --loadings_path ./figures/connectivity_seeds/seed_connectivity_17Networks_LH_SomMotA_1-lh_Fast1_alpha.npy \
    --out_dir ./figures/spatial_weights/seed_maps/
```

### Stage 9.6: Demixed Principal Component Analysis (dPCA) (Replaces `092_Mixed_PCA-COMPUTE.ipynb`)
dPCA requires constructing a massive N-dimensional tensor representing every possible intersection of behavioral conditions. Our `batch_dpca.py` tool automates this natively in memory using Pandas metadata querying.

> ```bash
> # Example N: Time-Resolved Demixed PCA
> # Provide the metadata columns to group by. The tool dynamically constructs the N-dimensional 
> # tensor, runs dPCA, and automatically saves components and permutation significance masks.
> python -m meg_tokens.utils.batch_dpca \
>     --marginalize_cols sTrialClass nChoiceMade nCorrectChoice \
>     --labels 'dust' \
>     --n_components 20
> ```
> 
> ```bash
> # Example O: Plotting dPCA Components (Replaces 092_Mixed_PCA_PLOT.ipynb)
> # Reads the saved dPCA dictionaries and automatically generates shaded significance
> # plots for the top N components of every marginalized behavioral axis.
> python -m meg_tokens.utils.batch_plot_dpca \
>     --dpca_prefix ./dpca_results/dpca_sTrialClass_nChoiceMade_nCorrectChoice \
>     --n_components 3
> ```
> 
> ---
> 
> ## SLURM Cluster Job Submission (`cluster/`)
> The `cluster/` directory contains modernized `.sh` wrappers designed to submit jobs to a SLURM high-performance computing cluster. 
> 
> Instead of having dozens of hard-coded shell scripts, we use unified parameterized scripts. For example, all `job_classif_*.sh` decoding scripts are now fully encapsulated by **`cluster/job_decoding.sh`**:
> 
> ```bash
> # Submit a job to decode Fast vs Slow behavior on the "Enter" event
> sbatch cluster/job_decoding.sh "17Networks_LH_SomMotA_1-lh" "./data/epochs_enter/" "speed in ['Fast', 'Slow']"
> ```
> If you run `sbatch cluster/job_decoding.sh` without arguments, it will print a help menu showing exactly how to replicate all legacy `.sh` commands.
> 
> ### Submitting All ROIs Automatically
> The `cluster/submit_all_rois.sh` master script Replaces all 12 legacy `run_all_classif_*.sh` scripts. It iterates through all 360 HCPMMP1 brain regions and launches a parallel SLURM job for each region.
> 
> ```bash
> # Example: Submit 360 parallel decoding jobs (one for each ROI) across the cluster
> ./cluster/submit_all_rois.sh "./data/epochs_enter/" "speed in ['Fast', 'Slow']"
> ```
>
> ### SLURM Array Jobs (32 Subjects)
> We have replaced all the legacy subject-looping scripts (`run_all_power`, `run_all_sources`, `run_all_conn`, `run_all_resample_raw`) with extremely efficient SLURM Array Jobs. They will automatically parallelize execution across all 32 subjects (H01-H32):
> 
> ```bash
> sbatch cluster/job_epochs.sh        # Replaces job_resample_raw.sh
> sbatch cluster/job_sources.sh       # Replaces job_sources.sh
> sbatch cluster/job_power.sh         # Replaces job_power.sh
> sbatch cluster/job_connectivity.sh  # Replaces job_conn.sh
> ```
