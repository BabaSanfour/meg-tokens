# Legacy DDM Notebooks and Scripts: Stage Groupings

This document categorizes all legacy notebooks and scripts from the 2018 Decision-Making Dynamics (DDM) MEG project by processing stage. This serves as a master index to guide our modular refactoring and migration into the clean, tested `meg-tokens` library.

For the strict one-row-per-executable replication status, see [`docs/legacy_traceability.md`](docs/legacy_traceability.md).

---

## 🗺️ Stages Overview

Here is the 12-stage architecture of the legacy analysis pipeline:

1. **Stage 1: Dataframe Parsing** - TDMS behavioral parsing and CSV mapping.
2. **Stage 2: MEG Preprocessing & Trial Alignment** - Notch/bandpass filtering, ICA, headshape alignment, and trial epoching.
3. **Stage 3: Behavioral Data Analysis** - DT subtraction, Fast/Slow comparison, error vs. correct, and post-error slowing.
4. **Stage 4: Neural Source Reconstruction** - Volumetric/surface source space modeling, leadfield computing, and dSPM/sLORETA.
5. **Stage 5: Time-Frequency & Spectrograms** - Morlet Wavelet, Multitaper, and Hilbert envelope computations across bands.
6. **Stage 6: ERP Source-Space Aggregation** - Source-space Event-Related Potential downsampling, shifting, parcellation, and Matlab export.
7. **Stage 7: Group Statistics & Permutations** - Permutation t-tests and cluster permutations on source-space data.
8. **Stage 8: Brain-Behavior Correlations** - Correlations between peak neural commitment times, PC projections, and behavior.
9. **Stage 9: MVPA Decoding & Classification** - Time-resolved MVPA classifiers (Fast vs. Slow, Choice, Difficulty) on source space data.
10. **Stage 10: Functional Connectivity** - Seed-based connectivity and circular connectivity plots.
11. **Stage 11: Hilbert Features for PAC/CFC** - Band-filtered signal, phase, amplitude, and power extraction for downstream coupling analyses.
12. **Stage 12: PAC/CFC Statistics** - Phase-amplitude modulation-index estimation from staged Hilbert derivatives.

---

## 🔍 Stage 8: Brain-Behavior Correlations (In Focus)

This stage explores the relationships between neural features reconstructed in source space and behavioral metrics like Decision Time (DT).

### Active Legacy Files

| Notebook / Script | Description / Purpose | Key Dependencies | Status |
| :--- | :--- | :--- | :--- |
| [`00_Correlations_Peak_Commitment.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Correlations_Peak_Commitment.ipynb) | **Primary Correlation Analysis**:<br>1. Calculates subject-specific Reaction Times (RT) and Decision Times (DT = Button Press Time - RT).<br>2. Loads Matlab loading matrices (`*loadings_mean.mat`) and variance-explained files (`*nVarExpl.mat`) for target frequency bands (e.g., beta).<br>3. Loads source-space regional time-series data matrices (`.npy` files) for trial conditions (Easy/Ambiguous/Misleading, Fast/Slow, Left/Right target choices).<br>4. Multiplies loading vectors with source power values to project neural signals into demixed Principal Component (dPCA) space.<br>5. Correlates peak latency and commitment amplitudes with subject Decision Times and saves trajectory visualizations to PDF. | `numpy`, `pandas`, `scipy.io` (sio), `matplotlib`, `dPCA`, `visbrain`, `brainpipe` | **Replicated** (via `batch_plot_statistics.py`) |
| [`00_Matlab_loadings_brain.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain.ipynb) | Visualizes brain topography maps using PCA/dPCA spatial loading weights mapped onto MNE fsaverage brain structures. | `mne`, `matplotlib`, `scipy.io` | **Replicated** (via `batch_plot_statistics.py`) |
| [`00_Matlab_loadings_brain_transfer_all_subjects-LDA.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA.ipynb) | Evaluates transfer performance of Linear Discriminant Analysis (LDA) decision classifiers across all subjects. | `scikit-learn`, `numpy` | **Replicated** (via `batch_plot_statistics.py`) |
| [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT.ipynb) | Plots group classification accuracy curves and confidence intervals over time from the cross-subject LDA transfer models. | `matplotlib`, `seaborn` | **Replicated** (via `batch_plot_statistics.py`) |
| [`00_Matlab_loadings_brain_transfer_all_subjects_Confidence_Interval.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects_Confidence_Interval.ipynb) | Computes statistical bootstrap confidence intervals for the multi-subject classifier performance metrics. | `numpy`, `scipy.stats` | **Replicated** (via `batch_plot_statistics.py`) |

---

## 📂 Other Stages File Groupings

### Stage 1: Dataframe Parsing
*   [`Create_df.ipynb`](archive/replicated/DDM_analysis_scripts/Create_df.ipynb) *(Replicated)*
*   [`00_Create_df.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Create_df.ipynb) *(Replicated)*
*   [`00_Create_df-iEEG.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Create_df-iEEG.ipynb) *(Replicated)*
*   [`scripts_students/00_Create_df.ipynb`](archive/replicated/DDM_scripts/scripts_new/scripts_students/00_Create_df.ipynb) *(Replicated)*

### Stage 2: Preprocessing & Trial Alignment
*   [`Preprocessing.ipynb`](archive/replicated/DDM_analysis_scripts/Preprocessing.ipynb) *(Replicated)*
*   [`01_New_digitalized_pts.ipynb`](archive/replicated/DDM_scripts/scripts_new/01_New_digitalized_pts.ipynb) *(Replicated)*
*   [`02_Preprocessing.ipynb`](archive/replicated/DDM_scripts/scripts_new/02_Preprocessing.ipynb) *(Replicated)*
| [`03_Epochs_Button_press.ipynb`](archive/replicated/DDM_scripts/scripts_new/03_Epochs_Button_press.ipynb)<br>[`03_Epochs_Button_press-Students.ipynb`](archive/replicated/DDM_scripts/scripts_new/03_Epochs_Button_press-Students.ipynb)<br>[`Epochs_Button_press.ipynb`](archive/replicated/DDM_scripts/scripts_new/Epochs_Button_press.ipynb)<br>[`scripts_students/03_Epochs_Button_press.ipynb`](archive/replicated/DDM_scripts/scripts_new/scripts_students/03_Epochs_Button_press.ipynb) | Aligns triggers and saves MNE Epochs. | `mne`, `pandas` | **Replicated** |
| [`Create_new_digitalization_pts.py`](archive/replicated/DDM_scripts/scripts_new/Create_new_digitalization_pts.py) | **(Obsolete/Orphaned)** Legacy 2007 Eye-tracking parser (`.pos`) not part of core DDM pipeline. | `python` | **Replicated** |
*   [`44_Create_new_digitalization_pts.py`](archive/replicated/DDM_scripts/scripts_new/44_Create_new_digitalization_pts.py) *(Replicated)*
*   [`script_coreg_mne.py`](archive/replicated/DDM_scripts/scripts_new/script_coreg_mne.py) *(Replicated)*
*   [`smri_reconall.py`](archive/replicated/DDM_scripts/scripts_new/smri_reconall.py) *(Replicated)*
*   [`scripts_students/01_New_digitalized_pts.ipynb`](archive/replicated/DDM_scripts/scripts_new/scripts_students/01_New_digitalized_pts.ipynb) *(Replicated)*
*   [`scripts_students/02_Preprocessing.ipynb`](archive/replicated/DDM_scripts/scripts_new/scripts_students/02_Preprocessing.ipynb) *(Replicated)*
*   [`scripts_students/03_Epochs_Button_press.ipynb`](archive/replicated/DDM_scripts/scripts_new/scripts_students/03_Epochs_Button_press.ipynb) *(Replicated)*

### Stage 3: Behavioral Data Analysis
*   [`00_44_Behavior_Slow_Fast.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_44_Behavior_Slow_Fast.ipynb) *(Replicated)*
*   [`00_44_Behavior_Trial_Types.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_44_Behavior_Trial_Types.ipynb) *(Replicated)*
*   [`44_Behavior_DT_error_correct.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_DT_error_correct.ipynb) *(Replicated)*
*   [`44_Behavior_Fast_Slow_DT.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_Fast_Slow_DT.ipynb) *(Replicated)*
*   [`44_Behavior_Fast_Slow_DT-iEEG.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_Fast_Slow_DT-iEEG.ipynb) *(Replicated)*
*   [`44_Behavior_post_error_slowing.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_post_error_slowing.ipynb) *(Replicated)*
*   [`44_Behavior_Ratio_DT_PurcentCorrect.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_Ratio_DT_PurcentCorrect.ipynb) *(Replicated)*
*   [`44_Behavior_Trial_Class_DT-iEEG.ipynb`](archive/replicated/DDM_scripts/scripts_new/44_Behavior_Trial_Class_DT-iEEG.ipynb) *(Replicated)*
*   [`Behavior_Fast_Slow_DT_5v5.ipynb`](archive/replicated/DDM_scripts/scripts_new/Behavior_Fast_Slow_DT_5v5.ipynb) *(Replicated)*
*   [`Behavior_purcent_correct_5v5.ipynb`](archive/replicated/DDM_scripts/scripts_new/Behavior_purcent_correct_5v5.ipynb) *(Replicated)*
*   [`Behavior_Fast_Slow_DT.ipynb`](archive/replicated/DDM_analysis_scripts/Behavior_Fast_Slow_DT.ipynb) *(Replicated)*
*   [`Behavior_Trial_Class_DT.ipynb`](archive/replicated/DDM_analysis_scripts/Behavior_Trial_Class_DT.ipynb) *(Replicated)*
*   [`Separate_raw_by_conditions.ipynb`](archive/replicated/DDM_scripts/scripts_new/Separate_raw_by_conditions.ipynb) *(Replicated)*
*   [`Separate_raw_by_conditions-Copy1.ipynb`](archive/replicated/DDM_scripts/scripts_new/Separate_raw_by_conditions-Copy1.ipynb) *(Replicated)*
*   [`Separate_raw_by_conditions-LEFT_RIGHT_ONLY_MISLEADING.ipynb`](archive/replicated/DDM_scripts/scripts_new/Separate_raw_by_conditions-LEFT_RIGHT_ONLY_MISLEADING.ipynb) *(Replicated)*

### Stage 4: Neural Source Reconstruction
*   [`0000000_Compute_vol_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/0000000_Compute_vol_sources.ipynb) *(Replicated)*
*   [`04_compute_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources.ipynb) *(Replicated)*
*   [`04_compute_sources.py`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources.py) *(Replicated)*
*   [`04_compute_sources_aseg_local.py`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources_aseg_local.py) *(Replicated)*
*   [`04_compute_sources_avant_dernier.py`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources_avant_dernier.py) *(Replicated)*
*   [`04_compute_sources_local.py`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources_local.py) *(Replicated)*
*   [`04_compute_sources_old.py`](archive/replicated/DDM_scripts/scripts_new/04_compute_sources_old.py) *(Replicated)*
*   [`04_regroup_sources_local.py`](archive/replicated/DDM_scripts/scripts_new/04_regroup_sources_local.py) *(Replicated)*
*   [`55_source_reconstruction_annalisa.py`](archive/replicated/DDM_scripts/scripts_new/55_source_reconstruction_annalisa.py) *(Replicated)*
*   [`source_reconstruction.py`](archive/replicated/DDM_scripts/scripts_new/source_reconstruction.py) *(Replicated)*
*   [`source_reconstruction_parallel_group.py`](archive/replicated/DDM_scripts/scripts_new/source_reconstruction_parallel_group.py) *(Replicated)*
*   [`source_reconstruction_parallel_group_power.py`](archive/replicated/DDM_scripts/scripts_new/source_reconstruction_parallel_group_power.py) *(Replicated)*
*   [`source_reconstruction_parallel_group_power_5v5.py`](archive/replicated/DDM_scripts/scripts_new/source_reconstruction_parallel_group_power_5v5.py) *(Replicated)*
*   [`smri_params.py`](archive/replicated/DDM_scripts/scripts_new/smri_params.py) *(Replicated)*

### Stage 5: Time-Frequency & Spectrogram Analysis
*   [`05_compute_HG_multitaper.ipynb`](archive/replicated/DDM_scripts/scripts_new/05_compute_HG_multitaper.ipynb) *(Replicated)*
*   [`05_compute_power.ipynb`](archive/replicated/DDM_scripts/scripts_new/05_compute_power.ipynb) *(Replicated)*
*   [`05_compute_power.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_power.py) *(Replicated)*
*   [`05_compute_power_local.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_power_local.py) *(Replicated)*
*   [`05_compute_power_new.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_power_new.py) *(Replicated)*
*   [`05_compute_power_new_baseline.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_power_new_baseline.py) *(Replicated)*
*   [`05_compute_power_new_new.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_power_new_new.py) *(Replicated)*
*   [`05_compute_resample_RAW.ipynb`](archive/replicated/DDM_scripts/scripts_new/05_compute_resample_RAW.ipynb) *(Replicated)*
*   [`05_compute_resample_raw.py`](archive/replicated/DDM_scripts/scripts_new/05_compute_resample_raw.py) *(Replicated)*
*   [`05_regroup_sources_block.py`](archive/replicated/DDM_scripts/scripts_new/05_regroup_sources_block.py) *(Replicated)*
*   [`05_Time_Frequency_Maps.ipynb`](archive/replicated/DDM_scripts/scripts_new/05_Time_Frequency_Maps.ipynb) *(Replicated)*
*   [`0000_FOOF_AND_PSD.ipynb`](archive/replicated/DDM_scripts/scripts_new/0000_FOOF_AND_PSD.ipynb) *(Replicated/modernized - PSD extraction and aperiodic/periodic fitting now run through `batch_psd_fooof.py`, which consumes Stage 2 Epochs FIF derivatives, uses `specparam`, and writes `.npy`/`.tsv` outputs with JSON sidecars)*
*   [`Power_DDM-no_baseline.ipynb`](archive/replicated/DDM_scripts/scripts_new/Power_DDM-no_baseline.ipynb) *(Replicated - Third-party time-frequency analysis absorbed by batch_time_frequency.py in Stage 5)*
*   [`Filtered_signal_DDM.ipynb`](archive/replicated/DDM_scripts/scripts_new/Filtered_signal_DDM.ipynb) *(Replicated)*
*   [`Power_DDM_Lab1.ipynb`](archive/replicated/DDM_scripts/scripts_new/Power_DDM_Lab1.ipynb) *(Replicated)*

### SLURM Cluster Scripts
*   [`job_classif_baseline_enter.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_baseline_enter.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_baseline_go.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_baseline_go.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_fast_vs_slow_enter.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_fast_vs_slow_enter.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_fast_vs_slow_go.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_fast_vs_slow_go.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_lh_rh_enter.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_lh_rh_enter.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_lh_rh_go.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_lh_rh_go.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_sensory_evidence_enter.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_sensory_evidence_enter.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_sensory_evidence_enter_all_sources.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_sensory_evidence_enter_all_sources.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_sensory_evidence_go.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_sensory_evidence_go.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_sensory_evidence_go_all_sources.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_sensory_evidence_go_all_sources.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_trial_class_enter.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_trial_class_enter.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`job_classif_trial_class_go.sh`](archive/replicated/DDM_scripts/scripts_new/job_classif_trial_class_go.sh) *(Replicated - Replaced by parameterized cluster/job_decoding.sh)*
*   [`run_all_classif_baseline_enter.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_baseline_enter.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_baseline_go.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_baseline_go.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_fast_vs_slow_enter.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_fast_vs_slow_enter.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_fast_vs_slow_go.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_fast_vs_slow_go.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_lh_rh_enter.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_lh_rh_enter.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_lh_rh_go.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_lh_rh_go.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_sensory_evidence_enter.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_sensory_evidence_enter.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_sensory_evidence_enter_all_sources.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_sensory_evidence_enter_all_sources.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_sensory_evidence_go.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_sensory_evidence_go.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_sensory_evidence_go_all_sources.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_sensory_evidence_go_all_sources.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_trial_class_enter.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_trial_class_enter.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`run_all_classif_trial_class_go.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_classif_trial_class_go.sh) *(Replicated - Replaced by parameterized cluster/submit_all_rois.sh)*
*   [`job_conn.sh`](archive/replicated/DDM_scripts/scripts_new/job_conn.sh) *(Replicated - Replaced by cluster/job_connectivity.sh array job)*
*   [`run_all_conn.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_conn.sh) *(Replicated - Replaced by cluster/job_connectivity.sh array job)*
*   [`job_power.sh`](archive/replicated/DDM_scripts/scripts_new/job_power.sh) *(Replicated - Replaced by cluster/job_power.sh array job)*
*   [`run_all_power.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_power.sh) *(Replicated - Replaced by cluster/job_power.sh array job)*
*   [`job_sources.sh`](archive/replicated/DDM_scripts/scripts_new/job_sources.sh) *(Replicated - Replaced by cluster/job_sources.sh array job)*
*   [`run_all_sources.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_sources.sh) *(Replicated - Replaced by cluster/job_sources.sh array job)*
*   [`job_resample_raw.sh`](archive/replicated/DDM_scripts/scripts_new/job_resample_raw.sh) *(Replicated - Replaced by cluster/job_epochs.sh array job)*
*   [`run_all_resample_raw.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_resample_raw.sh) *(Replicated - Replaced by cluster/job_epochs.sh array job)*
*   [`job_conn_all2ROI.sh`](archive/replicated/DDM_scripts/scripts_new/job_conn_all2ROI.sh) *(Replicated - Replaced by unified cluster/job_connectivity.sh array job)*
*   [`run_all_conn_all2ROI.sh`](archive/replicated/DDM_scripts/scripts_new/run_all_conn_all2ROI.sh) *(Replicated - Replaced by unified cluster/job_connectivity.sh array job)*

## MATLAB Scripts
*   [`array2struct.m`](archive/replicated/DDM_scripts/matlab_scripts/array2struct.m) *(Replicated/Obsolete - Data wrangling to the legacy Spike/Firing-Rate structure used by the @nmData framework, completely replaced by native Python MNE/pandas)*
*   [`Neural_space_AL_all_sources.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_AL_all_sources.m) *(Replicated - PCA trajectory & loadings extraction now replaced by derivative-aware `batch_dpca.py --analysis pca`; plotting is handled by `batch_plot_pca_trajectory.py`, `batch_plot_component_timecourse.py`, `batch_plot_pca_variance.py`, and `batch_plot_pca_loadings.py`)*
*   [`NeuralSpaceSimulation_AN.m`](archive/replicated/DDM_scripts/matlab_scripts/NeuralSpaceSimulation_AN.m) *(Out of scope - modeling-only script from a 2006 paper, not related to MEG Tokens project data)*
*   [`Neural_space_Thomas_all_sources_correct_error.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_correct_error.m) *(Replicated - Correct vs Error PCA trajectory extraction replaced by `batch_dpca.py --analysis pca --conditions Correct Error` over Stage 4/5 derivatives)*
*   [`Neural_space_Thomas_all_sources_correct_error_ROIs.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_correct_error_ROIs.m) *(Replicated - ROI-constrained PCA is now `batch_dpca.py --analysis pca --labels ...`, using staged label coordinates instead of MATLAB vertex `.mat` masks)*
*   [`Neural_space_Thomas_all_sources_DEEP_enter.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_DEEP_enter.m) *(Replicated - deep/volume PCA trajectories are handled by Stage 4 power or Stage 5 ERP derivatives plus `batch_dpca.py --analysis pca`; no MATLAB `.mat` export is used)*
*   [`Neural_space_Thomas_all_sources_deep_fast_slow.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_deep_fast_slow.m) *(Replicated - Fast vs Slow deep-source trajectories replaced by derivative PCA with `--conditions Fast Slow` and appropriate staged deep features)*
*   [`Neural_space_Thomas_all_sources_DEEP_go.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_DEEP_go.m) *(Replicated - Go-aligned deep-source trajectories replaced by derivative PCA with `--align_to go`)*
*   [`Neural_space_Thomas_all_sources_DEEP.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_DEEP.m) *(Replicated - all-deep-source PCA trajectories replaced by derivative PCA over staged deep features)*
*   [`Neural_space_Thomas_all_sources_ERP.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_ERP.m) *(Replicated - DLPFC/beta trajectory extraction replaced by derivative PCA with `--feature_source power --band beta --labels ...`)*
*   [`Neural_space_Thomas_all_sources_raw.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources_raw.m) *(Replicated - broadband/source ERP trajectory extraction replaced by Stage 5 ERP derivatives plus `batch_dpca.py --analysis pca`)*
*   [`Neural_space_Thomas_all_sources.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas_all_sources.m) *(Replicated - Early Visual Cortex gamma-low trajectory extraction replaced by derivative PCA with `--feature_source power --band gamma_low --labels ...`)*
*   [`Neural_space_Thomas.m`](archive/replicated/DDM_scripts/matlab_scripts/Neural_space_Thomas.m) *(Replicated - the hardcoded ROI loop is replaced by repeated `batch_dpca.py --analysis pca --labels ...` calls over staged HCPMMP1 derivatives)*
*   [`Distribution_baseline_Fast_slow.ipynb`](archive/replicated/DDM_scripts/scripts_new/Distribution_baseline_Fast_slow.ipynb) *(Replicated)*
*   [`00_Correlations_Peak_Commitment.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Correlations_Peak_Commitment.ipynb) *(Replicated)*
*   [`00_plot_success_probability.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_plot_success_probability.ipynb) *(Replicated)*

### Stage 6: ERP Source-Space Aggregation (In Focus)

The legacy stage downsampled, trial-aligned, parcellated, and exported MATLAB `.mat` files. The modern replication keeps the behavioral operations but writes Stage 5 `.npy` arrays plus JSON sidecars and aligned `erptrials.tsv` metadata, so PCA, decoding, and statistics all consume the same derivative contract.

#### Active Legacy Files

| Notebook / Script | Description / Purpose | Key Dependencies | Status |
| :--- | :--- | :--- | :--- |
| [`00_ERP_Make_matlab_files_neural_space.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_ERP_Make_matlab_files_neural_space.ipynb) | **Primary ERP Aggregator & Parcellator**:<br>1. **Downsampling**: Loads source-space estimates (`-stc.h5`) for Go and Enter Target blocks and downsamples waveforms from 600 Hz to 100 Hz (`resample(..., down=6.0)`).<br>2. **NaN Alignment & Slicing**: Slices trial waveforms relative to event times (Go trigger and Enter Target choice). For Go-aligned trials, truncates the waveform at `tEnterTarget - 300ms` (motor preparation onset) and pads with NaNs up to 400 samples (4 seconds) to avoid motor artifact contamination.<br>3. **Glasser Parcellation**: Loads `lh.HCPMMP1.annot` and `rh.HCPMMP1.annot` labels, extracts time series for all 360 regions via `stc.in_label(label)`, and averages across vertices within each label.<br>4. **MATLAB Export**: Aggregates datasets across subjects and exports them as `.mat` files (`sio.savemat`). Also scales and merges frequency bands (delta * 3, theta * 6, alpha * 11.5, beta * 22.5, gamma_low * 45) into concatenated files. | `mne`, `numpy`, `pandas`, `scipy.io` (sio), `os` | **Replicated** (via [`erp.py`](meg_tokens/meg/erp.py)) |
| [`00_Make_matlab_files_neural_space.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Make_matlab_files_neural_space.ipynb) | Backup and alternative version of the main ERP aggregation and MATLAB export pipeline. | `mne`, `numpy`, `scipy.io` | **Replicated** (via [`erp.py`](meg_tokens/meg/erp.py)) |
| [`00_Make_matlab_files_neural_space-all_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Make_matlab_files_neural_space-all_sources.ipynb) | Variant that processes and exports unparcellated whole-brain source estimates (all 8,196 vertices) rather than region-of-interest labels. | `mne`, `numpy`, `scipy.io` | **Replicated** (via `batch_erp_parcellation.py --feature_space all_source`) |
| [`00_Make_matlab_files_neural_space-Copy1.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Make_matlab_files_neural_space-Copy1.ipynb) | Duplicate scratch version of the source-space matlab file creator. | `mne`, `numpy`, `scipy.io` | **Replicated** (via [`erp.py`](meg_tokens/meg/erp.py)) |
| [`Script_split_ERP.ipynb`](archive/replicated/DDM_scripts/scripts_new/Script_split_ERP.ipynb) | Slices and splits ERP time courses into pre-stimulus baseline and active decision periods. | `numpy`, `scipy.io` | **Replicated** (via [`erp.py`](meg_tokens/meg/erp.py)) |
| [`00_Prepare_ROIS_data.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Prepare_ROIS_data.ipynb) | Extracts structural vertex indices for subcortical volume labels from MNE source space and saves them as ROI masks. | `mne`, `numpy` | **Replicated** (via [`batch_extract_roi_masks.py`](meg_tokens/utils/batch_extract_roi_masks.py)) |
| [`08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py`](archive/replicated/DDM_scripts/scripts_new/08_Decoding_SRC_POWER_Trial_types_time_Trial_Types.py) | Slices, aligns to trials, and parcellates source estimates for MVPA trial type data structures. | `mne`, `numpy` | **Replicated** (via [`batch_erp_parcellation.py`](meg_tokens/utils/batch_erp_parcellation.py)) |


### Stage 7: Group Statistics & Permutations (In Focus)

This stage calculates group-level statistics (e.g., permutation t-tests, cluster permutations) and plots waveforms, timing significance, and cortical topographies.

#### Relationship to Stage 6:
*   The old Stage 6 (`00_ERP`) MATLAB export path has been replaced by staged `.npy`/`.tsv` derivatives. **Stage 7 Python notebooks** (e.g., `071_` and `06_` notebooks) represent the historical Python-based statistical inference pathway.
*   The `071_` notebooks downsample, trial-align, and parcellate source activity using the Destrieux (`aparc.a2009s`) atlas, saving the results in directories like `figures/time/left_minus_right_aparc/`.
*   The `06_` notebooks load these `.npy` files to run permutation t-tests, determine timing significance windowing, plot region-of-interest waveforms with SEM shading, and render cortical topoplots.

#### Active Legacy Files

| Notebook / Script | Description / Purpose | Key Dependencies | Status |
| :--- | :--- | :--- | :--- |
| [`071_SRC_ERP_left_minus_right_time-sem-Final_Trial_Types.ipynb`](archive/replicated/DDM_scripts/scripts_new/071_SRC_ERP_left_minus_right_time-sem-Final_Trial_Types.ipynb) | **Python Parcellation and Pipeline**: Extracts, aligns, downsamples, and parcellates trials (using Destrieux `aparc.a2009s.annot`) to output regional `.npy` files in Python, mirroring the `00_ERP` pipeline. | `mne`, `numpy`, `matplotlib` | **Replicated** |
| [`06_SRC_SIGNIF_TIMING_left_minus_right_time-ROI.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_SIGNIF_TIMING_left_minus_right_time-ROI.ipynb) | **Timing Significance Test**: Loads parcellated time courses and runs permutation t-tests over time windows to find windows of significant difference between left/right choices. | `mne`, `numpy`, `scipy` | **Replicated** |
| [`06_SRC_plot_left_right_ttest-ROI.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_left_right_ttest-ROI.ipynb)<br>[`06_SRC_plot_left_right_ttest.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_left_right_ttest.ipynb)<br>[`06_SRC_plot_left_right_ttest.py`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_left_right_ttest.py) | **ROI & Source Statistics**: Calculates and plots cluster/permutation t-tests for choice-aligned activity (Left vs. Right target choices). | `mne`, `matplotlib`, `visbrain` | **Replicated** |
| [`06_SRC_plot_correct_error_ttest.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_correct_error_ttest.ipynb)<br>[`06_SRC_plot_easy_ambiguous_ttest-ROI.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_easy_ambiguous_ttest-ROI.ipynb)<br>[`06_SRC_plot_easy_difficult_ttest.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_easy_difficult_ttest.ipynb)<br>[`06_SRC_plot_slow_fast_ttest.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_slow_fast_ttest.ipynb)<br>[`06_SRC_plot_slow_fast_ttest-ROI.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_slow_fast_ttest-ROI.ipynb) | **Condition T-Tests**: Runs t-test contrasts between trial conditions (Correct vs. Error, Easy vs. Difficult, Fast vs. Slow). | `mne`, `numpy`, `matplotlib` | **Replicated** |
| [`06_SRC_plot_left_right_correlations_timing.ipynb`](archive/replicated/DDM_scripts/scripts_new/06_SRC_plot_left_right_correlations_timing.ipynb) | Computes and plots the correlation between behavioral decision times and the onset/peak latency of the left-minus-right neural divergence. | `numpy`, `scipy.stats`, `matplotlib` | **Replicated** |
| [`00000_Error_Correct_ROIS_power_plots.ipynb`](archive/replicated/DDM_scripts/scripts_new/00000_Error_Correct_ROIS_power_plots.ipynb) | Restructures parcellated region-of-interest source power datasets for group comparison and plotting. | `numpy`, `pandas` | **Replicated** (via [`batch_plot_statistics.py`](meg_tokens/utils/batch_plot_statistics.py)) |
| [`EOG_Grp_Stats_in_time_Left_Right.ipynb`](archive/replicated/DDM_scripts/scripts_new/EOG_Grp_Stats_in_time_Left_Right.ipynb)<br>[`ERP_Grp_Stats_in_time_Correct_Error.ipynb`](archive/replicated/DDM_scripts/scripts_new/ERP_Grp_Stats_in_time_Correct_Error.ipynb)<br>[`ERP_Grp_Stats_in_time_Left_Right.ipynb`](archive/replicated/DDM_scripts/scripts_new/ERP_Grp_Stats_in_time_Left_Right.ipynb) | Computes grand average time-courses with standard error shading across all subjects for EOG and ERP sensor space signals. | `numpy`, `matplotlib` | **Replicated** |
| [`SRC_ERP_correct_minus_error_time.ipynb`](archive/replicated/DDM_scripts/scripts_new/SRC_ERP_correct_minus_error_time.ipynb)<br>[`SRC_ERP_easy_minus_difficult_time.ipynb`](archive/replicated/DDM_scripts/scripts_new/SRC_ERP_easy_minus_difficult_time.ipynb)<br>[`SRC_ERP_left_minus_right_time.ipynb`](archive/replicated/DDM_scripts/scripts_new/SRC_ERP_left_minus_right_time.ipynb) | Computes and plots the subtraction waveforms for source-space ERPs (e.g., Left-aligned minus Right-aligned choice waveforms). | `mne`, `numpy`, `matplotlib` | **Replicated** |
| [`Stats_in_time_Easy_Difficult.ipynb`](archive/replicated/DDM_scripts/scripts_new/Stats_in_time_Easy_Difficult.ipynb)<br>[`Stats_in_time_Fast_slow.ipynb`](archive/replicated/DDM_scripts/scripts_new/Stats_in_time_Fast_slow.ipynb)<br>[`Stats_in_time_Left_Right.ipynb`](archive/replicated/DDM_scripts/scripts_new/Stats_in_time_Left_Right.ipynb) | Windowed statistics scripts running cluster-based t-tests over time bins for primary contrasts. | `mne`, `numpy` | **Replicated** |
| [`07_SRC_POWER_*.ipynb`](archive/replicated/DDM_scripts/scripts_new/)<br>[`07_SRC_CORRELATIONS_*.ipynb`](archive/replicated/DDM_scripts/scripts_new/) | Group-level source power (time-frequency) statistics, mapping lateralization (Contra vs Ipsi), condition contrasts, and correlating peak latencies with behavioral decision times. | `mne`, `matplotlib`, `scipy` | **Replicated** |
| [`Time_Frequency_maps_correct_error.ipynb`](archive/replicated/DDM_scripts/scripts_new/Time_Frequency_maps_correct_error.ipynb)<br>[`Time_Frequency_maps_left_right_group.ipynb`](archive/replicated/DDM_scripts/scripts_new/Time_Frequency_maps_left_right_group.ipynb)<br>[`Time_Frequency_maps_easy_difficult.ipynb`](archive/replicated/DDM_scripts/scripts_new/Time_Frequency_maps_easy_difficult.ipynb)<br>[`Time_Frequency_maps_fast_slow-.ipynb`](archive/replicated/DDM_scripts/scripts_new/Time_Frequency_maps_fast_slow-.ipynb)<br>[`Time_Frequency_maps_left_right-.ipynb`](archive/replicated/DDM_scripts/scripts_new/Time_Frequency_maps_left_right-.ipynb) | Plots group time-frequency power spectrogram maps (TFR) across ROIs. | `mne`, `matplotlib` | **Replicated** |

### Stage 9: MVPA Decoding & Classification
*   [`compute_classif_*.py`](archive/replicated/DDM_scripts/scripts_new/) *(Replicated - Python LDA engines replaced by decoding.py)*
*   [`*Classif_in_time*.ipynb`](archive/replicated/DDM_scripts/scripts_new/) *(Replicated - Sensor Space MVPAs & Permutations replaced by batch_decoding.py)*
*   [`Da_grp_topoplot*.ipynb`, `Classif_group_topoplot*.ipynb`](archive/replicated/DDM_scripts/scripts_new/) *(Replicated - Spatial searchlight MVPAs & topoplots replaced by batch_decoding_topoplot.py)*
*   [`08_Decoding_SRC_POWER_arrange_data.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Decoding_SRC_POWER_arrange_data.ipynb) *(Replicated - Multi-class classification replaced by batch_decoding.py)*
*   [`08_Decoding_SRC_POWER_Baseline_figures.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Decoding_SRC_POWER_Baseline_figures.ipynb) *(Replicated - Permutation plotting replaced by batch_decoding.py)*
*   [`08_Decoding_SRC_POWER_Sensory_evidence_figures.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Decoding_SRC_POWER_Sensory_evidence_figures.ipynb) *(Replicated - Permutation plotting replaced by batch_decoding.py)*
*   [`09_Test_decoding_local.ipynb`](archive/replicated/DDM_scripts/scripts_new/09_Test_decoding_local.ipynb) *(Replicated - Lateralized spatial searchlight replaced by batch_decoding_lateralized.py)*
*   [`09_Decoding_SRC_POWER_arrange_data_all_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/09_Decoding_SRC_POWER_arrange_data_all_sources.ipynb) *(Replicated - MNE Epochs.metadata slicing logic absorbed into batch_decoding_lateralized.py)*
*   [`09_Figures_brain_decoding_all_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/09_Figures_brain_decoding_all_sources.ipynb) *(Replicated - Time-resolved 3D PyVista animations absorbed by batch_plot_pca_loadings.py --save_movie)*
*   [`091_Figures_brain_stats_all_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Figures_brain_stats_all_sources.ipynb) *(Replicated - Time-resolved 3D PyVista animations absorbed by batch_plot_pca_loadings.py --save_movie)*
*   [`091_Stats_SRC_POWER_arrange_data_all_sources.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources.ipynb) *(Replicated - MVPA on Parcellation ROIs replaced by batch_decoding_roi.py)*
*   [`091_Stats_SRC_POWER_arrange_data_all_sources-DEEP.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources-DEEP.ipynb) *(Replicated - Deep volume source extraction to MATLAB replaced by mixed source spaces from `batch_sources.py --volume_labels ...` plus `batch_erp_parcellation.py --feature_space volume`)*
*   [`091_Stats_SRC_POWER_arrange_data_all_sources-_correct_error.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources-_correct_error.ipynb)<br>[`091_Stats_SRC_POWER_arrange_data_all_sources-ERP.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources-ERP.ipynb)<br>[`091_Stats_SRC_POWER_arrange_data_all_sources-_post_error_slowing.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources-_post_error_slowing.ipynb) *(Replicated - Manual CSV behavior slicing replaced by shared trial metadata, behavior columns, and `batch_erp_parcellation.py --feature_space all_source`)*
*   [`09_1st_moment_signif_DT_trial_types.ipynb`](archive/replicated/DDM_scripts/scripts_new/09_1st_moment_signif_DT_trial_types.ipynb)<br>[`09_1st_moment_signif_DT_trial_types-Fast_Slow.ipynb`](archive/replicated/DDM_scripts/scripts_new/09_1st_moment_signif_DT_trial_types-Fast_Slow.ipynb) *(Replicated - Manual bar charting replaced by batch_plot_decoding_onset.py)*
*   [`00_Plot_brains_MNE_and_CSV.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Plot_brains_MNE_and_CSV.ipynb) *(Replicated - replaced by batch_plot_pca_loadings.py --export_csv)*
*   [`00_Matlab_loadings_brain.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain.ipynb) *(Replicated - replaced by batch_plot_pca_variance.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA-DEEP.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA-DEEP.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA-Error_Correct.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA-Error_Correct.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA-ROIs.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA-ROIs.ipynb) *(Replicated - replaced by batch_plot_pca_loadings.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Copy1.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Copy1.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Correct_Error.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Correct_Error.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-DEEP.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-DEEP.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-DEEP-ROIs.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-DEEP-ROIs.ipynb) *(Replicated - replaced by batch_plot_pca_heatmap.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-ROIs.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-ROIs.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Subregions_ROIs.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-Subregions_ROIs.ipynb) *(Replicated - replaced by batch_plot_pca_loadings.py)*
*   [`00_Matlab_loadings_brain_transfer_go_enter.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_go_enter.ipynb) *(Replicated - replaced by batch_plot_pca_trajectory.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects_Confidence_Interval.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects_Confidence_Interval.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT.ipynb) *(Replicated - replaced by batch_plot_component_timecourse.py)*
*   [`091_Stats_SRC_POWER_arrange_data_all_sources-ERP.ipynb`](archive/replicated/DDM_scripts/scripts_new/091_Stats_SRC_POWER_arrange_data_all_sources-ERP.ipynb) *(Replicated)*
*   [`00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-ERP.ipynb`](archive/replicated/DDM_scripts/scripts_new/00_Matlab_loadings_brain_transfer_all_subjects-LDA_PLOT-ERP.ipynb) *(Replicated - replaced by batch_plot_pca_loadings.py)*
*   [`Grp_Classif_in_time_Correct_Error.ipynb`](archive/replicated/DDM_scripts/scripts_new/Grp_Classif_in_time_Correct_Error.ipynb) *(Replicated)*
*   [`Grp_Classif_in_time_easy_Trial_class.ipynb`](archive/replicated/DDM_scripts/scripts_new/Grp_Classif_in_time_easy_Trial_class.ipynb) *(Replicated)*
*   [`Grp_Classif_in_time_Fast_slow.ipynb`](archive/replicated/DDM_scripts/scripts_new/Grp_Classif_in_time_Fast_slow.ipynb) *(Replicated)*
*   [`Grp_Classif_in_time_Fast_slow-5vs5.ipynb`](archive/replicated/DDM_scripts/scripts_new/Grp_Classif_in_time_Fast_slow-5vs5.ipynb) *(Replicated)*
*   [`Grp_Classif_in_time_Left_Right.ipynb`](archive/replicated/DDM_scripts/scripts_new/Grp_Classif_in_time_Left_Right.ipynb) *(Replicated)*
*   [`Modify_df_preproc.ipynb`](archive/replicated/DDM_scripts/scripts_new/Modify_df_preproc.ipynb) *(Replicated)*
*   [`Portfolio_deep_learning.ipynb`](archive/replicated/DDM_scripts/scripts_new/Portfolio_deep_learning.ipynb) *(Replicated)*

### Stage 9.2: Demixed Principal Component Analysis
*   [`092_Mixed_PCA-COMPUTE.ipynb`](archive/replicated/DDM_scripts/scripts_new/092_Mixed_PCA-COMPUTE.ipynb) *(Replicated - tensor construction now reads real ERP derivatives and `erptrials.tsv` sidecars via `batch_dpca.py --analysis dpca`; outputs are `.npy`/`.tsv` derivatives with JSON sidecars)*
*   [`092_Mixed_PCA_PLOT.ipynb`](archive/replicated/DDM_scripts/scripts_new/092_Mixed_PCA_PLOT.ipynb) *(Replicated - Matplotlib component plotting replaced by batch_plot_dpca.py)*

### Stage 10: Functional Connectivity
*   [`08_SRC_Connectivity.py`](archive/replicated/DDM_scripts/scripts_new/08_SRC_Connectivity.py)<br>[`08_SRC_Connectivity.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_SRC_Connectivity.ipynb) *(Replicated - before/after spectral connectivity is now computed directly over Stage 5 ROI time-course derivatives by `batch_connectivity.py`, avoiding legacy full vertex-to-vertex intermediates)*
*   [`08_SRC_Connectivity_all2ROI.py`](archive/replicated/DDM_scripts/scripts_new/08_SRC_Connectivity_all2ROI.py) *(Replicated/modernized - ROI aggregation is handled upstream by Stage 5 parcellation, so Stage 10 consumes label time courses directly)*
*   [`08_Plot_connectivity_circle.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Plot_connectivity_circle.ipynb)<br>[`08_Plot_connectivity_circle-subject-by-subject.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Plot_connectivity_circle-subject-by-subject.ipynb) *(Replicated - `batch_plot_connectivity_circle.py` reads Stage 10 derivatives, averages runs within subject, and writes sidecar-backed stats plus figures)*
*   [`08_Seed_based_connectivity_final.ipynb`](archive/replicated/DDM_scripts/scripts_new/08_Seed_based_connectivity_final.ipynb)<br>[`08_SRC_Connectivity_seed_based_plot.py`](archive/replicated/DDM_scripts/scripts_new/08_SRC_Connectivity_seed_based_plot.py) *(Replicated - `batch_plot_seed_connectivity.py` writes seed-to-all node vectors, t-statistics, p-values, and null distributions as derivatives)*
*   **Alpha-Band Seed Connectivity**: Refer to [`Untitled10.ipynb`](archive/replicated/DDM_scripts/scripts_new/Untitled10.ipynb) and [`Untitled6.ipynb`](archive/replicated/DDM_scripts/scripts_new/Untitled6.ipynb) for seed-based spectral connectivity computing.
*   **Cross-Frequency Coupling (CFC)**: [`Untitled.ipynb`](archive/replicated/DDM_analysis_scripts/Untitled.ipynb) contains Brainpipe band-filtered signal, amplitude, and power extraction. This behavior is implemented by `batch_hilbert_features.py`, and final modulation-index PAC/CFC statistics are implemented by `batch_pac_cfc.py`.

### Stage 11: Hilbert Features for PAC/CFC
*   [`Untitled.ipynb`](archive/replicated/DDM_analysis_scripts/Untitled.ipynb) *(Replicated/modernized for the MEG Tokens path - the real-data Brainpipe `power`, `amplitude`, and `sigfilt` export behavior is replaced by `batch_hilbert_features.py`, which reads Stage 5 ERP/parcellation derivatives and writes sidecar-backed `.npy` Hilbert features. The old notebook's non-DDM autism example is out of project scope.)*

### Stage 12: PAC/CFC Statistics
*   [`Untitled.ipynb`](archive/replicated/DDM_analysis_scripts/Untitled.ipynb) *(Replicated/modernized - phase-amplitude coupling statistics are now computed by `batch_pac_cfc.py` using Tort-style modulation index over Stage 11 phase and amplitude derivatives.)*

### Uncategorized / Scratch Files (Ignore or Cleanup)
*   [`Untitled*.ipynb`](archive/replicated/DDM_scripts/scripts_new/) (Several scratch notebooks: `Untitled.ipynb` through `Untitled15.ipynb`, `Untitled-Copy1.ipynb`, etc.)
