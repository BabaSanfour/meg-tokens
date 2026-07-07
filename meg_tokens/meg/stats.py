import numpy as np
import mne
from mne.stats import permutation_t_test, spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test

def compute_permutation_t_test(
    data: np.ndarray,
    n_permutations: int = 1000,
    tail: int = 0,
    n_jobs: int = 1
) -> tuple:
    """
    Computes a non-parametric permutation t-test on the data array.
    This is useful for comparing a difference contrast against zero.
    
    Note: Legacy `SRC_` scripts often computed contrasts as a relative percentage 
    difference (e.g. `(Cond A - Cond B) / Cond B * 100`) rather than a raw difference 
    (`Cond A - Cond B`). Ensure the `data` array is normalized appropriately if replicating 
    those exact visualizations.
    
    Args:
        data: Array of shape (n_observations, n_features). 
              For source space, usually (n_subjects, n_vertices).
        n_permutations: Number of permutations to compute.
        tail: 0 for two-tailed, 1 for upper tail, -1 for lower tail.
        n_jobs: Number of parallel jobs.
        
    Returns:
        Tuple of (T_obs, p_values, H0)
        - T_obs: T-statistic observed for all variables.
        - p_values: P-values for all variables.
        - H0: Permutation distribution of the max statistic.
    """
    T_obs, p_values, H0 = permutation_t_test(
        data, 
        n_permutations=n_permutations, 
        tail=tail, 
        n_jobs=n_jobs
    )
    return T_obs, p_values, H0

def compute_cluster_permutation_test(
    data: np.ndarray,
    threshold: float = None,
    n_permutations: int = 1000,
    tail: int = 0,
    adjacency=None,
    n_jobs: int = 1
) -> tuple:
    """
    Computes a spatio-temporal cluster permutation 1-sample test.
    This tests whether the data (typically a difference matrix of shape 
    (n_subjects, n_times, n_vertices)) is significantly different from zero,
    while controlling for multiple comparisons via clustering.
    
    Args:
        data: Array of shape (n_subjects, n_times, n_vertices).
        threshold: The threshold to form clusters (t-value). If None, an 
                   F-threshold based on a p=0.05 significance level is chosen.
        n_permutations: Number of permutations.
        tail: 0 for two-tailed, 1 for upper, -1 for lower.
        adjacency: Sparse adjacency matrix defining the spatial connectivity of vertices.
        n_jobs: Number of parallel jobs.
        
    Returns:
        Tuple of (T_obs, clusters, cluster_p_values, H0)
    """
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        data,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        n_jobs=n_jobs,
        out_type='indices'
    )
    return T_obs, clusters, cluster_p_values, H0

def get_significance_windows(p_values: np.ndarray, alpha: float = 0.05) -> list:
    """
    Identifies contiguous indices (e.g. time windows or vertices) where 
    the p-value is less than or equal to the specified alpha.
    
    Args:
        p_values: 1D array of p-values over time or space.
        alpha: Significance threshold (default 0.05).
        
    Returns:
        List of tuples, where each tuple (start_idx, end_idx) represents a 
        contiguous window of significant p-values.
    """
    significant = (p_values <= alpha).astype(int)
    # Find edges where significance switches
    diffs = np.diff(np.pad(significant, (1, 1), mode='constant', constant_values=0))
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    windows = list(zip(starts, ends))
    return windows

def get_significance_onset(p_values: np.ndarray, alpha: float = 0.05) -> int:
    """
    Finds the first time point (latency index) where the p-value crosses the 
    significance threshold. Useful for correlation with behavioral reaction times.
    
    Args:
        p_values: 1D array of p-values over time.
        alpha: Significance threshold.
        
    Returns:
        Index of the first significant time point, or -1 if no significance found.
    """
    significant_indices = np.where(p_values <= alpha)[0]
    if len(significant_indices) > 0:
        return int(significant_indices[0])
    return -1

def get_peak_latency(data: np.ndarray, tmin_idx: int = 0, tmax_idx: int = None, find_min: bool = True) -> int:
    """
    Finds the latency index of the peak (maximum or minimum) amplitude within a specified time window.
    
    Args:
        data: 1D array of neural amplitude over time.
        tmin_idx: Start index of the time window.
        tmax_idx: End index of the time window (if None, goes to end of array).
        find_min: If True, finds the global minimum (e.g., for beta desynchronization). 
                  If False, finds the global maximum.
                  
    Returns:
        Index of the peak amplitude.
    """
    if tmax_idx is None:
        tmax_idx = len(data)
        
    window_data = data[tmin_idx:tmax_idx]
    
    if find_min:
        peak_idx = np.argmin(window_data)
    else:
        peak_idx = np.argmax(window_data)
        
    return int(tmin_idx + peak_idx)

def compute_motor_lateralization(
    data_left_choice: np.ndarray,
    data_right_choice: np.ndarray,
    left_hemi_indices: list,
    right_hemi_indices: list
) -> np.ndarray:
    """
    Computes Contralateral minus Ipsilateral lateralization contrasts, typically used 
    for Motor channels (LRP) or Beta desynchronization.
    
    Contra = (Left Choice, Right Hemi) + (Right Choice, Left Hemi)
    Ipsi = (Left Choice, Left Hemi) + (Right Choice, Right Hemi)
    
    Args:
        data_left_choice: Data array for left choices (n_channels, n_times)
        data_right_choice: Data array for right choices (n_channels, n_times)
        left_hemi_indices: Indices of left hemisphere channels/ROIs
        right_hemi_indices: Indices of right hemisphere channels/ROIs
        
    Returns:
        Array of shape (n_selected_channels, n_times) representing Contra - Ipsi
    """
    # Contralateral: right hemi for left choices, left hemi for right choices
    contra_right = data_left_choice[right_hemi_indices, ...]
    contra_left = data_right_choice[left_hemi_indices, ...]
    contra = np.mean([contra_right, contra_left], axis=0)
    
    # Ipsilateral: left hemi for left choices, right hemi for right choices
    ipsi_left = data_left_choice[left_hemi_indices, ...]
    ipsi_right = data_right_choice[right_hemi_indices, ...]
    ipsi = np.mean([ipsi_left, ipsi_right], axis=0)
    
    return contra - ipsi
