import os
import argparse
import numpy as np
import mne
from mne.stats import permutation_t_test

def run_batch_stats_lateralized(
    data_dir: str,
    fwd_dir: str,
    output_dir: str,
    subjects: list,
    condition: str,
    behavior_filter: str = None,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    n_jobs: int = 4
):
    """
    Computes univariate spatio-temporal permutation T-tests on lateralized source data.
    
    This replaces the legacy scripts (like 091_Figures_brain_stats_all_sources.ipynb) 
    that manually performed T-tests vertex-by-vertex.
    """
    print(f"=== Starting Lateralized Spatio-Temporal T-Tests ===")
    print(f"Condition: {condition}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    class_mapping = {'Easy': 1, 'Ambiguous': 2, 'Misleading': 3}
    target_class = class_mapping.get(condition, 1)
    
    stc_list = []
    
    # 1. Load Data
    print("Loading epoched data and computing source estimates...")
    for s in subjects:
        epochs_file = os.path.join(data_dir, f"{s}-epo.fif")
        inv_file = os.path.join(fwd_dir, f"{s}-inv.fif")
        
        if not os.path.exists(epochs_file) or not os.path.exists(inv_file):
            continue
            
        # Construct query based on condition mapping and optional behavior filters
        query = f"sTrialClass == {target_class} and nChoiceMade in [1, 2]"
        if behavior_filter:
            query = f"({query}) and ({behavior_filter})"
            
        target_epochs = epochs[query]
        
        if len(target_epochs) == 0:
            continue
            
        from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
        inverse_operator = read_inverse_operator(inv_file)
        
        # We average epochs to get the ERP per subject before running stats across subjects
        evoked = target_epochs.average()
        
        from mne.minimum_norm import apply_inverse
        stc = apply_inverse(evoked, inverse_operator, lambda2=1.0/9.0, method='dSPM')
        
        # Calculate lateralization
        lh_data = stc.data[:4098, :]
        rh_data = stc.data[4098:8196, :]
        lat_data = lh_data - rh_data  # Shape: (4098, times)
        
        stc_list.append(lat_data)
        
    if len(stc_list) < 2:
        print("Not enough subjects to run T-tests.")
        return
        
    # Shape: (n_subjects, vertices, times)
    X = np.stack(stc_list, axis=0)
    n_subjects, n_vertices, n_times = X.shape
    
    print(f"Data matrix assembled: {X.shape}")
    print(f"Running permutation T-tests (n_permutations={n_permutations}) across {n_vertices} vertices and {n_times} timepoints...")
    
    t_scores = np.zeros((n_vertices, n_times))
    p_values = np.ones((n_vertices, n_times))
    
    # Run tests per timepoint (parallelizing across vertices internally in mne)
    for t in range(n_times):
        # Shape passed to permutation_t_test: (n_subjects, n_vertices)
        T_obs, p_vals, H0 = permutation_t_test(
            X[:, :, t], n_permutations=n_permutations, tail=0, n_jobs=n_jobs
        )
        t_scores[:, t] = T_obs
        p_values[:, t] = p_vals
        
    # Apply alpha mask (zero out non-significant t-scores)
    sig_mask = p_values < alpha
    t_scores_masked = np.where(sig_mask, t_scores, 0)
    
    # Mirror lateralized data back to full brain for plotting
    # (Left hemisphere gets positive lateralization, Right gets negative)
    full_brain_t = np.vstack([t_scores_masked, -t_scores_masked])
    
    out_file = os.path.join(output_dir, f"t_scores_lateralized_{condition}.npy")
    np.save(out_file, full_brain_t)
    print(f"Saved thresholded T-scores (p < {alpha}) to {out_file}")
    print(f"You can plot this using batch_plot_pca_loadings.py --save_movie")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatio-Temporal T-tests on Lateralized data")
    parser.add_argument("--data_dir", type=str, default='./data/epochs/',
                        help="Directory containing epoched data (.fif)")
    parser.add_argument("--fwd_dir", type=str, default='./data/forward/',
                        help="Directory containing forward solutions/inverse operators")
    parser.add_argument("--out_dir", type=str, default='./stats_results/lateralized/',
                        help="Directory to save the spatial-temporal T-score maps")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02'],
                        help="Subjects to process")
    parser.add_argument("--condition", type=str, default='Easy',
                        help="Condition to test")
    parser.add_argument("--behavior_filter", type=str, default=None,
                        help="Pandas query string to filter behavior (e.g., 'nChoiceMade == nCorrectChoice')")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="P-value threshold")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=4)
    
    args = parser.parse_args()
    run_batch_stats_lateralized(
        args.data_dir, args.fwd_dir, args.out_dir, args.subjects, 
        args.condition, args.behavior_filter, args.alpha, args.permutations, args.n_jobs
    )
