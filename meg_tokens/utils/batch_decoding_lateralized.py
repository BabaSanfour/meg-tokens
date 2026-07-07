import os
import argparse
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
try:
    # Try importing scikit-learn's permutation_test_score
    from sklearn.model_selection import permutation_test_score
except ImportError:
    pass

def compute_vertex_time_decoding(X, y, groups, n_permutations=0):
    """
    Computes univariate decoding accuracy for a single vertex over all timepoints.
    X shape: (epochs, n_times)
    Returns: array of shape (n_times,) with decoding accuracy.
    """
    n_epochs, n_times = X.shape
    scores = np.zeros(n_times)
    perm_scores = np.zeros((n_times, n_permutations)) if n_permutations > 0 else None
    
    logo = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis()
    
    for t in range(n_times):
        X_t = X[:, t].reshape(-1, 1) # Univariate feature for this timepoint
        
        if n_permutations > 0:
            score, perm_score, pval = permutation_test_score(
                clf, X_t, y, groups=groups, scoring="accuracy", cv=logo, n_permutations=n_permutations, n_jobs=1
            )
            scores[t] = score
            perm_scores[t, :] = perm_score
        else:
            # Manual cross-validation for speed when not permuting
            t_scores = []
            for train_idx, test_idx in logo.split(X_t, y, groups):
                clf.fit(X_t[train_idx], y[train_idx])
                preds = clf.predict(X_t[test_idx])
                t_scores.append(accuracy_score(y[test_idx], preds))
            scores[t] = np.mean(t_scores)
            
    return scores, perm_scores

def run_batch_decoding_lateralized(
    data_dir: str,
    fwd_dir: str,
    output_dir: str,
    subjects: list,
    conditions: list,
    behavior_filter: str = None,
    n_permutations: int = 100,
    n_jobs: int = 4
):
    """
    Performs massive univariate spatial searchlight decoding on LATERALIZED source data.
    Instead of full source matrices, we compute (Left Hemisphere - Right Hemisphere) 
    per vertex, and decode conditions independently at each vertex and timepoint.
    """
    print(f"=== Starting Lateralized Source Searchlight MVPA ===")
    
    print(f"Loading STC data for {len(conditions)} conditions across {len(subjects)} subjects...")
    
    # In a real environment, you would pass the actual frequencies or loop over them.
    freq = 'alpha' 
    
    X_list = []
    y_list = []
    groups_list = []
    
    for class_idx, cond_name in enumerate(conditions):
        # We will map the cond_name to the legacy sTrialClass (1=Easy, 2=Ambiguous, 3=Misleading)
        class_mapping = {'Easy': 1, 'Ambiguous': 2, 'Misleading': 3}
        target_class = class_mapping.get(cond_name, 1)
        
        cond_lat_data = []
        cond_groups = []
        
        for subj_idx, s in enumerate(subjects):
            epochs_file = os.path.join(data_dir, f"{s}-epo.fif")
            inv_file = os.path.join(fwd_dir, f"{s}-inv.fif")
            
            if not os.path.exists(epochs_file) or not os.path.exists(inv_file):
                # We skip missing subjects on this machine
                continue
                
            # 1. Load the modern Epochs file which already contains the behavioral CSV embedded inside it!
            epochs = mne.read_epochs(epochs_file, preload=True)
            
            # 2. Slice the epochs natively using Pandas syntax on the metadata
            query = f"sTrialClass == {target_class} and nChoiceMade in [1, 2]"
            if behavior_filter:
                query = f"({query}) and ({behavior_filter})"
                
            target_epochs = epochs[query]
            
            if len(target_epochs) == 0:
                continue
                
            # 3. Load inverse operator and compute source estimates dynamically
            from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
            inverse_operator = read_inverse_operator(inv_file)
            snr = 3.0
            lambda2 = 1.0 / snr ** 2
            
            stcs = apply_inverse_epochs(target_epochs, inverse_operator, lambda2, method='dSPM')
            
            # 4. Compute Left - Right lateralization natively per epoch
            for stc in stcs:
                # Assuming fsaverage oct6 with 4098 vertices per hemisphere
                lh_data = stc.data[:4098, :]
                rh_data = stc.data[4098:8196, :]
                lat_data = lh_data - rh_data
                
                cond_lat_data.append(lat_data)
                cond_groups.append(subj_idx)
                
        if len(cond_lat_data) > 0:
            X_list.append(np.stack(cond_lat_data))
            y_list.append(np.full(len(cond_lat_data), class_idx))
            groups_list.append(np.array(cond_groups))
        
    if len(X_list) == 0:
        print("No valid STC and CSV data found on this machine! Cannot proceed with real data computation.")
        return
        
    X_lat = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    
    print(f"Data Matrix: {X_lat.shape} (Epochs x Hemi-Vertices x Times)")
    
    print(f"Running Univariate Decoder (Vertex by Vertex) with {n_permutations} permutations...")
    # Parallelize across vertices
    # X_lat is (n_epochs, n_vertices, n_times)
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_vertex_time_decoding)(
            X_lat[:, v, :], y, groups, n_permutations
        ) for v in range(n_vertices_hemi)
    )
    
    # Reconstruct the 4D output map
    # results is a list of tuples (scores, perm_scores)
    da_scores = np.array([r[0] for r in results]) # Shape: (n_vertices, n_times)
    
    if n_permutations > 0:
        da_perms = np.array([r[1] for r in results]) # Shape: (n_vertices, n_times, n_perms)
    else:
        da_perms = None
        
    # The legacy script reconstructed the full 8196 array by duplicating it with a sign flip
    # We will save the 4098 hemisphere map directly, which is mathematically identical but cleaner
    
    out_file = os.path.join(output_dir, f"da_lateralized_{'_vs_'.join(conditions)}.npy")
    np.save(out_file, da_scores)
    
    if da_perms is not None:
        perm_file = os.path.join(output_dir, f"da_lateralized_{'_vs_'.join(conditions)}_perms.npy")
        np.save(perm_file, da_perms)
        
    print(f"Completed! Saved Lateralized Spatial Searchlight map to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lateralized Spatial Searchlight MVPA")
    parser.add_argument("--data_dir", type=str, default='./data/epochs/',
                        help="Directory containing epoched data (.fif)")
    parser.add_argument("--fwd_dir", type=str, default='./data/forward/',
                        help="Directory containing forward solutions/inverse operators")
    parser.add_argument("--out_dir", type=str, default='./decoding_results/lateralized/',
                        help="Directory to save the spatial-temporal decoding maps")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02'],
                        help="Subjects to process")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Easy', 'Ambiguous', 'Misleading'],
                        help="Conditions to decode")
    parser.add_argument("--behavior_filter", type=str, default=None,
                        help="Pandas query string to filter behavior (e.g., 'nChoiceMade == nCorrectChoice')")
    parser.add_argument("--permutations", type=int, default=100,
                        help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=4)
    
    args = parser.parse_args()
    run_batch_decoding_lateralized(
        args.data_dir, args.fwd_dir, args.out_dir, args.subjects, 
        args.conditions, args.behavior_filter, args.permutations, args.n_jobs
    )
