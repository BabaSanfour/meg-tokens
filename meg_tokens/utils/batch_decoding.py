"""
Batch script to run Time-Resolved MVPA Decoding across source-space features.
Obsoletes the legacy `compute_classif_*.py` and MATLAB LDA scripts.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from meg_tokens.meg.decoding import compute_time_resolved_decoding, compute_decoding_permutations

def run_batch_decoding(
    data_dir: str,
    output_dir: str,
    conditions: list,
    permutations: int = 0,
    n_jobs: int = 4
):
    print(f"=== Starting Time-Resolved MVPA Decoding ===")
    print(f"Conditions: {conditions}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # This is a scaffold for the CLI to demonstrate data loading and decoding.
    # In practice, one would load the parcellated SourceEstimates or Epochs 
    # from data_dir corresponding to the subjects and conditions.
    
    print(f"Searching for data in {data_dir}...")
    
    # --- MOCK DATA GENERATION FOR DEMONSTRATION ---
    # Imagine we loaded 28 subjects, 100 trials each, for 68 ROIs over 63 time points.
    n_subjects = 28
    trials_per_subj = 40
    n_epochs = n_subjects * trials_per_subj
    n_features = 68 # e.g. 68 cortical ROIs
    n_times = 63 # e.g. time points
    
    print(f"Simulating feature extraction for {len(conditions)} conditions across {n_subjects} subjects (Inter-Subject Decoding)...")
    
    X_list = []
    y_list = []
    groups_list = []
    
    for class_idx, cond in enumerate(conditions):
        # Create dummy data matrix (epochs, features, time)
        # Adding some temporal "signal" halfway through for condition 1 to demonstrate decoding accuracy spiking
        signal_offset = 0.5 if class_idx == 1 else 0.0
        X_cond = np.random.normal(0, 1, (n_epochs, n_features, n_times))
        X_cond[:, :, 30:45] += signal_offset
        
        y_cond = np.full(n_epochs, class_idx)
        
        # Group labels: [0,0,0..., 1,1,1..., 27,27,27...]
        groups_cond = np.repeat(np.arange(n_subjects), trials_per_subj)
        
        X_list.append(X_cond)
        y_list.append(y_cond)
        groups_list.append(groups_cond)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    # -----------------------------------------------
    
    print(f"Constructed Data Matrix X: {X.shape} (Epochs x ROIs x Times)")
    print(f"Constructed Labels y: {y.shape}")
    print(f"Constructed Subject Groups: {groups.shape}")
    
    if permutations > 0:
        print(f"Running Time-Resolved LDA with Leave-One-Subject-Out Cross Validation and {permutations} permutations...")
        scores, perm_scores, threshold = compute_decoding_permutations(
            X=X, y=y, groups=groups, balance=True, n_permutations=permutations, n_jobs=n_jobs
        )
        print(f"Calculated 95% permutation threshold at Accuracy: {threshold:.3f}")
    else:
        print("Running Time-Resolved LDA with Leave-One-Subject-Out Cross Validation...")
        scores = compute_time_resolved_decoding(
            X=X, y=y, groups=groups, balance=True, n_jobs=n_jobs
        )
        threshold = None
    
    # Average across all cross-validation splits
    mean_scores = np.mean(scores, axis=0)
    
    # Save scores
    out_file = os.path.join(output_dir, f"decoding_scores_{'_vs_'.join(conditions)}.npy")
    np.save(out_file, mean_scores)
    print(f"Saved decoding scores to {out_file}")
    
    if threshold is not None:
        thresh_file = os.path.join(output_dir, f"decoding_threshold_{'_vs_'.join(conditions)}.npy")
        np.save(thresh_file, np.array([threshold]))
        print(f"Saved decoding threshold to {thresh_file}")
    
    # Plotting Accuracy
    times = np.arange(n_times) * 10 # Mock time axis in ms
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mean_scores, label='LDA Accuracy', lw=2, color='darkblue')
    ax.axhline(1.0 / len(conditions), color='k', linestyle='--', label='Chance Level')
    
    if threshold is not None:
        ax.axhline(threshold, color='red', linestyle=':', label='p<0.05 Threshold')
        # Shade significant regions
        significant = mean_scores > threshold
        if np.any(significant):
            ax.fill_between(times, mean_scores, threshold, where=significant, color='red', alpha=0.3)
            
    ax.set_title(f"Time-Resolved Decoding: {' vs '.join(conditions)}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    
    fig_file = os.path.join(output_dir, f"decoding_plot_{'_vs_'.join(conditions)}.png")
    fig.savefig(fig_file)
    print(f"Saved decoding plot to {fig_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Time-Resolved MVPA Decoding")
    parser.add_argument("--data_dir", type=str, default='/media/external/DDM/source_rec/',
                        help="Directory containing parcellated source or sensor data")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/decoding_results/',
                        help="Directory to save scores and plots")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Fast', 'Slow'],
                        help="List of conditions to decode (e.g. Fast Slow, or Easy Ambiguous Misleading)")
    parser.add_argument("--alignment", type=str, default='enter', choices=['enter', 'go'],
                        help="Which epoch alignment to load (enter or go).")
    parser.add_argument("--compare_to_baseline", action="store_true",
                        help="If set, decodes the current timepoint against a pre-stimulus baseline window.")
    parser.add_argument("--permutations", type=int, default=0,
                        help="Number of permutations to calculate statistical significance threshold.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs for cross-validation")
    
    args = parser.parse_args()
    
    # In a real implementation, data_dir would be modified to load from data_dir/alignment/
    # If compare_to_baseline is true, conditions would be forced to ['Baseline', 'Task']
    
    run_batch_decoding(args.data_dir, args.out_dir, args.conditions, args.permutations, args.n_jobs)
