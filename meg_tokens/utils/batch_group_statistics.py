"""
Pipeline execution script for Stage 7: Group Statistics & Permutations.
Loads parcellated neural space matrices from Stage 6 and runs group-level contrasts across conditions.
"""

import os
import glob
import numpy as np
from meg_tokens.meg.stats import compute_permutation_t_test, get_significance_windows

def run_group_statistics_contrast(
    erp_export_dir: str,
    output_dir: str,
    conditions: tuple = ('Fast', 'Slow'),
    n_permutations: int = 1000,
    subjects_list: list = None
):
    """
    Executes group-level statistics contrasting two conditions across subjects.
    """
    print(f"=== Running Group Statistics Contrast: {conditions[0]} vs {conditions[1]} ===")
    
    # Collect data for all subjects
    # Shape expectations: (n_subjects, n_labels, n_times)
    cond1_data = []
    cond2_data = []
    
    if subjects_list is None:
        subjects = sorted([d for d in os.listdir(erp_export_dir) if d.startswith('H')])
    else:
        subjects = subjects_list
    
    for subject in subjects:
        # Load condition 1
        cond1_files = glob.glob(os.path.join(erp_export_dir, subject, f"*{conditions[0]}*.npy"))
        if cond1_files:
            cond1_data.append(np.load(cond1_files[0]))
            
        # Load condition 2
        cond2_files = glob.glob(os.path.join(erp_export_dir, subject, f"*{conditions[1]}*.npy"))
        if cond2_files:
            cond2_data.append(np.load(cond2_files[0]))
            
    if not cond1_data or not cond2_data:
        print("Insufficient data to run statistics.")
        return
        
    cond1_arr = np.array(cond1_data)
    cond2_arr = np.array(cond2_data)
    
    # Compute Difference Contrast (Cond1 - Cond2)
    contrast = cond1_arr - cond2_arr
    
    # Typically, permutation_t_test expects shape (n_observations, n_features)
    # We flatten (n_labels, n_times) into n_features, then reshape back
    n_subj, n_labels, n_times = contrast.shape
    contrast_flat = contrast.reshape(n_subj, -1)
    
    print(f"Running permutation t-test on {n_subj} subjects ({n_permutations} perms)...")
    t_obs, p_values, H0 = compute_permutation_t_test(contrast_flat, n_permutations=n_permutations)
    
    # Reshape p-values back to (n_labels, n_times)
    p_values_reshaped = p_values.reshape(n_labels, n_times)
    t_obs_reshaped = t_obs.reshape(n_labels, n_times)
    
    # Extract significance windows for each label
    for label_idx in range(n_labels):
        windows = get_significance_windows(p_values_reshaped[label_idx, :], alpha=0.05)
        if windows:
            print(f"Label index {label_idx} has significant time windows: {windows}")
            
    # Save statistics results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, f"stats_{conditions[0]}_vs_{conditions[1]}_pvals.npy"), p_values_reshaped)
    np.save(os.path.join(output_dir, f"stats_{conditions[0]}_vs_{conditions[1]}_tobs.npy"), t_obs_reshaped)
    print("Group statistics completed and exported.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run group-level statistical contrasts (permutation t-tests) between two conditions."
    )
    parser.add_argument("--erp_dir", type=str, default='/media/external/DDM/export_erp/',
                        help="Directory containing the subject-level ERP exports (default: /media/external/DDM/export_erp/)")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/stats_export/',
                        help="Output directory to save statistics (default: /media/external/DDM/stats_export/)")
    parser.add_argument("--conditions", type=str, nargs=2, default=['Fast', 'Slow'],
                        help="Two condition names to contrast (e.g., --conditions Correct Error). Default is Fast Slow.")
    parser.add_argument("--perms", type=int, default=1000,
                        help="Number of permutations to run (default: 1000)")
    parser.add_argument("--subjects", type=str, nargs='+', default=None,
                        help="Specific subjects to include (e.g., --subjects H01 H02). If omitted, runs all subjects.")
    
    args = parser.parse_args()
    
    # Execute the pipeline with parsed arguments
    run_group_statistics_contrast(
        erp_export_dir=args.erp_dir,
        output_dir=args.out_dir,
        conditions=tuple(args.conditions),
        n_permutations=args.perms,
        subjects_list=args.subjects
    )
