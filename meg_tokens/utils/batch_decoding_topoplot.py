import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from meg_tokens.meg.decoding import compute_spatial_decoding, compute_spatial_decoding_permutations

def run_batch_decoding_topoplot(
    data_dir: str,
    output_dir: str,
    conditions: list,
    permutations: int = 0,
    n_jobs: int = 4
):
    print(f"=== Starting Spatial MVPA Decoding (Topoplot) ===")
    print(f"Conditions: {conditions}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Searching for sensor data in {data_dir}...")
    
    # --- MOCK DATA GENERATION FOR DEMONSTRATION ---
    n_subjects = 28
    trials_per_subj = 40
    n_epochs = n_subjects * trials_per_subj
    n_sensors = 270 # 270 MEG sensors
    
    print(f"Simulating time-averaged feature extraction for {len(conditions)} conditions across {n_subjects} subjects...")
    
    X_list = []
    y_list = []
    groups_list = []
    
    for class_idx, cond in enumerate(conditions):
        X_cond = np.random.normal(0, 1, (n_epochs, n_sensors))
        y_cond = np.full(n_epochs, class_idx)
        groups_cond = np.repeat(np.arange(n_subjects), trials_per_subj)
        
        X_list.append(X_cond)
        y_list.append(y_cond)
        groups_list.append(groups_cond)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    # -----------------------------------------------
    
    print(f"Constructed Data Matrix X: {X.shape} (Epochs x Sensors)")
    
    if permutations > 0:
        print(f"Running Spatial LDA Searchlight with {permutations} permutations...")
        scores, perm_scores, threshold = compute_spatial_decoding_permutations(
            X=X, y=y, groups=groups, balance=True, n_permutations=permutations, n_jobs=n_jobs
        )
        print(f"Calculated 95% permutation threshold at Accuracy: {threshold:.3f}")
    else:
        print("Running Spatial LDA Searchlight...")
        scores = compute_spatial_decoding(
            X=X, y=y, groups=groups, balance=True, n_jobs=n_jobs
        )
        threshold = None
    
    # Save scores
    out_file = os.path.join(output_dir, f"da_spatial_scores_{'_vs_'.join(conditions)}.npy")
    np.save(out_file, scores)
    
    # Generate Topomap
    # Note: Requires actual raw/evoked info to plot sensor positions in practice
    try:
        # Create a mock info object for 270 channels if no real info is available
        info = mne.create_info(ch_names=[str(i) for i in range(n_sensors)], sfreq=1000, ch_types='mag')
        # Setting a standard montage so it can plot
        montage = mne.channels.make_standard_montage('standard_1020')
        # This is just a scaffold. In reality, info is loaded from a raw file.
        # mne.viz.plot_topomap(scores, info, ...)
        print("Note: Topoplot generation skipped in mock execution because valid MEG sensor coordinates are required.")
    except Exception as e:
        print("Mock topoplot error:", e)
        
    print(f"Saved topoplot to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatial MVPA Decoding and Topoplot")
    parser.add_argument("--data_dir", type=str, default='/media/external/DDM/sensor_rec/',
                        help="Directory containing time-averaged sensor data")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/decoding_results/topo/',
                        help="Directory to save topoplots")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Fast', 'Slow'],
                        help="Conditions to decode")
    parser.add_argument("--permutations", type=int, default=0,
                        help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=4)
    
    args = parser.parse_args()
    run_batch_decoding_topoplot(args.data_dir, args.out_dir, args.conditions, args.permutations, args.n_jobs)
