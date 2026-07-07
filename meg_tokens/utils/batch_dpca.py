import os
import argparse
import numpy as np
import mne
try:
    from dPCA import dPCA
except ImportError:
    print("Warning: dPCA package not found. Please install it using 'pip install dPCA'")

def run_batch_dpca(
    data_dir: str,
    fwd_dir: str,
    output_dir: str,
    subjects: list,
    marginalize_cols: list,
    dpca_labels: str = None,
    n_components: int = 20,
    behavior_filter: str = None,
    rois: list = None,
    parc: str = 'aparc.a2009s',
    subjects_dir: str = './data/fs_subjects_dir',
    is_volume: bool = False
):
    """
    Dynamically constructs a multidimensional tensor from Epochs.metadata 
    and runs Demixed Principal Component Analysis (dPCA).
    """
    print(f"=== Starting Demixed PCA (dPCA) ===")
    print(f"Marginalizing over: {marginalize_cols}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_subjects_stcs = []
    all_subjects_meta = []
    active_vertices = None
    
    # 1. Load data for all subjects
    for s in subjects:
        epochs_file = os.path.join(data_dir, f"{s}-epo.fif")
        inv_file = os.path.join(fwd_dir, f"{s}-inv.fif")
        
        if not os.path.exists(epochs_file) or not os.path.exists(inv_file):
            continue
            
        epochs = mne.read_epochs(epochs_file, preload=True)
        
        if behavior_filter:
            epochs = epochs[behavior_filter]
            
        if len(epochs) == 0:
            continue
            
        from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
        inverse_operator = read_inverse_operator(inv_file)
        
        # We need single-trial data for dPCA
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2=1.0/9.0, method='dSPM')
        
        if rois:
            print(f"Restricting source space to ROIs: {rois}")
            if is_volume:
                # Volume label extraction logic
                pass # Volumetric ROI extraction would go here, often requires nibabel/atlas masking
            else:
                labels = mne.read_labels_from_annot(s, parc=parc, subjects_dir=subjects_dir)
                target_labels = [lbl for lbl in labels if lbl.name in rois]
                if not target_labels:
                    print(f"Warning: None of the ROIs found in {parc} for subject {s}")
                    continue
                label_combined = target_labels[0]
                for lbl in target_labels[1:]:
                    label_combined += lbl
                stcs = [stc.in_label(label_combined) for stc in stcs]
                
        if active_vertices is None and len(stcs) > 0:
            active_vertices = stcs[0].vertices
        
        # Convert list of STCs to array: (n_epochs, n_vertices, n_times)
        stc_data = np.array([stc.data for stc in stcs])
        
        all_subjects_stcs.append(stc_data)
        all_subjects_meta.append(epochs.metadata.copy())
        
    if len(all_subjects_stcs) == 0:
        print("No valid STC data found! Cannot proceed with dPCA.")
        return
        
    # Concatenate all subjects together to form one giant pseudo-subject matrix
    # Shape: (total_epochs, n_vertices, n_times)
    X_concat = np.concatenate(all_subjects_stcs, axis=0)
    
    import pandas as pd
    meta_concat = pd.concat(all_subjects_meta, axis=0).reset_index(drop=True)
    
    print(f"Aggregated Data Shape: {X_concat.shape}")
    
    # 2. Dynamically construct the multidimensional dPCA tensor
    # Find unique values for each marginalization column
    unique_vals = [sorted(meta_concat[col].dropna().unique()) for col in marginalize_cols]
    dims = [len(uv) for uv in unique_vals]
    
    n_features = X_concat.shape[1]
    n_times = X_concat.shape[2]
    
    # We must find the maximum number of trials in any specific condition intersection
    # to pre-allocate the trialR array. dPCA pads with NaNs for unbalanced trial counts.
    grouped = meta_concat.groupby(marginalize_cols)
    max_trials = grouped.size().max()
    
    # Output tensor shape: (n_trials, n_features, dim1, dim2, ..., n_times)
    tensor_shape = [max_trials, n_features] + dims + [n_times]
    trialR = np.full(tensor_shape, np.nan)
    
    print(f"Constructing dPCA Tensor of shape: {tensor_shape}...")
    
    import itertools
    for idxs in itertools.product(*[range(d) for d in dims]):
        # Build query dictionary for this specific intersection
        query_dict = {col: unique_vals[i][idx] for i, (col, idx) in enumerate(zip(marginalize_cols, idxs))}
        
        # Filter metadata for these condition values
        mask = np.ones(len(meta_concat), dtype=bool)
        for col, val in query_dict.items():
            mask &= (meta_concat[col] == val)
            
        target_indices = np.where(mask)[0]
        n_found = len(target_indices)
        
        if n_found > 0:
            # Extract the raw trials
            trials_data = X_concat[target_indices, :, :] # (n_found, n_features, n_times)
            
            # Construct the slice index: [0:n_found, :, idx1, idx2, ..., :]
            slice_idx = [slice(0, n_found), slice(None)] + list(idxs) + [slice(None)]
            trialR[tuple(slice_idx)] = trials_data
            
    # Compute the mean tensor R (ignoring NaNs from unbalanced trials)
    print("Computing mean tensor R...")
    with np.errstate(invalid='ignore'):
        R = np.nanmean(trialR, axis=0)
        
    print(f"Mean Tensor R shape: {R.shape}")
    
    # 3. Run dPCA
    print(f"Running dPCA with {n_components} components...")
    
    if dpca_labels is None:
        # Generate generic labels: 'a', 'b', 'c', ..., 't'
        # Last dimension is ALWAYS time ('t')
        import string
        dpca_labels = "".join(string.ascii_lowercase[:len(marginalize_cols)]) + "t"
        
    try:
        dpca = dPCA(labels=dpca_labels, regularizer=0, n_components=n_components)
        dpca.protect = ['t']
        
        # Fit dPCA
        Z = dpca.fit_transform(R, trialX=trialR)
        weights = dpca.D
        
        # Significance testing
        print("Running dPCA significance analysis...")
        significance_masks, true_scores, shuffle_scores = dpca.significance_analysis(
            R, trialR, n_shuffles=100, n_splits=10, n_consecutive=1, full=True, axis=True
        )
        
        # Save results
        out_prefix = os.path.join(output_dir, f"dpca_{'_'.join(marginalize_cols)}")
        np.save(f"{out_prefix}_weights.npy", weights)
        np.save(f"{out_prefix}_components.npy", Z)
        np.save(f"{out_prefix}_signif_masks.npy", significance_masks)
        np.save(f"{out_prefix}_true_scores.npy", true_scores)
        np.save(f"{out_prefix}_shuffle_scores.npy", shuffle_scores)
        
        if active_vertices is not None:
            # np.save handles lists of arrays nicely if dtype=object
            np.save(f"{out_prefix}_vertices.npy", np.array(active_vertices, dtype=object))
        
        print(f"Finished dPCA. Results saved to {output_dir}")
        
    except NameError:
        print("dPCA is not installed. Please install it and re-run.")
    except Exception as e:
        print(f"dPCA execution failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Demixed PCA (dPCA) from MNE Epochs")
    parser.add_argument("--data_dir", type=str, default='./data/epochs/',
                        help="Directory containing epoched data (.fif)")
    parser.add_argument("--fwd_dir", type=str, default='./data/forward/',
                        help="Directory containing forward solutions/inverse operators")
    parser.add_argument("--out_dir", type=str, default='./dpca_results/',
                        help="Directory to save dPCA outputs")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02'],
                        help="Subjects to process")
    parser.add_argument("--marginalize_cols", type=str, nargs='+', required=True,
                        help="Metadata columns to group by (e.g. sTrialClass nChoiceMade)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Custom dPCA string label matching dimensions (e.g., 'st')")
    parser.add_argument("--n_components", type=int, default=20,
                        help="Number of dPCA components to extract")
    parser.add_argument("--behavior_filter", type=str, default=None,
                        help="Optional pandas query string to pre-filter trials")
    parser.add_argument("--rois", type=str, nargs='+', default=None,
                        help="Specific ROIs to restrict extraction to")
    parser.add_argument("--parc", type=str, default='aparc.a2009s',
                        help="Parcellation scheme for ROIs")
    parser.add_argument("--volume", action='store_true',
                        help="Flag indicating inverse operator uses volumetric source space")
    
    args = parser.parse_args()
    run_batch_dpca(
        args.data_dir, args.fwd_dir, args.out_dir, args.subjects, 
        args.marginalize_cols, args.labels, args.n_components, args.behavior_filter,
        args.rois, args.parc, is_volume=args.volume
    )
