import os
import argparse
import numpy as np
import mne

from meg_tokens.meg.decoding import compute_time_resolved_decoding, compute_decoding_permutations

def run_batch_decoding_roi(
    data_dir: str,
    fwd_dir: str,
    output_dir: str,
    subjects: list,
    conditions: list,
    behavior_filter: str = None,
    parcellation: str = 'HCPMMP1',
    n_permutations: int = 100,
    n_jobs: int = 4
):
    """
    Runs Time-Resolved Lateralized MVPA Decoding on ROI averages instead of raw vertices.
    
    This replaces scripts like '091_Stats_SRC_POWER_arrange_data_all_sources.ipynb'
    by dynamically loading the parcellation and extracting label time courses.
    """
    print(f"=== Starting Lateralized ROI Decoding MVPA ({parcellation}) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    X_list = []
    y_list = []
    groups_list = []
    
    print("Loading parcellation labels...")
    try:
        # Load labels for the specified parcellation
        labels = mne.read_labels_from_annot('fsaverage', parc=parcellation, subjects_dir=None)
        
        # Filter for cortical labels and pair left/right
        lh_labels = [l for l in labels if l.name.endswith('-lh') and 'unknown' not in l.name.lower()]
        rh_labels = [l for l in labels if l.name.endswith('-rh') and 'unknown' not in l.name.lower()]
        
        # Ensure they match
        lh_labels = sorted(lh_labels, key=lambda x: x.name)
        rh_labels = sorted(rh_labels, key=lambda x: x.name)
        n_rois = min(len(lh_labels), len(rh_labels))
        print(f"Extracted {n_rois} matched lateralized ROI pairs.")
    except Exception as e:
        print(f"Could not load parcellation {parcellation}: {e}")
        print("Note: In a real environment, you must have the freesurfer subject directory configured.")
        return
        
    for class_idx, cond_name in enumerate(conditions):
        class_mapping = {'Easy': 1, 'Ambiguous': 2, 'Misleading': 3}
        target_class = class_mapping.get(cond_name, 1)
        
        cond_lat_data = []
        cond_groups = []
        
        for subj_idx, s in enumerate(subjects):
            epochs_file = os.path.join(data_dir, f"{s}-epo.fif")
            inv_file = os.path.join(fwd_dir, f"{s}-inv.fif")
            src_file = os.path.join(fwd_dir, f"{s}-src.fif")
            
            if not os.path.exists(epochs_file) or not os.path.exists(inv_file) or not os.path.exists(src_file):
                continue
                
            epochs = mne.read_epochs(epochs_file, preload=True)
            
            query = f"sTrialClass == {target_class} and nChoiceMade in [1, 2]"
            if behavior_filter:
                query = f"({query}) and ({behavior_filter})"
                
            target_epochs = epochs[query]
            
            if len(target_epochs) == 0:
                continue
                
            from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
            inverse_operator = read_inverse_operator(inv_file)
            src = mne.read_source_spaces(src_file)
            
            # Compute STCs for the epochs
            stcs = apply_inverse_epochs(target_epochs, inverse_operator, lambda2=1.0/9.0, method='dSPM')
            
            # Extract ROI timecourses using 'mean_flip' to account for source dipole orientations
            # Extract left hemisphere ROIs
            lh_tc = mne.extract_label_time_course(stcs, lh_labels, src, mode='mean_flip') # shape: (epochs, n_rois, times)
            lh_tc = np.array(lh_tc) 
            
            # Extract right hemisphere ROIs
            rh_tc = mne.extract_label_time_course(stcs, rh_labels, src, mode='mean_flip') # shape: (epochs, n_rois, times)
            rh_tc = np.array(rh_tc)
            
            # Calculate Lateralization per epoch (Left - Right)
            lat_tc = lh_tc - rh_tc
            
            cond_lat_data.append(lat_tc)
            cond_groups.append(np.full(lat_tc.shape[0], subj_idx))
            
        if len(cond_lat_data) > 0:
            X_list.append(np.concatenate(cond_lat_data, axis=0))
            y_list.append(np.full(X_list[-1].shape[0], class_idx))
            groups_list.append(np.concatenate(cond_groups, axis=0))
            
    if len(X_list) == 0:
        print("No valid STC and Label data found on this machine! Cannot proceed with real data computation.")
        return
        
    X_lat = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    
    # We now have a feature matrix of shape (epochs, rois, times). 
    # To run spatial decoding (one classifier per ROI), we loop over ROIs.
    # To run a unified multivariate decoding across all ROIs, we pass the matrix directly.
    # We will loop over ROIs to mimic the legacy script.
    
    for roi_idx, roi_label in enumerate(lh_labels):
        roi_name = roi_label.name.replace('-lh', '')
        print(f"Decoding ROI: {roi_name}...")
        
        # Extract data for this specific ROI
        X_roi = X_lat[:, roi_idx, :][:, np.newaxis, :] # shape: (epochs, 1, times)
        
        if n_permutations > 0:
            scores, perm_scores, threshold = compute_decoding_permutations(
                X=X_roi, y=y, groups=groups, balance=True, n_permutations=n_permutations, n_jobs=n_jobs
            )
        else:
            scores = compute_time_resolved_decoding(
                X=X_roi, y=y, groups=groups, balance=True, n_jobs=n_jobs
            )
            threshold = None
            
        mean_scores = np.mean(scores, axis=0)
        
        roi_out_file = os.path.join(output_dir, f"decoding_scores_{roi_name}_{'_vs_'.join(conditions)}.npy")
        np.save(roi_out_file, mean_scores)
        
        if threshold is not None:
            thresh_file = os.path.join(output_dir, f"decoding_threshold_{roi_name}_{'_vs_'.join(conditions)}.npy")
            np.save(thresh_file, np.array([threshold]))
            
    print(f"Finished ROI decoding. All results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lateralized ROI Decoding MVPA")
    parser.add_argument("--data_dir", type=str, default='./data/epochs/',
                        help="Directory containing epoched data (.fif)")
    parser.add_argument("--fwd_dir", type=str, default='./data/forward/',
                        help="Directory containing forward solutions/inverse operators")
    parser.add_argument("--out_dir", type=str, default='./decoding_results/roi/',
                        help="Directory to save the ROI decoding scores")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02'],
                        help="Subjects to process")
    parser.add_argument("--conditions", type=str, nargs='+', default=['Easy', 'Ambiguous', 'Misleading'],
                        help="Conditions to decode")
    parser.add_argument("--behavior_filter", type=str, default=None,
                        help="Pandas query string to filter behavior (e.g., 'nChoiceMade == nCorrectChoice')")
    parser.add_argument("--parcellation", type=str, default='HCPMMP1',
                        help="FreeSurfer parcellation annotation (e.g. HCPMMP1, aparc)")
    parser.add_argument("--permutations", type=int, default=100,
                        help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=4)
    
    args = parser.parse_args()
    run_batch_decoding_roi(
        args.data_dir, args.fwd_dir, args.out_dir, args.subjects, 
        args.conditions, args.behavior_filter, args.parcellation, args.permutations, args.n_jobs
    )
