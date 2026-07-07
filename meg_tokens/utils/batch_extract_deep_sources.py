import os
import argparse
import numpy as np
import scipy.io as sio
import mne

def run_batch_extract_deep_sources(
    data_dir: str,
    fwd_dir: str,
    output_dir: str,
    subjects: list,
    condition: str,
    behavior_filter: str = None
):
    """
    Extracts deep brain volume sources from mixed source spaces and exports them to MATLAB.
    
    This replaces the legacy '091_Stats_SRC_POWER_arrange_data_all_sources-DEEP.ipynb' 
    which manually sliced vertices [8196:] and exported them via scipy.io.savemat.
    """
    print(f"=== Starting Deep Volume Source Extraction ===")
    print(f"Condition: {condition}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    class_mapping = {'Easy': 1, 'Ambiguous': 2, 'Misleading': 3}
    target_class = class_mapping.get(condition, 1)
    
    deep_data_list = []
    
    # 1. Load Data
    print("Loading epoched data and computing mixed source estimates...")
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
        
        stcs = apply_inverse_epochs(target_epochs, inverse_operator, lambda2=1.0/9.0, method='dSPM')
        
        # MNE MixedSourceEstimate objects contain both surface and volume sources.
        # To dynamically find the deep sources, we count the cortical vertices.
        # src[0] is Left Hemisphere cortex, src[1] is Right Hemisphere cortex.
        src = mne.read_source_spaces(src_file)
        n_cortical_vertices = src[0]['nuse'] + src[1]['nuse']
        
        for stc in stcs:
            # Check if this is a mixed source space containing volume data
            if stc.data.shape[0] > n_cortical_vertices:
                # Isolate the deep volume sources by slicing past the cortical vertices
                deep_sources = stc.data[n_cortical_vertices:, :]
                deep_data_list.append(deep_sources)
            else:
                print(f"Warning: Subject {s} inverse operator does not contain deep volume sources.")
                break
                
    if len(deep_data_list) == 0:
        print("No valid deep source data found on this machine! Cannot proceed with extraction.")
        return
        
    # Stack across epochs: Shape (epochs, deep_vertices, times)
    deep_data_array = np.stack(deep_data_list, axis=0)
    print(f"Extracted Deep Source Array Shape: {deep_data_array.shape}")
    
    # Calculate the mean across epochs for this condition
    deep_data_mean = np.nanmean(deep_data_array, axis=0)
    
    # Export to MATLAB (.mat) replicating the legacy DEEP script behavior
    out_file = os.path.join(output_dir, f"{condition}_all_deep_volume.mat")
    out_mean_file = os.path.join(output_dir, f"{condition}_all_deep_volume_mean.mat")
    
    sio.savemat(out_file, {'data': deep_data_array})
    sio.savemat(out_mean_file, {'data': deep_data_mean[np.newaxis, ...]})
    
    print(f"Successfully exported deep volume sources to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Export Deep Brain Volume Sources")
    parser.add_argument("--data_dir", type=str, default='./data/epochs/',
                        help="Directory containing epoched data (.fif)")
    parser.add_argument("--fwd_dir", type=str, default='./data/forward/',
                        help="Directory containing forward solutions/inverse operators")
    parser.add_argument("--out_dir", type=str, default='./stats_results/deep/',
                        help="Directory to save the MATLAB matrices")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02'],
                        help="Subjects to process")
    parser.add_argument("--condition", type=str, default='Easy',
                        help="Condition to test")
    parser.add_argument("--behavior_filter", type=str, default=None,
                        help="Pandas query string to filter behavior (e.g., 'nChoiceMade == nCorrectChoice')")
    
    args = parser.parse_args()
    run_batch_extract_deep_sources(
        args.data_dir, args.fwd_dir, args.out_dir, args.subjects, 
        args.condition, args.behavior_filter
    )
