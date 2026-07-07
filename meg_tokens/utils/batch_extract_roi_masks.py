import os
import argparse
import numpy as np
import mne
from meg_tokens.io import ensure_dir, require_file, save_array

def run_batch_extract_roi_masks(
    src_path: str,
    output_dir: str,
    roi_labels: list = None
):
    """
    Extracts vertex indices for specified Region of Interest (ROI) volume labels 
    from an MNE source space and saves them as .npy arrays.
    """
    print(f"=== Starting ROI Mask Extraction ===")
    
    if roi_labels is None:
        # Default to the deep subcortical structures used in the legacy pipeline
        roi_labels = [
            'Left-Accumbens-area', 'Left-Pallidum', 'Left-Caudate', 'Left-Putamen',
            'Left-Amygdala', 'Left-Thalamus-Proper', 'Left-Cerebellum-Cortex',
            'Brain-Stem',
            'Right-Accumbens-area', 'Right-Pallidum', 'Right-Caudate', 'Right-Putamen',
            'Right-Amygdala', 'Right-Thalamus-Proper', 'Right-Cerebellum-Cortex'
        ]
        
    output_path = ensure_dir(output_dir)
        
    print(f"Loading volume source space from: {src_path}")
    vol_src = mne.read_source_spaces(require_file(src_path, purpose="volume source space"))
    try:
        available_labels = mne.get_volume_labels_from_src(vol_src, 'fsaverage', 'aseg')
        print(f"Found {len(available_labels)} available volume labels in source space.")
    except Exception as e:
        print(f"Could not list volume labels from src: {e}")
            
    for label in roi_labels:
        print(f"Processing ROI: {label}")
        
        vertices = []
        for s in vol_src:
            if s.get('type') == 'vol' and 'seg_name' in s and label in s['seg_name']:
                vertices = s['vertno'][s['inuse'].astype(bool)]
                break

        if not len(vertices):
            raise ValueError(f"Could not locate ROI {label!r} in source space {src_path}")
            
        out_file = output_path / f"{label}_vertices.npy"
        save_array(out_file, np.asarray(vertices), dims=("vertex",), metadata={"roi": label, "source_space": str(src_path)})
        print(f"  -> Saved {len(vertices)} vertices to {out_file}")

    print("=== Extraction Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structural ROI vertex masks from an MNE Volume Source Space")
    parser.add_argument("--src_path", type=str, required=True,
                        help="Path to the MNE source space .fif file (e.g., fsaverage-vol-5-src.fif)")
    parser.add_argument("--out_dir", type=str, default='./roi_masks/',
                        help="Directory to save the exported .npy mask arrays")
    parser.add_argument("--rois", type=str, nargs='+', default=None,
                        help="List of explicit volume labels to extract (defaults to standard deep structures)")
    
    args = parser.parse_args()
    run_batch_extract_roi_masks(args.src_path, args.out_dir, args.rois)
