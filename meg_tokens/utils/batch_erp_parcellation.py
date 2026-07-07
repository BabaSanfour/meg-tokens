"""
Pipeline execution script for Stage 6: ERP Slicing, Parcellation, & Export.
Iterates over subjects to extract source estimates, align to trial events, parcellate into ROIs, and export data.
"""

import os
import glob
import mne
import pandas as pd
from meg_tokens.meg.erp import align_and_pad_epochs, parcellate_source_estimates, export_neural_space

def run_erp_parcellation_pipeline(
    subjects_list: list,
    stc_base_dir: str,
    behavior_dir: str,
    subjects_dir: str,
    output_dir: str,
    align_to: str = 'go',
    min_rt_ms: float = 100.0
):
    """
    Executes the ERP alignment and parcellation pipeline across multiple subjects.
    """
    for subject in subjects_list:
        print(f"=== Processing ERP Parcellation for subject: {subject} ===")
        
        # Load behavior df for the subject (example path format)
        df_paths = glob.glob(os.path.join(behavior_dir, subject, f"{subject}_*.csv"))
        if not df_paths:
            print(f"No behavioral dataframes found for {subject}")
            continue
            
        for df_path in df_paths:
            df = pd.read_csv(df_path)
            condition = os.path.basename(df_path).replace('.csv', '')
            
            # Example stc load path (needs to be adapted to true project paths)
            stc_path = os.path.join(stc_base_dir, subject, f"{condition}-lh.stc")
            if not os.path.exists(stc_path):
                continue
                
            stc = mne.read_source_estimate(stc_path)
            # Make sure it's a list since align_and_pad expects list of STCs per trial,
            # Or if it's already an epoch STC, split it. This is a scaffold.
            # Here we assume a simplified call:
            
            # 1. Align and Pad
            aligned = align_and_pad_epochs([stc], df, align_to=align_to, min_rt_ms=min_rt_ms)
            
            if not aligned or aligned[0] is None:
                continue
                
            # Convert back to STC for parcellation
            aligned_stc = stc.copy()
            aligned_stc.data = aligned[0]
            
            # 2. Parcellate
            label_names, parc_data = parcellate_source_estimates(
                aligned_stc, subjects_dir=subjects_dir, subject='fsaverage', parc='HCPMMP1', hemi='both'
            )
            
            # 3. Export
            out_prefix = f"{subject}_{condition}_{align_to}_parcellated"
            export_neural_space(parc_data, label_names, os.path.join(output_dir, subject), out_prefix, format='both')
            print(f"Exported {out_prefix}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ERP slicing, parcellation, and export pipeline across subjects."
    )
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02', 'H03'],
                        help="List of subject IDs to process (e.g., --subjects H01 H02 H03). Default: H01 H02 H03.")
    parser.add_argument("--stc_dir", type=str, default='/media/external/DDM/source_rec/',
                        help="Directory containing source estimates (default: /media/external/DDM/source_rec/)")
    parser.add_argument("--behavior_dir", type=str, default='/media/external/DDM/dataframes/',
                        help="Directory containing the behavioral dataframes (default: /media/external/DDM/dataframes/)")
    parser.add_argument("--subjects_dir", type=str, default='/media/external/DDM/IRM/',
                        help="Freesurfer subjects directory (default: /media/external/DDM/IRM/)")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/export_erp/',
                        help="Output directory to save parcellated data (default: /media/external/DDM/export_erp/)")
    parser.add_argument("--align_to", type=str, default='go', choices=['go', 'enter', 'after', 'after_2'],
                        help="Event to align epochs to (e.g., 'go', 'enter'). Default: 'go'.")
    parser.add_argument("--min_rt_ms", type=float, default=100.0,
                        help="Minimum reaction time in ms to filter trials (default: 100.0)")
    
    args = parser.parse_args()
    
    run_erp_parcellation_pipeline(
        subjects_list=args.subjects,
        stc_base_dir=args.stc_dir,
        behavior_dir=args.behavior_dir,
        subjects_dir=args.subjects_dir,
        output_dir=args.out_dir,
        align_to=args.align_to,
        min_rt_ms=args.min_rt_ms
    )
