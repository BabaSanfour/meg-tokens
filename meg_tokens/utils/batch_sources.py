"""
Pipeline execution script for Stage 4: Neural Source Localization.
"""

import os
import glob
import mne
from meg_tokens.meg.sources import (
    compute_noise_covariance,
    setup_bem_solution,
    setup_mixed_source_space
)

def run_sources_pipeline(
    subjects_list: list,
    raw_dir: str,
    subjects_dir: str,
    out_dir: str,
    spacing: str = 'oct6'
):
    for subject in subjects_list:
        print(f"=== Running Source Localization for {subject} ===")
        
        # 1. Compute Noise Covariance (expects empty room noise file)
        noise_files = glob.glob(os.path.join(raw_dir, subject, '*noise*.ds'))
        if not noise_files:
            noise_files = glob.glob(os.path.join(raw_dir, subject, '*noise*.fif'))
            
        if not noise_files:
            print(f"No empty-room noise file found for {subject}")
            continue
            
        cov = compute_noise_covariance(noise_files[0])
        
        # 2. BEM Solution
        try:
            bem = setup_bem_solution(subject, subjects_dir=subjects_dir)
        except Exception as e:
            print(f"Failed to create BEM for {subject}: {e}")
            continue
            
        # 3. Source Space
        src = setup_mixed_source_space(subject, subjects_dir=subjects_dir, spacing=spacing, bem=bem)
        
        # Save output
        subject_out = os.path.join(out_dir, subject)
        os.makedirs(subject_out, exist_ok=True)
        mne.write_cov(os.path.join(subject_out, f"{subject}-noise-cov.fif"), cov)
        mne.write_bem_solution(os.path.join(subject_out, f"{subject}-bem.fif"), bem)
        mne.write_source_spaces(os.path.join(subject_out, f"{subject}-{spacing}-src.fif"), src, overwrite=True)
        
        print(f"Saved Source Models for {subject}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run source localization pipeline.")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01'])
    parser.add_argument("--raw_dir", type=str, default='/media/external/DDM/MEG_data/')
    parser.add_argument("--subjects_dir", type=str, default='/media/external/DDM/IRM/')
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/source_rec/')
    parser.add_argument("--spacing", type=str, default='oct6')
    args = parser.parse_args()
    
    run_sources_pipeline(args.subjects, args.raw_dir, args.subjects_dir, args.out_dir, args.spacing)
