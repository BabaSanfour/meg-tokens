"""
Pipeline execution script for Stage 5: Time-Frequency & Spectrogram Analysis.
Iterates over subjects to extract frequency-band power spectrograms from epoched data.
"""

import os
import glob
import numpy as np
from meg_tokens.meg.time_frequency import compute_band_power, DEFAULT_BANDS

def run_time_frequency_pipeline(
    subjects_list: list,
    in_dir: str,
    out_dir: str,
    align_to: str = 'go',
    method: str = 'hilbert',
    width: int = 400,
    step: int = 110,
    n_jobs: int = 1
):
    """
    Executes the time-frequency power extraction pipeline across multiple subjects.
    """
    sfreq = 1200.0  # Default sampling frequency for MEG data

    for subject in subjects_list:
        print(f"=== Processing Time-Frequency for subject: {subject} ===")
        
        # Load epochs (.npy files) for the subject
        # Using a standard path structure: /MEG_data/<subject>/MEG_data_epoched_<align_to>/*.npy
        align_suffix = f"_{align_to}" if align_to != "go" else ""
        subject_in_dir = os.path.join(in_dir, subject, f"MEG_data_epoched{align_suffix}")
        
        if not os.path.exists(subject_in_dir):
            print(f"Input directory not found for {subject}: {subject_in_dir}")
            continue
            
        epoch_files = glob.glob(os.path.join(subject_in_dir, "*.npy"))
        if not epoch_files:
            print(f"No epoch files found for {subject} in {subject_in_dir}")
            continue

        subject_out_dir = os.path.join(out_dir, subject, align_to, 'feature', f'power_{width}_{step}')
        os.makedirs(subject_out_dir, exist_ok=True)

        for file_path in epoch_files:
            filename = os.path.basename(file_path).replace('.npy', '')
            print(f"  -> Processing file: {filename}")
            
            # Load raw epoched data
            # Assuming shape (n_channels, n_times, n_trials) based on legacy structure
            data = np.load(file_path)
            
            if data.ndim == 3:
                # Transpose to (n_channels, n_trials, n_times) for the compute function if needed
                # (Legacy scripts often saved as n_channels, n_times, n_trials)
                data = np.transpose(data, (0, 2, 1))

            # Compute power across all bands
            power_results = compute_band_power(
                data=data,
                sfreq=sfreq,
                freq_bands=DEFAULT_BANDS,
                method=method,
                width=width,
                step=step,
                n_jobs=n_jobs,
                return_mne=False
            )

            # Save the results for each frequency band
            for band_name, power_data in power_results.items():
                out_filename = f"{filename}_{band_name}_Power.npy"
                out_path = os.path.join(subject_out_dir, out_filename)
                
                # Squeeze back to legacy shape formats if desired
                power_data = np.squeeze(power_data)
                
                np.save(out_path, power_data)
                print(f"     Saved: {out_filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run time-frequency power extraction pipeline across subjects."
    )
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01', 'H02', 'H03'],
                        help="List of subject IDs to process (e.g., --subjects H01 H02 H03). Default: H01 H02 H03.")
    parser.add_argument("--in_dir", type=str, default='/media/external/DDM/MEG_data/',
                        help="Base directory containing the epoched MEG data (default: /media/external/DDM/MEG_data/)")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/MEG_data/',
                        help="Output base directory to save power features (default: /media/external/DDM/MEG_data/)")
    parser.add_argument("--align_to", type=str, default='go', choices=['go', 'enter', 'feedback'],
                        help="Event the epochs are aligned to (e.g., 'go', 'enter'). Default: 'go'.")
    parser.add_argument("--method", type=str, default='hilbert', choices=['hilbert', 'morlet', 'multitaper'],
                        help="Power computation method (default: hilbert)")
    parser.add_argument("--width", type=int, default=400,
                        help="Sliding window width in samples (default: 400)")
    parser.add_argument("--step", type=int, default=110,
                        help="Sliding window step in samples (default: 110)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs to use (default: 1)")
    
    args = parser.parse_args()
    
    run_time_frequency_pipeline(
        subjects_list=args.subjects,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        align_to=args.align_to,
        method=args.method,
        width=args.width,
        step=args.step,
        n_jobs=args.n_jobs
    )
