"""
Pipeline execution script for computing Power Spectral Density (PSD) and
modeling aperiodic/periodic components using FOOOF.
"""

import os
import glob
import numpy as np
import mne
from meg_tokens.meg.time_frequency import compute_psd, fit_fooof

def run_psd_fooof(
    epochs_dir: str,
    output_dir: str,
    subjects_list: list = None,
    fmin: float = 1.0,
    fmax: float = 100.0,
    method: str = 'welch'
):
    print(f"=== Running PSD & FOOOF ({method}) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if subjects_list is None:
        subjects_list = sorted([d.split('_')[0] for d in os.listdir(epochs_dir) if d.startswith('H') and d.endswith('.fif')])
        # unique subjects
        subjects_list = sorted(list(set(subjects_list)))
        
    for subject in subjects_list:
        print(f"Processing Subject: {subject}")
        # Find epoched fif files
        fif_files = glob.glob(os.path.join(epochs_dir, f"{subject}*.fif"))
        
        subject_psds = []
        freqs = None
        
        for fif in fif_files:
            try:
                epochs = mne.read_epochs(fif, preload=True, verbose=False)
                # Compute PSD
                psds, freqs = compute_psd(epochs, fmin=fmin, fmax=fmax, method=method)
                # Average across epochs (dimension 0) -> shape (n_channels, n_freqs)
                psds_mean = psds.mean(axis=0)
                subject_psds.append(psds_mean)
            except Exception as e:
                print(f"Error processing {fif}: {e}")
                
        if not subject_psds:
            continue
            
        # Average across runs -> shape (n_channels, n_freqs)
        subject_psds = np.mean(subject_psds, axis=0)
        
        # Save raw PSDs
        np.save(os.path.join(output_dir, f"{subject}_psd_{int(fmin)}_{int(fmax)}.npy"), subject_psds)
        np.save(os.path.join(output_dir, f"{subject}_freqs_{int(fmin)}_{int(fmax)}.npy"), freqs)
        
        # Run FOOOF Group fitting (across channels)
        try:
            fg = fit_fooof(freqs, subject_psds, freq_range=[fmin, fmax])
            # Save FOOOF report
            fg.save_report(os.path.join(output_dir, f"{subject}_fooof_report"))
            print(f"  -> Saved PSD and FOOOF report for {subject}.")
        except ImportError:
            print("  -> FOOOF is not installed. Saved PSDs only.")
        except Exception as e:
            print(f"  -> Error fitting FOOOF for {subject}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PSD computation and FOOOF fitting.")
    parser.add_argument("--subjects", type=str, nargs='+', default=None, help="List of subjects (e.g. H01 H02)")
    parser.add_argument("--epochs_dir", type=str, default='/media/external/DDM/MEG_data_epoched_enter/', help="Directory containing .fif epochs")
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/PSDs/', help="Output directory for PSDs and FOOOF reports")
    parser.add_argument("--method", type=str, default='welch', choices=['welch', 'multitaper'], help="PSD method")
    parser.add_argument("--fmin", type=float, default=1.0, help="Minimum frequency")
    parser.add_argument("--fmax", type=float, default=100.0, help="Maximum frequency")
    args = parser.parse_args()
    
    run_psd_fooof(
        epochs_dir=args.epochs_dir,
        output_dir=args.out_dir,
        subjects_list=args.subjects,
        fmin=args.fmin,
        fmax=args.fmax,
        method=args.method
    )
