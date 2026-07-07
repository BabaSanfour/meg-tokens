"""
Pipeline execution script for Stage 2: Epoch Extraction & Event Alignment.
"""

import os
import glob
import pandas as pd
import numpy as np
import mne
from meg_tokens.utils.epochs_builder import build_epochs_with_metadata, save_epochs_and_events

def run_epochs_pipeline(
    subjects_list: list,
    raw_dir: str,
    behavior_dir: str,
    out_dir: str,
    tmin: float = -0.5,
    tmax: float = 2.0,
    align_to: str = 'go'
):
    # DDM specific event dicts (simplified)
    event_dicts = {
        'go': {'Go': 524288},
        'enter': {'Enter': 1048576},
        'feedback': {'Feedback': 2097152}
    }
    event_id = event_dicts.get(align_to, {'Event': 1})
    
    for subject in subjects_list:
        print(f"=== Building Epochs for {subject} ===")
        
        # This script expects pre-filtered raw files and csv behavior files
        raw_files = glob.glob(os.path.join(raw_dir, subject, '*_filt_raw.fif'))
        if not raw_files:
            print(f"No filtered raw files found for {subject}")
            continue
            
        for raw_path in raw_files:
            run_id = os.path.basename(raw_path).split('_')[1]
            csv_path = os.path.join(behavior_dir, subject, f"{subject}_{run_id}.csv")
            
            if not os.path.exists(csv_path):
                continue
                
            raw = mne.io.read_raw_fif(raw_path, preload=True)
            events = mne.find_events(raw)
            behavior_df = pd.read_csv(csv_path)
            
            epochs = build_epochs_with_metadata(
                raw=raw, events=events, event_ids=event_id,
                tmin=tmin, tmax=tmax, behavior_df=behavior_df
            )
            
            save_epochs_and_events(epochs, out_dir, subject, run_id, bids_format=False)
            print(f"Saved {subject} {run_id} epochs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run epoching pipeline.")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01'])
    parser.add_argument("--raw_dir", type=str, default='/media/external/DDM/MEG_data/')
    parser.add_argument("--behavior_dir", type=str, default='/media/external/DDM/dataframes/')
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/epoched/')
    parser.add_argument("--align_to", type=str, default='go')
    args = parser.parse_args()
    
    run_epochs_pipeline(args.subjects, args.raw_dir, args.behavior_dir, args.out_dir, align_to=args.align_to)
