"""
Pipeline execution script for Stage 2: Epoch Extraction & Event Alignment.
"""

import os
import glob
import pandas as pd
import numpy as np
import mne
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.epochs_builder import (
    build_epochs_with_metadata,
    find_behavior_table,
    get_event_id,
    load_behavior_table,
    parse_run_label,
    save_epochs_and_events,
)


def infer_run_id_from_raw(raw_path: str) -> str:
    """Infer run label from BIDS-style or legacy filtered raw filenames."""
    name = os.path.basename(raw_path)
    import re
    match = re.search(r'run-([A-Za-z0-9]+)', name)
    if match:
        return match.group(1)
    legacy = re.search(r'_(Slow|Fast|RT)?([0-9]+)_?(?:filt|clean)?_?raw\.fif$', name, re.IGNORECASE)
    if legacy:
        prefix = legacy.group(1) or ""
        return f"{prefix}{legacy.group(2)}"
    raise ValueError(f"Could not infer run from raw filename: {raw_path}")


def find_raw_files(raw_dir: str, subject: str) -> list:
    subject = normalize_subject_id(subject)
    patterns = [
        os.path.join(raw_dir, "derivatives", "meg-tokens", f"sub-{subject}", "meg", "*_raw.fif"),
        os.path.join(raw_dir, f"sub-{subject}", "meg", "*_raw.fif"),
        os.path.join(raw_dir, subject, "*_filt_raw.fif"),
        os.path.join(raw_dir, subject, "*_clean_raw.fif"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))

def run_epochs_pipeline(
    subjects_list: list,
    raw_dir: str,
    behavior_dir: str,
    out_dir: str,
    tmin: float = -0.5,
    tmax: float = 2.0,
    align_to: str = 'go'
):
    for subject in subjects_list:
        subject = normalize_subject_id(subject)
        print(f"=== Building Epochs for {subject} ===")
        
        raw_files = find_raw_files(raw_dir, subject)
        if not raw_files:
            raise FileNotFoundError(f"No cleaned/filtered raw FIF files found for {subject} under {raw_dir}")
            
        for raw_path in raw_files:
            run_id = infer_run_id_from_raw(raw_path)
            run, condition = parse_run_label(run_id)
            behavior_path = find_behavior_table(behavior_dir, subject, run, condition=condition)
            event_id = get_event_id(align_to, subject)
                
            raw = mne.io.read_raw_fif(raw_path, preload=True)
            events = mne.find_events(raw)
            behavior_df = load_behavior_table(str(behavior_path))
            
            epochs = build_epochs_with_metadata(
                raw=raw, events=events, event_ids=event_id,
                tmin=tmin, tmax=tmax, behavior_df=behavior_df,
                on_mismatch="error",
            )
            
            save_epochs_and_events(
                epochs,
                out_dir,
                subject,
                run,
                bids_format=True,
                condition=condition,
                align_to=align_to,
            )
            print(f"Saved {subject} run-{run} {align_to} epochs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run epoching pipeline.")
    parser.add_argument("--subjects", type=str, nargs='+', default=['H01'])
    parser.add_argument("--raw_dir", type=str, default='/media/external/DDM/MEG_data/')
    parser.add_argument("--behavior_dir", type=str, default='/media/external/DDM/tokens-bids/')
    parser.add_argument("--out_dir", type=str, default='/media/external/DDM/epoched/')
    parser.add_argument("--align_to", type=str, default='go', choices=['go', 'enter', 'feedback'])
    args = parser.parse_args()
    
    run_epochs_pipeline(args.subjects, args.raw_dir, args.behavior_dir, args.out_dir, align_to=args.align_to)
