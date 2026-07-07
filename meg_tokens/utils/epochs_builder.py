import os
import numpy as np
import pandas as pd
import mne

def build_epochs_with_metadata(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_ids: dict,
    tmin: float,
    tmax: float,
    behavior_df: pd.DataFrame,
    picks: np.ndarray = None,
    baseline: tuple = None,
    reject: dict = None,
    preload: bool = True
) -> mne.Epochs:
    """
    Builds an MNE Epochs object with behavioral dataframe attached as metadata.
    Handles alignment and shape mismatch between events and behavioral trials.
    
    Args:
        raw: Filtered mne.io.Raw object.
        events: Events array returned by mne.find_events().
        event_ids: Event IDs dictionary (e.g. {"Go": 524288}).
        tmin: Start time before event in seconds.
        tmax: End time after event in seconds.
        behavior_df: Pandas DataFrame containing behavioral log trials.
        picks: Channel selection indices.
        baseline: Baseline correction time interval (e.g. (None, 0)).
        reject: Rejection thresholds (e.g. dict(mag=5e-12)).
        preload: If True, load data into memory.
        
    Returns:
        mne.Epochs object with synced metadata.
    """
    # Filter raw events to find the ones matching our event_ids
    matching_codes = list(event_ids.values())
    matched_events_mask = np.isin(events[:, 2], matching_codes)
    matched_events = events[matched_events_mask]
    
    n_events = len(matched_events)
    n_trials = len(behavior_df)
    
    if n_events != n_trials:
        print(f"Warning: Trial count mismatch! MEG events: {n_events}, Behavioral trials: {n_trials}")
        if n_events < n_trials:
            print(f"Aligning: Truncating behavioral trials to match the {n_events} MEG events.")
            behavior_df = behavior_df.iloc[:n_events].copy()
        else:
            print(f"Aligning: Selecting only the first {n_trials} matching MEG events.")
            # Find the indices of the first n_trials matching events in the original events array
            matched_indices = np.where(matched_events_mask)[0][:n_trials]
            # Construct a mask for the original events array to only keep these indices and non-matching events
            keep_mask = np.ones(len(events), dtype=bool)
            all_matched_indices = np.where(matched_events_mask)[0]
            indices_to_drop = all_matched_indices[n_trials:]
            keep_mask[indices_to_drop] = False
            events = events[keep_mask]
            
    # Attach events sample indices to the behavior metadata
    # We copy to avoid modifying the original dataframe
    metadata_df = behavior_df.copy()
    
    # We slice matched_events to match metadata_df length in case of any remaining minor mismatch
    final_matched_events = events[np.isin(events[:, 2], matching_codes)][:len(metadata_df)]
    metadata_df['meg_sample'] = final_matched_events[:, 0]
    metadata_df['meg_trigger_value'] = final_matched_events[:, 2]
    
    # Build Epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=baseline,
        reject=reject,
        preload=preload,
        metadata=metadata_df,
        reject_by_annotation=False
    )
    
    return epochs

def save_epochs_and_events(
    epochs: mne.Epochs,
    output_dir: str,
    subject_id: str,
    run_id: str,
    bids_format: bool = True
) -> dict:
    """
    Saves Epochs and event files. Supports standard MNE files and BIDS-compliant tsv formats.
    
    Args:
        epochs: mne.Epochs object (with metadata attached).
        output_dir: Directory where files should be written.
        subject_id: Subject label (e.g. 'H1', 'H32').
        run_id: Run or condition label (e.g. 'Slow1', 'RT2').
        bids_format: If True, uses BIDS-compliant directory structure and filenames.
        
    Returns:
        Dict containing saved file paths.
    """
    if bids_format:
        # BIDS Folder Structure: sub-<label>/meg/
        sub_label = f"sub-{subject_id}"
        run_label = f"run-{run_id}".lower().replace("_", "")
        
        meg_dir = os.path.join(output_dir, sub_label, "meg")
        os.makedirs(meg_dir, exist_ok=True)
        
        epochs_filename = f"{sub_label}_task-tokens_{run_label}_epo.fif"
        events_filename = f"{sub_label}_task-tokens_{run_label}_events.tsv"
        eve_filename = f"{sub_label}_task-tokens_{run_label}_eve.eve"
        
        epochs_path = os.path.join(meg_dir, epochs_filename)
        events_tsv_path = os.path.join(meg_dir, events_filename)
        eve_path = os.path.join(meg_dir, eve_filename)
        
        # Save MNE epochs file
        epochs.save(epochs_path, overwrite=True)
        
        # Save standard MNE .eve file
        mne.write_events(eve_path, epochs.events, overwrite=True)
        
        # Save BIDS events.tsv file
        # Columns: onset, duration, trial_type, value, plus all behavioral metadata columns
        sfreq = epochs.info['sfreq']
        events_data = epochs.events
        
        bids_events = pd.DataFrame({
            'onset': events_data[:, 0] / sfreq,
            'duration': np.zeros(len(events_data)),
            'trial_type': [list(epochs.event_id.keys())[0]] * len(events_data),
            'value': events_data[:, 2]
        })
        
        # Merge behavior metadata if present
        if epochs.metadata is not None:
            # Drop columns that are already present or internal to meg
            meta_clean = epochs.metadata.drop(columns=['meg_sample', 'meg_trigger_value'], errors='ignore')
            # Reset index to align rows
            meta_clean = meta_clean.reset_index(drop=True)
            bids_events = pd.concat([bids_events, meta_clean], axis=1)
            
        bids_events.to_csv(events_tsv_path, sep='\t', index=False)
        
        return {
            'epochs': epochs_path,
            'events_tsv': events_tsv_path,
            'eve': eve_path
        }
    else:
        # Standard format
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"{subject_id}_{run_id}"
        epochs_path = os.path.join(output_dir, f"{base_name}-epo.fif")
        eve_path = os.path.join(output_dir, f"{base_name}-eve.eve")
        
        # Save epochs
        epochs.save(epochs_path, overwrite=True)
        
        # Save events
        mne.write_events(eve_path, epochs.events, overwrite=True)
        
        return {
            'epochs': epochs_path,
            'eve': eve_path
        }
