import os
import json
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Optional, Tuple

from meg_tokens.io import derivative_path, ensure_dir, save_table, sidecar_path
from meg_tokens.utils.batch_processor import normalize_subject_id
from meg_tokens.utils.tdms_parser import validate_behavior_dataframe


DEFAULT_EVENT_IDS = {
    'go': {'Go': 524288},
    'enter': {'Enter': 1048576},
    'feedback': {'Feedback': 2097152},
}

SUBJECT_EVENT_OVERRIDES = {
    'H06': {
        'go': {'Go': 262144},
    }
}


def get_event_id(align_to: str, subject_id: Optional[str] = None) -> dict:
    """Return event IDs, including known subject-specific legacy overrides."""
    key = align_to.lower()
    if key not in DEFAULT_EVENT_IDS:
        raise ValueError(f"Unknown alignment event: {align_to}")
    if subject_id is not None:
        subject = normalize_subject_id(subject_id)
        override = SUBJECT_EVENT_OVERRIDES.get(subject, {}).get(key)
        if override is not None:
            return override.copy()
    return DEFAULT_EVENT_IDS[key].copy()


def parse_run_label(run_id: str) -> Tuple[str, Optional[str]]:
    """Return numeric run and optional condition from labels like Slow1 or run-1."""
    text = str(run_id).strip()
    if text.lower().startswith("run-"):
        return text.split("-", 1)[1], None

    match = None
    import re
    match = re.fullmatch(r"([A-Za-z]+)?0*([0-9]+)", text)
    if match is None:
        cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "+"))
        return cleaned, None
    condition = match.group(1)
    run = str(int(match.group(2)))
    return run, condition.capitalize() if condition else None


def find_behavior_table(
    behavior_root: str,
    subject_id: str,
    run_id: str,
    condition: Optional[str] = None,
) -> Path:
    """Find a Stage 1 behavior TSV derivative for a subject/run."""
    subject = normalize_subject_id(subject_id)
    run, inferred_condition = parse_run_label(run_id)
    condition = condition or inferred_condition
    beh_dir = Path(behavior_root) / "derivatives" / "meg-tokens" / f"sub-{subject}" / "beh"
    if condition:
        pattern = f"sub-{subject}_task-tokens_run-{run}_desc-{condition.lower()}_beh.tsv"
        candidates = [beh_dir / pattern]
    else:
        candidates = sorted(beh_dir.glob(f"sub-{subject}_task-tokens_run-{run}_desc-*_beh.tsv"))

    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No behavior TSV derivative found for subject={subject}, run={run} under {behavior_root}")
    if len(existing) > 1:
        raise ValueError(f"Multiple behavior TSV derivatives matched subject={subject}, run={run}: {existing}")
    return existing[0]


def load_behavior_table(path: str) -> pd.DataFrame:
    """Load and validate a Stage 1 behavior TSV derivative."""
    df = pd.read_csv(path, sep="\t")
    validate_behavior_dataframe(df)
    return df


def synchronize_events_and_behavior(
    events: np.ndarray,
    event_ids: dict,
    behavior_df: pd.DataFrame,
    on_mismatch: str = "error",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Synchronize matching MEG events with behavior rows."""
    matching_codes = list(event_ids.values())
    matched_events = events[np.isin(events[:, 2], matching_codes)]

    n_events = len(matched_events)
    n_trials = len(behavior_df)
    if n_events == 0:
        raise ValueError(f"No MEG events matched event IDs: {event_ids}")

    if n_events != n_trials:
        message = f"Trial count mismatch: MEG events={n_events}, behavior rows={n_trials}"
        if on_mismatch == "error":
            raise ValueError(message)
        if on_mismatch == "truncate":
            n_keep = min(n_events, n_trials)
            matched_events = matched_events[:n_keep]
            behavior_df = behavior_df.iloc[:n_keep].copy()
        else:
            raise ValueError("on_mismatch must be 'error' or 'truncate'")

    metadata_df = behavior_df.copy().reset_index(drop=True)
    final_events = matched_events[:len(metadata_df)]
    metadata_df['meg_sample'] = final_events[:, 0]
    metadata_df['meg_trigger_value'] = final_events[:, 2]
    return final_events, metadata_df

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
    preload: bool = True,
    on_mismatch: str = "error",
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
    final_events, metadata_df = synchronize_events_and_behavior(events, event_ids, behavior_df, on_mismatch=on_mismatch)
    
    # Build Epochs
    epochs = mne.Epochs(
        raw,
        final_events,
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
    bids_format: bool = True,
    condition: Optional[str] = None,
    align_to: str = "go",
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
        subject = normalize_subject_id(subject_id)
        run, inferred_condition = parse_run_label(run_id)
        condition = condition or inferred_condition
        description = align_to if condition is None else f"{condition.lower()}-{align_to}"

        epochs_path = derivative_path(
            output_dir,
            subject=subject,
            datatype="meg",
            task="tokens",
            run=run,
            description=description,
            suffix="epo",
            extension=".fif",
        )
        events_tsv_path = derivative_path(
            output_dir,
            subject=subject,
            datatype="meg",
            task="tokens",
            run=run,
            description=description,
            suffix="events",
            extension=".tsv",
        )
        eve_path = derivative_path(
            output_dir,
            subject=subject,
            datatype="meg",
            task="tokens",
            run=run,
            description=description,
            suffix="eve",
            extension=".eve",
        )
        ensure_dir(epochs_path.parent)
        
        # Save MNE epochs file
        epochs.save(str(epochs_path), overwrite=True)
        
        # Save standard MNE .eve file
        mne.write_events(str(eve_path), epochs.events, overwrite=True)
        
        # Save BIDS events.tsv file
        # Columns: onset, duration, trial_type, value, plus all behavioral metadata columns
        sfreq = epochs.info['sfreq']
        events_data = epochs.events
        
        event_name_by_value = {value: name for name, value in epochs.event_id.items()}
        bids_events = pd.DataFrame({
            'onset': events_data[:, 0] / sfreq,
            'duration': np.zeros(len(events_data)),
            'sample': events_data[:, 0],
            'trial_type': [event_name_by_value.get(value, str(value)) for value in events_data[:, 2]],
            'value': events_data[:, 2]
        })
        
        # Merge behavior metadata if present
        if epochs.metadata is not None:
            # Drop columns that are already present or internal to meg
            meta_clean = epochs.metadata.drop(columns=['meg_sample', 'meg_trigger_value'], errors='ignore')
            # Reset index to align rows
            meta_clean = meta_clean.reset_index(drop=True)
            bids_events = pd.concat([bids_events, meta_clean], axis=1)
            
        save_table(
            events_tsv_path,
            bids_events,
            metadata={
                "stage": "epoching",
                "subject": subject,
                "condition": condition,
                "run": run,
                "alignment": align_to,
                "tmin": epochs.tmin,
                "tmax": epochs.tmax,
                "sfreq": sfreq,
            },
        )

        with sidecar_path(epochs_path).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "mne-epochs-fif",
                    "stage": "epoching",
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "alignment": align_to,
                    "n_epochs": len(epochs),
                    "event_id": epochs.event_id,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        
        return {
            'epochs': str(epochs_path),
            'events_tsv': str(events_tsv_path),
            'eve': str(eve_path)
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
