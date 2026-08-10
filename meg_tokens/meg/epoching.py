"""Event alignment, epoch construction, and epoch derivative persistence."""

import os
import json
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Optional, Tuple

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.behavior.trials import started_trials
from meg_tokens.core import normalize_subject_id, parse_run_label
from meg_tokens.io import DerivativeLayout, ensure_dir, save_table, sidecar_path


DEFAULT_EVENT_IDS = {
    'go': {'Go': 524288},
    'start': {'Start': 262144},
    'enter': {'Enter': 1048576},
    'feedback': {'Feedback': 2097152},
}

SUBJECT_EVENT_OVERRIDES = {
    'H06': {
        'go': {'Go': 262144},
        'start': {'Start': 524288},
    }
}

KNOWN_TRAILING_TRIAL_MISMATCHES = {
    ('H01', 'Fast', '2'), ('H01', 'Fast', '3'), ('H01', 'Fast', '4'), ('H01', 'RT', '1'), ('H01', 'RT', '2'), ('H01', 'Slow', '3'),
    ('H01', 'Slow', '4'), ('H03', 'RT', '1'), ('H03', 'RT', '2'), ('H03', 'Slow', '4'), ('H04', 'RT', '1'),
    ('H05', 'RT', '1'), ('H05', 'RT', '2'), ('H05', 'Slow', '2'), ('H06', 'Fast', '3'), ('H06', 'RT', '1'), ('H06', 'RT', '2'),
    ('H06', 'Slow', '2'), ('H07', 'RT', '1'), ('H07', 'RT', '2'), ('H08', 'RT', '1'), ('H08', 'RT', '2'), ('H08', 'Slow', '2'),
    ('H09', 'RT', '1'), ('H09', 'RT', '2'), ('H10', 'Fast', '2'), ('H10', 'RT', '1'), ('H10', 'RT', '2'), ('H11', 'Fast', '1'),
    ('H11', 'Fast', '2'), ('H11', 'Fast', '3'), ('H11', 'RT', '1'), ('H11', 'RT', '2'), ('H12', 'RT', '1'), ('H12', 'RT', '2'),
    ('H12', 'Slow', '1'), ('H13', 'RT', '1'), ('H13', 'RT', '2'), ('H14', 'RT', '1'), ('H14', 'RT', '2'), ('H15', 'RT', '1'),
    ('H15', 'RT', '2'), ('H15', 'Slow', '2'), ('H16', 'RT', '1'), ('H16', 'RT', '2'), ('H17', 'Fast', '3'), ('H17', 'RT', '1'),
    ('H17', 'RT', '2'), ('H18', 'Fast', '3'), ('H18', 'RT', '1'), ('H18', 'RT', '2'), ('H18', 'Slow', '1'), ('H19', 'Fast', '4'),
    ('H19', 'RT', '1'), ('H19', 'RT', '2'), ('H19', 'Slow', '2'), ('H19', 'Slow', '3'), ('H20', 'RT', '1'), ('H20', 'RT', '2'),
    ('H20', 'Slow', '1'), ('H21', 'Fast', '1'), ('H21', 'RT', '1'), ('H21', 'RT', '2'), ('H22', 'Fast', '4'), ('H22', 'RT', '1'),
    ('H22', 'RT', '2'), ('H23', 'Fast', '1'), ('H23', 'Fast', '2'), ('H23', 'RT', '1'), ('H23', 'RT', '2'), ('H23', 'Slow', '1'),
    ('H23', 'Slow', '3'), ('H23', 'Slow', '4'), ('H24', 'RT', '1'), ('H24', 'RT', '2'), ('H24', 'Slow', '3'), ('H24', 'Slow', '4'),
    ('H25', 'Fast', '1'), ('H25', 'Fast', '3'), ('H25', 'RT', '1'), ('H25', 'RT', '2'), ('H25', 'Slow', '3'), ('H26', 'Fast', '1'),
    ('H26', 'Fast', '2'), ('H26', 'Fast', '3'), ('H26', 'RT', '1'), ('H26', 'RT', '2'), ('H26', 'Slow', '2'), ('H26', 'Slow', '3'),
    ('H27', 'Fast', '2'), ('H27', 'Fast', '4'), ('H27', 'RT', '1'), ('H27', 'RT', '2'), ('H27', 'Slow', '2'), ('H27', 'Slow', '3'),
    ('H27', 'Slow', '4'), ('H28', 'RT', '1'), ('H28', 'RT', '2'), ('H29', 'RT', '1'), ('H29', 'RT', '2'), ('H29', 'Slow', '1'),
    ('H29', 'Slow', '3'), ('H29', 'Slow', '4'), ('H30', 'Fast', '2'), ('H30', 'Fast', '3'), ('H30', 'RT', '1'), ('H30', 'RT', '2'),
    ('H30', 'Slow', '1'), ('H30', 'Slow', '2'), ('H31', 'RT', '1'), ('H31', 'RT', '2'), ('H32', 'Fast', '4'), ('H32', 'RT', '1'),
    ('H32', 'RT', '2'),
}


def mismatch_policy(subject: str, condition: Optional[str], run: str) -> str:
    """Return the on_mismatch policy for synchronize_events_and_behavior.

    Defaults to "error" for every run except the confirmed-safe ones in
    KNOWN_TRAILING_TRIAL_MISMATCHES, which resolve to "truncate".

    Those 114 runs all show the same structural pattern at their trailing
    boundary: after excluding OUTCOME_NEVER_STARTED trials, the recording's
    trial-start pulses align with TDMS's own trial sequence for their entire
    length except at the very end, where either MEG has one or more extra
    trial-start pulses with no behavioral log entry (a trial that was
    triggered but never got written -- e.g. H03RT1, delta +1) or TDMS logged
    trials MEG never captured a matching start pulse for (e.g. H06Slow2,
    delta -12). Every one of these 114 runs was individually verified two
    independent ways: (1) an exhaustive per-subject scan (every .ds file for
    that subject, not just the nearest-timestamp guess) finds one alignment
    offset that separates from the next-best by 100-1000x, at sub-ms mean
    error -- that scan is now
    ``meg_tokens.meg.raw_staging.fingerprint_candidates``, which Stage 0 uses
    to identify recordings in the first place; (2) the
    trial-start->go-cue latency on the aligned (non-boundary)
    portion matches TDMS tGO to within 1.6-4.25ms across all of them, zero
    exceptions -- confirming the alignment is correct, not coincidental.
    Full per-run evidence is in docs/meg.md. `truncate`
    (drop the extra tail from whichever side is longer) is safe for all of
    them because the discrepancy is confirmed to sit only at the trailing
    boundary, never scattered through the run and never at the front.

    One run, H12Slow2, has irregularities at *both* ends and is deliberately
    excluded here pending manual review, not auto-truncated. Runs needing a
    reconstructed go-cue rather than a trimmed trial -- H02RT1 (its entire
    go channel is missing; the same trailing-start-pulse surplus is present
    but reconstruction handles it directly, so it is intentionally *not*
    listed here) and 20 others with a single interior dropout -- are handled
    separately by KNOWN_GO_RECONSTRUCTION_RUNS / reconstruct_missing_go_events.
    """
    key = (normalize_subject_id(subject), condition, str(run))
    return "truncate" if key in KNOWN_TRAILING_TRIAL_MISMATCHES else "error"


# Trials with no MEG event at all -- no trial-start pulse, let alone a
# go-cue -- despite being a real, completed trial in TDMS. Distinct from
# OUTCOME_NEVER_STARTED (which is about the behavioral log, not the MEG
# recording) and from KNOWN_TRAILING_TRIAL_MISMATCHES (a single boundary
# discrepancy, fixable by truncation): these trials sit in the *middle* of
# an otherwise-recoverable run, unrecoverable because the MEG recording
# itself doesn't cover them. Currently just H12Slow2 trial 2: the recording
# started about 3 trials after TDMS logging began that file, missing
# trials 1-3 entirely; trials 1 and 3 are already excluded as
# OUTCOME_NEVER_STARTED, leaving trial 2 -- a real, normal trial -- as the
# only one needing this explicit exclusion. Verified: after dropping it, the
# remaining 57 trials match MEG's 57 go events exactly, no truncation
# needed (the run's other artifact, one trailing extra trial-start pulse
# with no go-cue, is already invisible to go-alignment since it's never a
# candidate "Go" event). See docs/meg.md.
KNOWN_UNRECOVERABLE_TRIALS = {
    ('H12', 'Slow', '2'): {2},
}


def exclude_unrecoverable_trials(
    subject: str, condition: Optional[str], run: str, behavior_df: pd.DataFrame
) -> pd.DataFrame:
    """Drop trials in KNOWN_UNRECOVERABLE_TRIALS for this run, if any.

    Call before mismatch_policy/synchronize_events_and_behavior so the
    trial-count comparison reflects what MEG can actually cover.
    """
    key = (normalize_subject_id(subject), condition, str(run))
    trial_indices = KNOWN_UNRECOVERABLE_TRIALS.get(key)
    if not trial_indices:
        return behavior_df
    return behavior_df.loc[~behavior_df['nTrialIndex'].isin(trial_indices)]


# Runs where one or more trials have a real trial-start pulse but no go-cue
# pulse in the raw MEG recording -- a genuine trigger dropout. H02RT1 is the
# extreme case (its entire go channel is missing, all ~39 trials); in the
# other 20 the dropout is a single trial, and in every one of those checked
# it is the run's *final* trial: the recording stopped between that trial's
# start pulse and its go cue. Verified per-run: trial-start pulses align with
# TDMS's nInitialTime sequence at sub-ms precision across the whole run, and
# for the trials that *do* have a go-cue, its latency after trial-start
# matches TDMS tGO to ~2-4ms (measured at 3.20 +/- 1.19ms over 2230 real
# trials) -- the same calibration reconstruct_missing_go_events uses to fill
# the gap. Reproduce with scripts/qc/meg_trial_pulse_qc.py; see also
# docs/meg.md.
#
# on_mismatch="truncate" would also "fix" a final-trial dropout, by dropping
# that trial. Reconstruction is preferred because it keeps it: the trial's
# start pulse, tGO and behavioral record are all intact, so there is no
# reason to discard real data over a missing trigger edge.
KNOWN_GO_RECONSTRUCTION_RUNS = {
    ('H01', 'Slow', '1'), ('H02', 'RT', '1'), ('H04', 'Fast', '2'), ('H05', 'Fast', '4'), ('H06', 'Fast', '2'), ('H06', 'Slow', '4'),
    ('H07', 'Slow', '2'), ('H07', 'Slow', '4'), ('H08', 'Fast', '2'), ('H08', 'Slow', '1'), ('H10', 'Fast', '1'), ('H10', 'Slow', '3'),
    ('H10', 'Slow', '4'), ('H14', 'Fast', '3'), ('H17', 'Fast', '4'), ('H21', 'Fast', '2'), ('H24', 'Fast', '1'), ('H26', 'Slow', '1'),
    ('H27', 'Fast', '3'), ('H28', 'Slow', '4'), ('H32', 'Fast', '3'),
}

DEFAULT_GO_LAG_S = 0.004  # hardware trigger lag; see reconstruct_missing_go_events


def needs_go_reconstruction(subject: str, condition: Optional[str], run: str) -> bool:
    """Whether this run is a confirmed candidate for reconstruct_missing_go_events."""
    key = (normalize_subject_id(subject), condition, str(run))
    return key in KNOWN_GO_RECONSTRUCTION_RUNS


def reconstruct_missing_go_events(
    events: np.ndarray,
    behavior_df: pd.DataFrame,
    sfreq: float,
    start_code: int,
    go_code: int,
    tolerance_s: float = 0.5,
) -> np.ndarray:
    """Synthesize go-cue events for trials with a trial-start pulse but no
    matching go-cue pulse in `events`.

    `behavior_df` must be the *full*, unfiltered trial sequence in
    nTrialIndex order, including OUTCOME_NEVER_STARTED rows -- trial i
    (0-indexed) is paired positionally with the i-th trial-start pulse, so
    every trial (whether or not it has its own go-cue) must stay in its
    original position to keep that correspondence correct for every trial
    after it. OUTCOME_NEVER_STARTED trials never get a synthetic go-cue
    (they structurally have none by design, confirmed on all 229 real
    occurrences -- see docs/meg.md) but their trial-start
    pulse still counts for positional indexing. This positional pairing is
    valid for the runs this is meant for, where trial-start pulses have
    already been confirmed (see KNOWN_GO_RECONSTRUCTION_RUNS) to align with
    TDMS's trial sequence from the first trial onward, with any
    extra/unlogged pulses only trailing past the last real trial (so they
    fall outside range(n_trials) and are never touched here). Raises if
    there are fewer trial-start pulses than behavior rows, since that means
    this precondition doesn't hold.

    The go-cue lag (go pulse time - (trial-start time + tGO)) is calibrated
    from this run's own trials that do have a real go-cue -- a small, highly
    consistent hardware delay (~2-4ms across every subject checked; see
    docs/meg.md). Falls back to DEFAULT_GO_LAG_S only when a
    run has zero real go-cues to calibrate from (H02RT1).
    """
    starts = np.sort(events[events[:, 2] == start_code][:, 0])
    go_times = events[events[:, 2] == go_code][:, 0] / sfreq

    n_trials = len(behavior_df)
    if len(starts) < n_trials:
        raise ValueError(
            f"Only {len(starts)} trial-start pulses for {n_trials} behavior "
            "rows -- cannot reconstruct go events without a start pulse to "
            "anchor each trial."
        )

    tgo_s = behavior_df['tGO'].to_numpy(dtype=float) / 1000.0
    never_started = (behavior_df['nOutcome'] == OUTCOME_NEVER_STARTED).to_numpy()
    start_times = starts[:n_trials] / sfreq
    expected_go_times = start_times + tgo_s

    matched = np.full(n_trials, np.nan)
    for i in range(n_trials):
        if never_started[i]:
            continue
        window = go_times[
            (go_times > start_times[i]) & (go_times < start_times[i] + tgo_s[i] + tolerance_s)
        ]
        if len(window) > 0:
            matched[i] = window[np.argmin(np.abs(window - expected_go_times[i]))]

    have_go = ~np.isnan(matched)
    calibrated_lags = matched[have_go] - expected_go_times[have_go]
    lag_s = float(calibrated_lags.mean()) if len(calibrated_lags) > 0 else DEFAULT_GO_LAG_S

    missing = np.where(~have_go & ~never_started)[0]
    if len(missing) == 0:
        return events

    synthetic_samples = np.round((expected_go_times[missing] + lag_s) * sfreq).astype(int)
    synthetic = np.column_stack([
        synthetic_samples,
        np.zeros(len(missing), dtype=events.dtype),
        np.full(len(missing), go_code, dtype=events.dtype),
    ])
    augmented = np.vstack([events, synthetic.astype(events.dtype)])
    return augmented[augmented[:, 0].argsort()]


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


def find_behavior_table(
    behavior_root: str,
    subject_id: str,
    run_id: str,
    condition: Optional[str] = None,
) -> Path:
    """Find a Stage 1 behavior TSV derivative for a subject/run."""
    return DerivativeLayout(behavior_root).find_behavior(
        subject=subject_id,
        run=run_id,
        condition=condition,
    )


def synchronize_events_and_behavior(
    events: np.ndarray,
    event_ids: dict,
    behavior_df: pd.DataFrame,
    on_mismatch: str = "error",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Synchronize matching MEG events with behavior rows.

    Trials with nOutcome == OUTCOME_NEVER_STARTED never received a go cue --
    tGO == 0, no tokens shown, nChoiceMade == 0 -- so they structurally have
    no MEG event to align to. They are excluded from the count/alignment
    here (not dropped from the Stage 1 behavior TSV itself), otherwise every
    run containing one would show a spurious trial-count mismatch. Confirmed
    on all 229 occurrences in the real dataset; see docs/meg.md.
    """
    matching_codes = list(event_ids.values())
    matched_events = events[np.isin(events[:, 2], matching_codes)]

    behavior_df = started_trials(behavior_df)

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
        layout = DerivativeLayout(output_dir)
        epochs_path = layout.epochs(
            subject=subject,
            run=run,
            condition=condition,
            alignment=align_to,
            suffix="epo",
            extension=".fif",
        )
        events_tsv_path = layout.epochs(
            subject=subject,
            run=run,
            condition=condition,
            alignment=align_to,
            suffix="events",
            extension=".tsv",
        )
        eve_path = layout.epochs(
            subject=subject,
            run=run,
            condition=condition,
            alignment=align_to,
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
                    "schema_version": 1,
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
