"""Trial-locked epochs from the cleaned recordings, with behavioral metadata.

The go cue is the one event matched to the behavioral log trial by trial, so
it anchors every lock: the others sit at a behavioral offset from it (button
press at ``rawRT``, feedback at ``tTrialEnd - tGO``, decision at ``dt_ms``,
the press minus the subject's hand-specific motor baseline). Epochs are cut
from the continuous recording with a margin on both sides of the analysis
window, so band-pass and Hilbert edges fall outside it; an epoch is dropped
when its analysis window, not its margin, touches a ``BAD`` annotation or
leaves the recording. Near the ends of a recording the margin is shorter, and
zero beyond the last recorded sample.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from mne_bids.read import events_file_to_annotation_kwargs

from meg_tokens.behavior.schema import OUTCOME_NEVER_STARTED
from meg_tokens.behavior.trials import started_trials
from meg_tokens.core import normalize_subject_id
from meg_tokens.io import sidecar_path


# Analysis windows in seconds, the paper's: -500/+1000 ms around the first
# token jump, -1500/+2000 ms around press, feedback and decision.
WINDOWS = {
    "go": (-0.5, 1.0),
    "enter": (-1.5, 2.0),
    "feedback": (-1.5, 2.0),
    "decision": (-1.5, 2.0),
}
PADDING = 1.0

# The RT blocks measure the motor baseline that decision time subtracts; their
# trials are not analysed, so they are not epoched.
EPOCH_CONDITIONS = ("Fast", "Slow")

# Stage 1 timing columns carried next to the trial features.
TIMING_COLUMNS = ["tGO", "tEnterTarget", "tTrialEnd"]


# The three lookups below normalize the subject, unlike the rest of the
# pipeline, because a miss here is silent: an unmatched key means a known-bad
# run is quietly treated as ordinary rather than raising.

# Runs whose MEG and behavioral trial sequences agree throughout and differ
# only at the trailing boundary: extra MEG trial-start pulses with no logged
# trial (H03RT1, +1) or logged trials MEG never captured (H06Slow2, -12).
# Each was verified two ways: one alignment offset separating from the next
# best by 100-1000x at sub-ms error (``raw_staging.fingerprint_candidates``),
# and trial-start to go-cue latency matching tGO to 1.6-4.25 ms. Truncating
# the longer side is safe because the discrepancy sits only at the end.
# H12Slow2 has irregularities at both ends and is handled by
# KNOWN_UNRECOVERABLE_TRIALS instead. Evidence in docs/meg.md.
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

# Real trials with no MEG event at all, in the middle of an otherwise
# recoverable run. H12Slow2 trial 2: the recording started about three trials
# after the behavioral log; trials 1 and 3 are never-started, trial 2 is real.
# Without it the remaining 57 trials match MEG's 57 go events exactly.
KNOWN_UNRECOVERABLE_TRIALS = {
    ('H12', 'Slow', '2'): {2},
}

# Runs that lose their final trial's go cue: the recording stopped between
# that trial's start pulse and its go cue. Trial starts align with the
# behavioral log at sub-ms precision across these runs, and real go cues follow
# start + tGO by 3.20 +/- 1.19 ms over 2230 trials; each run's own mean lag
# places the missing one. Reproduce with scripts/qc/meg_trial_pulse_qc.py.
KNOWN_GO_RECONSTRUCTION_RUNS = {
    ('H01', 'Slow', '1'), ('H04', 'Fast', '2'), ('H05', 'Fast', '4'), ('H06', 'Fast', '2'), ('H06', 'Slow', '4'),
    ('H07', 'Slow', '2'), ('H07', 'Slow', '4'), ('H08', 'Fast', '2'), ('H08', 'Slow', '1'), ('H10', 'Fast', '1'), ('H10', 'Slow', '3'),
    ('H10', 'Slow', '4'), ('H14', 'Fast', '3'), ('H17', 'Fast', '4'), ('H21', 'Fast', '2'), ('H24', 'Fast', '1'), ('H26', 'Slow', '1'),
    ('H27', 'Fast', '3'), ('H28', 'Slow', '4'), ('H32', 'Fast', '3'),
}


def _key(subject: str, condition: str, run) -> tuple[str, str, str]:
    return normalize_subject_id(subject), condition, str(run)


def read_triggers(bids_root: str | Path, subject: str, condition: str, run, *, task: str = "tokens") -> dict[str, np.ndarray]:
    """Trigger onsets in seconds from the first sample, by name, from the raw BIDS ``events.tsv``.

    Staging writes that file from the trigger channel with each subject's own
    codes (H06 swaps Start and Go), so the names here are already resolved.
    """
    path = BIDSPath(
        root=Path(bids_root), subject=subject, task=task, acquisition=condition.lower(),
        run=str(run), datatype="meg", suffix="events", extension=".tsv",
    )
    events = events_file_to_annotation_kwargs(path.fpath, verbose=False)
    onsets, names = events["onset"], np.asarray(events["description"])
    return {name: np.sort(onsets[names == name]) for name in np.unique(names)}


def reconstruct_missing_go(
    starts: np.ndarray, gos: np.ndarray, behavior: pd.DataFrame, tolerance_s: float = 0.5,
) -> np.ndarray:
    """Place a go cue at ``start + tGO + lag`` for every started trial without one.

    ``behavior`` is the full trial sequence in nTrialIndex order, never-started
    rows included: trial i is paired with the i-th trial-start pulse, so every
    row has to keep its position. Never-started trials get no go cue. The lag is
    the mean over this run's real go cues.
    """
    n_trials = len(behavior)
    if len(starts) < n_trials:
        raise ValueError(
            f"Only {len(starts)} trial-start pulses for {n_trials} behavior rows: "
            "no start pulse to anchor each trial."
        )
    tgo = behavior["tGO"].to_numpy(dtype=float) / 1000.0
    never_started = (behavior["nOutcome"] == OUTCOME_NEVER_STARTED).to_numpy()
    expected = starts[:n_trials] + tgo

    matched = np.full(n_trials, np.nan)
    for i in np.flatnonzero(~never_started):
        window = gos[(gos > starts[i]) & (gos < expected[i] + tolerance_s)]
        if len(window):
            matched[i] = window[np.argmin(np.abs(window - expected[i]))]
    have = ~np.isnan(matched)
    if not have.any():
        raise ValueError("no real go cue to calibrate the lag from")
    lag = float(np.mean(matched[have] - expected[have]))
    missing = ~have & ~never_started
    return np.sort(np.concatenate([gos, expected[missing] + lag]))


def align_go(
    triggers: dict[str, np.ndarray], behavior: pd.DataFrame, subject: str, condition: str, run,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Pair go-cue onsets with the started trials of one run's behavioral table.

    Returns the onsets, the matching trials and what the known-issue tables
    changed. A count mismatch outside KNOWN_TRAILING_TRIAL_MISMATCHES raises.
    """
    key = _key(subject, condition, run)
    gos = triggers.get("Go", np.array([]))
    n_real = len(gos)
    if key in KNOWN_GO_RECONSTRUCTION_RUNS:
        gos = reconstruct_missing_go(triggers["Start"], gos, behavior)
    n_reconstructed = len(gos) - n_real
    unrecoverable = KNOWN_UNRECOVERABLE_TRIALS.get(key, set())
    trials = started_trials(behavior.loc[~behavior["nTrialIndex"].isin(unrecoverable)])
    trials = trials.reset_index(drop=True)

    n_go, n_trials = len(gos), len(trials)
    if n_go == 0:
        raise ValueError(f"{subject} {condition}{run}: no go cue")
    if n_go != n_trials:
        if key not in KNOWN_TRAILING_TRIAL_MISMATCHES:
            raise ValueError(
                f"{subject} {condition}{run}: {n_go} go cues for {n_trials} started trials"
            )
        n = min(n_go, n_trials)
        gos, trials = gos[:n], trials.iloc[:n]
    return gos, trials, {
        "n_go_reconstructed": n_reconstructed,
        "n_trials_unrecoverable": len(unrecoverable),
        "n_truncated": abs(n_go - n_trials),
    }


def trial_metadata(trials: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """The trial features of these trials, in their order, with the Stage 1 timings."""
    merged = trials[["nTrialIndex", *TIMING_COLUMNS]].merge(
        features, on="nTrialIndex", how="left", validate="one_to_one", indicator=True,
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", "nTrialIndex"].tolist()
        raise ValueError(f"trials {missing} have no trial features")
    return merged.drop(columns="_merge")


def lock_offset(metadata: pd.DataFrame, lock: str) -> np.ndarray:
    """Seconds from the go cue to the lock event; NaN where the trial has none."""
    if lock == "go":
        offset_ms = np.zeros(len(metadata))
    elif lock == "enter":
        offset_ms = metadata["rawRT"].to_numpy(dtype=float)
    elif lock == "feedback":
        offset_ms = (metadata["tTrialEnd"] - metadata["tGO"]).to_numpy(dtype=float)
    elif lock == "decision":
        offset_ms = metadata["dt_ms"].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown lock {lock!r}; expected one of {list(WINDOWS)}")
    return offset_ms / 1000.0


def epoch(raw: mne.io.BaseRaw, go: np.ndarray, metadata: pd.DataFrame, lock: str) -> mne.EpochsArray:
    """Padded epochs around one lock, dropped where the analysis window touches a mark.

    An epoch is dropped when its analysis window touches a mark or leaves the
    recording. Its margin is kept as far as the recording reaches and zero
    beyond; ``pad_before_s`` and ``pad_after_s`` give the recorded part, and
    only that part may be filtered. The reference sensors are kept: the
    recordings are at compensation grade 3, and the forward model needs them to
    apply the same compensation.
    """
    offset = lock_offset(metadata, lock)
    defined = np.isfinite(offset)
    onset = go[defined] + offset[defined]
    metadata = metadata.loc[defined].reset_index(drop=True).assign(go_onset_s=go[defined], lock_onset_s=onset)
    sfreq = raw.info["sfreq"]
    samples = raw.first_samp + np.round(onset * sfreq).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])
    tmin, tmax = WINDOWS[lock]
    kept = mne.Epochs(
        raw, events, {lock: 1}, tmin=tmin, tmax=tmax, baseline=None, picks=["meg", "ref_meg"],
        reject_by_annotation=True, metadata=metadata, preload=False, verbose=False,
    ).drop_bad(verbose=False)

    # MNE's sample grid for the padded window, as mne.Epochs would cut it.
    start = int(round((tmin - PADDING) * sfreq))
    n_times = int(round((tmax + PADDING) * sfreq)) + 1 - start
    data = np.zeros((len(kept), len(kept.ch_names), n_times))
    pad = np.empty((len(kept), 2))
    for i, first in enumerate(kept.events[:, 0] - raw.first_samp + start):
        lo, hi = max(first, 0), min(first + n_times, raw.n_times)
        data[i, :, lo - first:hi - first] = raw.get_data(picks=kept.ch_names, start=lo, stop=hi)
        pad[i] = PADDING - (lo - first) / sfreq, PADDING - (first + n_times - hi) / sfreq

    epochs = mne.EpochsArray(
        data, kept.info, kept.events, tmin=start / sfreq, event_id=kept.event_id, baseline=None,
        metadata=kept.metadata.assign(pad_before_s=pad[:, 0], pad_after_s=pad[:, 1]),
        selection=kept.selection, drop_log=kept.drop_log, raw_sfreq=sfreq, on_missing="ignore", verbose=False,
    )
    return epochs.set_annotations(kept.annotations)


def save_epochs(epochs: mne.Epochs, path: Path, record: dict) -> Path:
    """Write the epochs and a sidecar with the window and every dropped epoch's reason."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=True, verbose=False)
    lock = next(iter(epochs.event_id))
    reasons = Counter(reason for log in epochs.drop_log for reason in log)
    sidecar = {
        **record,
        "lock": lock,
        "analysis_window_s": list(WINDOWS[lock]),
        "padding_s": PADDING,
        "sfreq": float(epochs.info["sfreq"]),
        "n_events": len(epochs.drop_log),
        "n_epochs": len(epochs),
        "n_short_padding": int(((epochs.metadata["pad_before_s"] < PADDING) | (epochs.metadata["pad_after_s"] < PADDING)).sum()),
        "dropped": dict(sorted(reasons.items())),
    }
    with sidecar_path(path).open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=1, sort_keys=True)
    return path


__all__ = [
    "WINDOWS", "PADDING", "EPOCH_CONDITIONS", "KNOWN_TRAILING_TRIAL_MISMATCHES", "KNOWN_UNRECOVERABLE_TRIALS",
    "KNOWN_GO_RECONSTRUCTION_RUNS", "read_triggers", "reconstruct_missing_go", "align_go",
    "trial_metadata", "lock_offset", "epoch", "save_epochs",
]
