"""Trial-locked epochs from the cleaned recordings, with behavioral metadata.

The go cue is the one event matched to the behavioral log trial by trial, so
it anchors every lock: the others sit at a behavioral offset from it (button
press at ``rawRT``, feedback at ``tTrialEnd - tGO``, decision at ``dt_ms``,
the press minus the subject's hand-specific motor baseline). An epoch spans
its lock's analysis window and no more: what is filtered, band power, is
filtered on the continuous recording (``meg.band_power``), which gives every
trial the filter's full span. An epoch is dropped when its window touches a
``BAD`` annotation or leaves the recording. Then an epoch is dropped as an outlier
when any magnetometer's peak-to-peak over its analysis window is far above
that sensor's level across the cohort: what remains after the marks is
abnormal against the other people, not within its own recording.
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

# An epoch is an outlier when the log peak-to-peak of any magnetometer over its
# analysis window exceeds that sensor's cohort median by this many robust
# standard deviations (1.4826 x the median absolute deviation), per stream and
# lock.
OUTLIER_SD = 5.0
OUTLIER_REASON = "OUTLIER"

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


def epoch(raw: mne.io.BaseRaw, go: np.ndarray, metadata: pd.DataFrame, lock: str) -> mne.Epochs:
    """Epochs over one lock's analysis window, dropped where it touches a mark or leaves the recording.

    The reference sensors are kept: the recordings are at compensation grade 3,
    and the forward model needs them to apply the same compensation.
    """
    offset = lock_offset(metadata, lock)
    defined = np.isfinite(offset)
    onset = go[defined] + offset[defined]
    metadata = metadata.loc[defined].reset_index(drop=True).assign(go_onset_s=go[defined], lock_onset_s=onset)
    samples = raw.first_samp + np.round(onset * raw.info["sfreq"]).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])
    tmin, tmax = WINDOWS[lock]
    return mne.Epochs(
        raw, events, {lock: 1}, tmin=tmin, tmax=tmax, baseline=None, picks=["meg", "ref_meg"],
        reject_by_annotation=True, metadata=metadata, preload=True, verbose=False,
    )


def amplitudes(epochs: mne.BaseEpochs) -> pd.DataFrame:
    """Log10 peak-to-peak of every magnetometer over each epoch's analysis window.

    One row per epoch, one column per magnetometer, named without its
    system suffix so that recordings and people line up.
    """
    window = epochs.copy().pick("mag")
    names = [name.split("-")[0] for name in window.ch_names]
    return pd.DataFrame(np.log10(np.ptp(window.get_data(), axis=2)), columns=names)


def outlier_thresholds(table: pd.DataFrame) -> pd.Series:
    """Per sensor, the cohort median of log peak-to-peak plus OUTLIER_SD robust SDs.

    ``table`` holds one row per epoch of one stream and lock, pooled over the
    cohort, as :func:`amplitudes` returns them.
    """
    median = table.median(axis=0)
    spread = 1.4826 * (table - median).abs().median(axis=0)
    return median + OUTLIER_SD * spread


def drop_outliers(epochs: mne.BaseEpochs, thresholds: pd.Series) -> mne.BaseEpochs:
    """Drop, in place, the epochs whose peak-to-peak exceeds a sensor's threshold."""
    table = amplitudes(epochs)
    over = (table > thresholds[table.columns].to_numpy()).any(axis=1).to_numpy()
    return epochs.drop(np.flatnonzero(over), reason=OUTLIER_REASON, verbose=False)


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
        "outlier_sd": OUTLIER_SD,
        "sfreq": float(epochs.info["sfreq"]),
        "n_events": len(epochs.drop_log),
        "n_epochs": len(epochs),
        "dropped": dict(sorted(reasons.items())),
    }
    with sidecar_path(path).open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=1, sort_keys=True)
    return path


__all__ = [
    "WINDOWS", "OUTLIER_SD", "OUTLIER_REASON", "EPOCH_CONDITIONS", "KNOWN_TRAILING_TRIAL_MISMATCHES", "KNOWN_UNRECOVERABLE_TRIALS",
    "KNOWN_GO_RECONSTRUCTION_RUNS", "read_triggers", "reconstruct_missing_go", "align_go",
    "trial_metadata", "lock_offset", "epoch", "amplitudes", "outlier_thresholds", "drop_outliers", "save_epochs",
]


# Event types of a run for the regression of event responses (meg/band_power.py).
TASK_EVENTS = {
    "go": 1,
    "token right before press": 2,
    "token left before press": 3,
    "token after press fast": 4,
    "token after press slow": 5,
    "press": 6,
    "feedback": 7,
}


def task_events(
    go: np.ndarray, trials: pd.DataFrame, condition: str, sfreq: float, first_samp: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    """Every go cue, token jump, press and feedback of one run, as MNE events.

    ``go`` and ``trials`` are ``align_go``'s pairing; the task log gives every
    token's time (``tTime``) and direction (``nTokenDir``), the press
    (``tEnterTarget``) and the trial's end, its feedback (``tTrialEnd``), in
    milliseconds from the trial's start, which lies ``tGO`` before its go cue.
    Tokens before the press are split by side (1 = right) and those after it
    by the run's speed, since they jump every 200, 20 or 170 ms. The response
    to a jump can change over the trial, so each token before the press also
    carries its place in the trial (1 for the first), a covariate for MNE's
    regression: a response that grows or shrinks with the place. Groups of
    places cannot be separated: the tokens jump on a fixed grid from the go
    cue, so where every trial of a run reaches the 6th token, tokens 4-6 are
    tokens 1-3 shifted by 600 ms. MNE's regression requires one event per
    sample: where a token shares its sample with the go cue, the press or the
    feedback, the token is dropped. In Fast the press sets off the rapid
    stream, its first jump at the press's own time (93 % of trials), so the
    press's response holds that jump. Returns the events, the event types
    present in the run, the covariates and each event's trial (0 for the
    run's first), the last two aligned with the events.
    """
    after = "token after press fast" if condition.lower() == "fast" else "token after press slow"
    rows = []  # (seconds, event type, place of a token before the press or 0, trial)
    for number, (onset, trial) in enumerate(zip(go, trials.itertuples())):
        start = onset - trial.tGO / 1000
        press = trial.tEnterTarget / 1000 if trial.tEnterTarget > 0 else np.inf
        rows.append((onset, "go", 0, number))
        for place, (time, side) in enumerate(zip(np.asarray(trial.tTime, float) / 1000, trial.nTokenDir), start=1):
            if time <= press:
                rows.append((start + time, f"token {'right' if side == 1 else 'left'} before press", place, number))
            else:
                rows.append((start + time, after, 0, number))
        if trial.tEnterTarget > 0:
            rows.append((start + press, "press", 0, number))
        if trial.tTrialEnd > 0:
            rows.append((start + trial.tTrialEnd / 1000, "feedback", 0, number))
    times, names, places, numbers = (np.array(column) for column in zip(*rows))
    samples = first_samp + np.round(times * sfreq).astype(int)
    order = np.lexsort((np.char.startswith(names, "token"), samples))
    keep = order[np.r_[True, np.diff(samples[order]) > 0]]
    codes = np.array([TASK_EVENTS[name] for name in names[keep]])
    events = np.column_stack([samples[keep], np.zeros(keep.size, int), codes])
    present = {name: code for name, code in TASK_EVENTS.items() if code in codes}
    covariates = {
        f"place of {name}": np.where(codes == present[name], places[keep], 0).astype(float)
        for name in ("token right before press", "token left before press") if name in present
    }
    return events, present, covariates, numbers[keep]
