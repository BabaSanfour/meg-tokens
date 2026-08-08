"""Discovery and matching of raw CTF MEG sessions on external media against
raw TDMS behavior runs.

This is Stage 0 -- it reads ``tdms/`` and the raw media directly, via the
same low-level TDMS parser ``meg_tokens.behavior.tdms_bids`` uses
(``parse_tdms_file``/``parse_tdms_filename``).

The raw acquisition media has no condition/run label in its session names
(``H02_DDM-tthiery_20180213_03.ds``); only a subject prefix, an acquisition
date, and a sequential session index. This module works out which numbered
session is which behavioral run (``Slow1``, ``Fast2``, ``RT1``, ...) using
three signals verified against the real dataset, applied in that order:

1. Nominal trial duration recorded in each session's ``.hist`` file (315s
   for Slow/Fast, 135s for RT, anything else excluded) -- unaffected by
   mid-recording truncation, since it reflects the configured protocol, not
   the measured recording length. This is a *pre-filter*: it narrows a
   run's candidates to the sessions recorded under the same protocol, and
   never picks between them.
2. The inter-trial-interval fingerprint (``fingerprint_candidates``) --
   within that duration class, the one candidate whose real trial-start
   pulse rhythm reproduces the run's own logged ``nInitialTime`` rhythm.
   This is the identifying signal: it reads the recording rather than
   reasoning about its position among its neighbours.
3. Chronological order, as a fallback for runs the fingerprint could not
   resolve, and only when the remaining candidate and run counts agree.

Every accepted match is cross-checked against the session's real
trial-start pulse count. Nothing is ever picked on a best-available basis:
a run whose fingerprint fails its acceptance guards, or whose best
candidate is also some other run's best candidate, is reported
``ambiguous`` for human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from meg_tokens.behavior.tdms import (
    FILENAME_RE,
    parse_tdms_file,
    parse_tdms_filename,
)
from meg_tokens.core import RawStagingConfig, normalize_subject_id
from meg_tokens.meg.anat_bids import RAW_T1_RELATIVE_PATH, discover_anat
from meg_tokens.meg.epoching import KNOWN_TRAILING_TRIAL_MISMATCHES

_SESSION_RE = re.compile(
    r"^(?P<prefix>H[0-9]+)_DDM-tthiery_(?P<date>[0-9]{8})_(?P<index>[0-9]+)\.ds$",
    re.IGNORECASE,
)
_HIST_DURATION_RE = re.compile(r"Trial duration:\s*(\S+)")
_HIST_DATE_RE = re.compile(
    r"DATE:\s*\n(?P<date>[0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}\s+[0-9]{1,2}:[0-9]{2})"
)
_NOISE_TEMPLATE = "NOISE_noise_{date}_{index:02d}.ds"

# Dates where the legacy-convention NOISE_noise_<date>_01.ds recording exists
# on media but is corrupt -- CTF failed to write head coil positions, so any
# read of it unconditionally raises `RuntimeError: HPI information not
# available`. An operator retake at a different index is used instead.
# 20181206 (H26/H27, shared empty-room): _01.ds's .hc reads "Unable to make
# head coil file. Reason: no data in this DataSet."; _02.ds has a complete,
# valid .hc and is the recording actually used for source analysis.
KNOWN_NOISE_OVERRIDES = {
    "20181206": 2,
}
_HEADSHAPE_TEMPLATE = "{subject}_DDM-tthiery_{date}_HEADSHAPE.eeg"

# Pinned per-run session mappings for H01 and H05, the two subjects with an
# extra or missing session in a duration class. Maps
# (subject, condition, run) -> the raw session's numeric index.
#
# Not the primary resolution path: the fingerprint reproduces all 17 entries
# from the recordings themselves. match_subject cross-checks every
# fingerprint result against this table and flags disagreement for review,
# and falls back to it only for a run the fingerprint cannot score at all.
# The evidence behind each entry -- and why these two subjects need one --
# is in docs/data_contract.md, "H01 and H05".
KNOWN_SESSION_OVERRIDES = {
    ("H01", "Slow", 1): 5,
    ("H01", "Fast", 1): 6,
    ("H01", "Slow", 2): 7,
    ("H01", "Fast", 2): 8,
    ("H01", "Slow", 3): 9,
    ("H01", "Fast", 3): 10,
    ("H01", "Slow", 4): 11,
    ("H01", "Fast", 4): 12,
    ("H05", "Fast", 1): 3,
    ("H05", "Slow", 1): 4,
    ("H05", "Fast", 2): 5,
    ("H05", "Slow", 2): 6,
    ("H05", "Fast", 3): 7,
    ("H05", "Slow", 3): 8,
    ("H05", "Fast", 4): 9,
    ("H05", "Slow", 4): 10,
    ("H05", "RT", 2): 11,
}


def _duration_class(
    nominal_duration: Optional[float],
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> str:
    """Bucket one raw session as ``"SLOWFAST"``, ``"RT"``, or ``"OTHER"``.

    Duration is used because nothing else on the media side distinguishes
    condition: session filenames carry only a subject prefix, date, and
    sequential index (no condition/run at all -- the reason Stage 0's
    matching problem exists), and the ``.hist`` file's other descriptive
    fields are useless in this real dataset (``Title`` is either the
    generic ``"TEST"`` -- identical across Slow/Fast/RT sessions -- or an
    unrelated label for non-task recordings; ``Procedure step description``
    and ``Keywords`` are always blank). ``Trial duration`` is the one
    populated field that tracks which protocol was configured for that
    recording.

    Real nominal durations across all 412 subject sessions on the media
    take exactly four values -- 135.0, 315.0, 360.0, 420.0 -- with a
    minimum 45s gap between clusters (verified by scanning the real
    drive), so ``settings.duration_tolerance``'s default of 20s has wide
    margin without risking a false match.
    """
    if nominal_duration is None:
        return "OTHER"
    if abs(nominal_duration - settings.slowfast_nominal_duration) <= settings.duration_tolerance:
        return "SLOWFAST"
    if abs(nominal_duration - settings.rt_nominal_duration) <= settings.duration_tolerance:
        return "RT"
    return "OTHER"


@dataclass(frozen=True)
class RawSession:
    """One numbered raw acquisition session discovered on media.

    ``index`` is the session's own numeric filename suffix
    (``..._03.ds`` -> ``3``) -- CTF's own sequential recording counter for
    that subject/date, used only as a tie-break when ``.hist`` timestamps
    (minute-resolution) collide, never as the primary sort key.
    ``nominal_duration``/``duration_class`` come from ``read_hist_metadata``
    -- see ``_duration_class`` for why duration, not the filename, decides
    which experimental protocol a session belongs to.
    """

    subject: str
    path: Path
    date: str  # YYYYMMDD, as it appears on media
    index: int
    nominal_duration: Optional[float]
    start_time: Optional[datetime]
    duration_class: str  # "SLOWFAST" | "RT" | "OTHER"


@dataclass(frozen=True)
class BehaviorRun:
    """One already-ingested behavioral run for a subject.

    ``trial_count`` is every logged trial row, including
    ``OUTCOME_NEVER_STARTED`` ones: those trials still receive their own
    trial-start trigger pulse in the raw MEG recording (their go-cue phase
    never begins, but the trial slot itself does) -- see
    ``reconstruct_missing_go_events`` in ``meg_tokens.meg.epoching`` -- so
    the full row count, not ``started_trials()``, is what should be
    compared against a session's real Start-pulse count.
    """

    subject: str
    condition: str
    run: int
    initial_time_min: float
    trial_count: int
    # Every trial's logged ``nInitialTime`` in ``nTrialIndex`` order. The
    # gaps between these are the run's fingerprint (see
    # ``fingerprint_candidates``); a tuple, not an array, to keep this
    # dataclass frozen/hashable. Empty when unavailable, which simply makes
    # the run unfingerprintable rather than an error.
    initial_times_ms: Tuple[float, ...] = ()

    @property
    def duration_class(self) -> str:
        """The duration class whose sessions this run may be matched against."""
        return "RT" if self.condition.upper() == "RT" else "SLOWFAST"

    @property
    def intervals_ms(self) -> np.ndarray:
        """Inter-trial intervals: this run's side of the fingerprint."""
        return np.diff(np.asarray(self.initial_times_ms, dtype=float))


@dataclass(frozen=True)
class MatchResult:
    """One manifest row: a staging decision for a run, noise, or headshape.

    Field order *is* the manifest's column order --
    ``meg_tokens.workflows.raw_staging`` derives the header from these
    fields rather than repeating them, so adding a column here adds it to
    the written table. ``docs/data_contract.md`` describes what each one
    means.
    """

    subject: str
    kind: str  # "run" | "noise" | "headshape"
    condition: Optional[str]
    run: Optional[int]
    date: str  # acquisition date, YYYYMMDD
    source_path: Optional[Path]
    match_method: str  # "fingerprint" | "known_override" | "fast_path" | "ambiguous" | "found" | "not_found"
    meg_start_pulse_count: Optional[int]
    behavior_trial_count: Optional[int]
    count_agreement: str  # "exact" | "within_tolerance" | "mismatch" | "not_applicable" | "not_checked"
    action: str  # "stage" | "review"
    note: str = ""
    # Fingerprint evidence for this row, when a fingerprint was scored:
    # the winning candidate's mean absolute inter-trial-interval error and
    # how many times worse the runner-up was. Recorded for *every* scored
    # row, including ``fast_path`` ones, so a chronologically-assigned
    # match is auditable on the same evidence a fingerprint match is.
    fingerprint_error_ms: Optional[float] = None
    fingerprint_separation: Optional[float] = None


def read_hist_metadata(ds_dir: Path) -> Optional[dict]:
    """Parse nominal duration and start timestamp from one session's .hist file.

    Returns None if the directory has no readable .hist file (never opens
    any other file in the session).
    """
    hist_files = sorted(ds_dir.glob("*.hist"))
    if not hist_files:
        return None
    text = hist_files[0].read_text(errors="replace")

    duration_match = _HIST_DURATION_RE.search(text)
    nominal_duration = float(duration_match.group(1)) if duration_match else None

    date_match = _HIST_DATE_RE.search(text)
    start_time = None
    if date_match:
        try:
            start_time = datetime.strptime(date_match.group("date"), "%d-%b-%Y %H:%M")
        except ValueError:
            start_time = None

    return {"nominal_duration": nominal_duration, "start_time": start_time}


def discover_raw_sessions(
    raw_root: Path,
    subject: str,
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> List[RawSession]:
    """Chronologically sorted numbered sessions for one subject on media.

    Globs only this subject's own ``<prefix>_DDM-tthiery_*_*.ds`` sessions
    (never noise/other-subject recordings), reads each one's ``.hist`` for
    nominal duration and start time, and classifies it via
    ``_duration_class`` -- see that function for why duration, not the
    filename or any other ``.hist`` field, is the classification signal.
    A session with no readable ``.hist`` file is silently excluded, not
    guessed at.
    """
    subject = normalize_subject_id(subject)
    raw_root = Path(raw_root)

    sessions: List[RawSession] = []
    for ds_dir in sorted(raw_root.glob(f"{subject}_DDM-tthiery_*_*.ds")):
        match = _SESSION_RE.match(ds_dir.name)
        if match is None or match.group("prefix").upper() != subject:
            continue
        metadata = read_hist_metadata(ds_dir)
        if metadata is None:
            continue
        sessions.append(
            RawSession(
                subject=subject,
                path=ds_dir,
                date=match.group("date"),
                index=int(match.group("index")),
                nominal_duration=metadata["nominal_duration"],
                start_time=metadata["start_time"],
                duration_class=_duration_class(metadata["nominal_duration"], settings=settings),
            )
        )

    # Sort by real acquisition timestamp first; the numeric session index
    # (always assigned in acquisition order by CTF) breaks ties, since
    # .hist timestamps are only minute-resolution.
    sessions.sort(key=lambda s: (s.start_time or datetime.min, s.index))
    return sessions


def discover_noise_session(raw_root: Path, subject: str, date: str) -> Optional[Path]:
    """The one legacy-convention empty-room recording for a subject's date, if any.

    Defaults to index 1 (``NOISE_noise_<date>_01.ds``); ``KNOWN_NOISE_OVERRIDES``
    redirects specific dates whose default recording is corrupt on media.
    """
    index = KNOWN_NOISE_OVERRIDES.get(date, 1)
    candidate = Path(raw_root) / _NOISE_TEMPLATE.format(date=date, index=index)
    return candidate if candidate.exists() else None


def discover_headshape(raw_root: Path, subject: str, date: str) -> Optional[Path]:
    """The digitized headshape file for a subject's date, if any."""
    subject = normalize_subject_id(subject)
    candidate = Path(raw_root) / _HEADSHAPE_TEMPLATE.format(subject=subject, date=date)
    return candidate if candidate.exists() else None


def load_behavior_runs(behavior_root: Path, subject: str) -> List[BehaviorRun]:
    """Chronologically sorted behavior runs for a subject, read straight from TDMS.

    Parses every strictly-named ``*.tdms`` run file under the subject's
    input directory directly (``parse_tdms_file`` (
    False)``, the same low-level parser ``meg_tokens.behavior.tdms_bids``
    uses) rather than reading ``behavior ingest``'s derivatives output --
    Stage 0 has no dependency on any other stage having run first. Files
    not matching the project's TDMS filename contract
    (``meg_tokens.behavior.tdms.FILENAME_RE``, e.g. a `temp_*.tdms` scratch
    export) are skipped, not raised on.
    """
    subject = normalize_subject_id(subject)
    subject_dir = Path(behavior_root) / subject

    runs: List[BehaviorRun] = []
    for tdms_path in sorted(subject_dir.glob("*.tdms")):
        if not FILENAME_RE.match(tdms_path.name):
            continue
        run_info = parse_tdms_filename(tdms_path.name)
        df = parse_tdms_file(str(tdms_path))
        ordered = df.sort_values("nTrialIndex")["nInitialTime"].to_numpy(dtype=float)
        runs.append(
            BehaviorRun(
                subject=subject,
                condition=run_info.condition,
                run=int(run_info.run),
                initial_time_min=float(df["nInitialTime"].min()),
                trial_count=int(len(df)),
                initial_times_ms=tuple(ordered),
            )
        )
    runs.sort(key=lambda r: r.initial_time_min)
    return runs


def read_start_pulse_times(ds_path: Path, subject: str) -> np.ndarray:
    """Real trial-start pulse times (ms, relative to the first pulse).

    Opens only the trigger channel via MNE (no full preload). Start pulses
    fire even for trials whose go-cue later dropped out -- and for
    ``OUTCOME_NEVER_STARTED`` trials, verified on every real occurrence by
    ``scripts/qc/meg_trial_pulse_qc.py`` -- so the start channel, not the
    go channel, is the one that tracks the behavioral log row-for-row.

    This is the single low-level read behind both signals this module
    uses: the pulse *count* (an audit check on an accepted match) and the
    pulse *rhythm* (``fingerprint_candidates``, which identifies the match
    in the first place). Callers cache it per path, since a candidate is
    typically scored against several runs.
    """
    import mne

    from meg_tokens.meg.epoching import get_event_id

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_ctf(str(ds_path), preload=False, verbose=False)
    events = mne.find_events(raw, verbose=False)
    start_code = get_event_id("start", subject)["Start"]
    samples = np.sort(events[events[:, 2] == start_code][:, 0])
    if len(samples) == 0:
        return np.empty(0, dtype=float)
    return (samples - samples[0]) / raw.info["sfreq"] * 1000.0


@dataclass(frozen=True)
class FingerprintScore:
    """How well one candidate session's pulse rhythm reproduces a run's."""

    session: "RawSession"
    mean_abs_error_ms: float
    window_offset: int


def _best_window_error(
    candidate_intervals: np.ndarray, run_intervals: np.ndarray
) -> Tuple[float, int]:
    """Best mean-absolute-error alignment of two interval sequences.

    Scored over every relative shift rather than head-to-head, because a
    real recording and its log routinely disagree at *both* ends at once:
    H12Slow2's recording starts three trials late (``KNOWN_UNRECOVERABLE_TRIALS``)
    *and* carries one extra trailing pulse, so neither sequence is contained
    in the other and no single sliding window over the longer one can
    express the alignment. Each shift is scored on the overlap it produces,
    which must cover at least half the shorter sequence -- otherwise a
    two-interval coincidence at the very edge could outscore the true
    full-length match.

    Returns the error in ms and the winning shift: the index in
    ``run_intervals`` that ``candidate_intervals[0]`` aligns to. Positive
    means the recording missed that many leading trials; negative means it
    has that many extra leading pulses.
    """
    n_candidate, n_run = len(candidate_intervals), len(run_intervals)
    if n_candidate == 0 or n_run == 0:
        return float("inf"), 0

    min_overlap = max(3, min(n_candidate, n_run) // 2)
    best_error, best_shift = float("inf"), 0
    for shift in range(-(n_candidate - min_overlap), n_run - min_overlap + 1):
        candidate_start = max(0, -shift)
        run_start = max(0, shift)
        overlap = min(n_candidate - candidate_start, n_run - run_start)
        if overlap < min_overlap:
            continue
        error = float(np.mean(np.abs(
            candidate_intervals[candidate_start:candidate_start + overlap]
            - run_intervals[run_start:run_start + overlap]
        )))
        if error < best_error:
            best_error, best_shift = error, shift
    return best_error, best_shift


def fingerprint_candidates(
    run: BehaviorRun,
    candidates: Sequence[RawSession],
    pulse_times: Dict[Path, np.ndarray],
) -> List[FingerprintScore]:
    """Score every candidate session against one run's inter-trial rhythm, best first.

    A behavioral run logs each trial's ``nInitialTime``; the raw recording
    emits a trial-start pulse per trial. Neither clock is shared, but the
    *gaps* between consecutive trials are the same physical intervals
    measured twice, so the correct pairing reproduces them to well under a
    millisecond while any other candidate -- a different run of the same
    protocol, with its own trial-by-trial timing -- is off by hundreds.
    That gap is what makes this identifying rather than merely suggestive,
    and it is why the decision needs no tie-break heuristic: see
    ``RawStagingConfig.fingerprint_max_error_ms`` /
    ``fingerprint_min_separation`` for the acceptance guards applied to
    these scores.

    ``pulse_times`` maps a candidate's path to its
    ``read_start_pulse_times`` array; the caller owns it so a session read
    once can be scored against every run in its duration class.
    """
    run_intervals = run.intervals_ms
    scores = []
    for candidate in candidates:
        error, offset = _best_window_error(
            np.diff(pulse_times[candidate.path]), run_intervals
        )
        scores.append(
            FingerprintScore(session=candidate, mean_abs_error_ms=error, window_offset=offset)
        )
    scores.sort(key=lambda score: score.mean_abs_error_ms)
    return scores


def _separation(scores: Sequence[FingerprintScore]) -> float:
    """How many times worse the runner-up is than the winner.

    ``inf`` when there is only one candidate (nothing to be confused with)
    or when the winner is a perfect match, which is the honest answer in
    both cases: the acceptance guard's job is to reject a *close* second,
    and there isn't one.
    """
    if len(scores) < 2:
        return float("inf")
    if scores[0].mean_abs_error_ms <= 0:
        return float("inf")
    return scores[1].mean_abs_error_ms / scores[0].mean_abs_error_ms


def resolve_by_fingerprint(
    runs: Sequence[BehaviorRun],
    candidates: Sequence[RawSession],
    pulse_times: Dict[Path, np.ndarray],
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> Dict[Tuple[str, int], Tuple[RawSession, float, float]]:
    """Fingerprint-resolve as many runs as the evidence unambiguously supports.

    Returns ``{(condition, run): (session, mean_abs_error_ms, separation)}``,
    containing only runs that cleared all three tests:

    * the winning error is within ``settings.fingerprint_max_error_ms``;
    * the runner-up is at least ``settings.fingerprint_min_separation``
      times worse;
    * no other run's winner is the same session.

    That last test is what keeps a duplicate from being resolved by
    preference: if two runs both fingerprint best to one recording, the
    evidence is self-contradictory and *both* are left for the caller to
    report as ambiguous, rather than awarding it to the better-scoring one.
    """
    winners: Dict[Tuple[str, int], Tuple[RawSession, float, float]] = {}
    for run in runs:
        if len(run.intervals_ms) == 0 or not candidates:
            continue
        scores = fingerprint_candidates(run, candidates, pulse_times)
        best = scores[0]
        separation = _separation(scores)
        if best.mean_abs_error_ms > settings.fingerprint_max_error_ms:
            continue
        if separation < settings.fingerprint_min_separation:
            continue
        winners[(run.condition, run.run)] = (best.session, best.mean_abs_error_ms, separation)

    claimed: Dict[Path, int] = {}
    for session, _, _ in winners.values():
        claimed[session.path] = claimed.get(session.path, 0) + 1
    return {
        key: value for key, value in winners.items() if claimed[value[0].path] == 1
    }


def _count_agreement(meg_count: Optional[int], behavior_count: Optional[int], *, tolerance: int) -> str:
    """Compare a matched session's real Start-pulse count to its behavior run's trial count.

    ``"not_checked"`` when either count is unavailable -- e.g. an
    ``ambiguous`` run has no candidate session picked yet, so there is
    nothing to count pulses on. Otherwise one of ``"exact"``,
    ``"within_tolerance"`` (see ``RawStagingConfig.count_tolerance``), or
    ``"mismatch"``.
    """
    if meg_count is None or behavior_count is None:
        return "not_checked"
    diff = abs(meg_count - behavior_count)
    if diff == 0:
        return "exact"
    if diff <= tolerance:
        return "within_tolerance"
    return "mismatch"


@dataclass(frozen=True)
class _MatchContext:
    """What every result builder needs besides the run being matched.

    ``pulse_times`` is the read cache: each candidate's trial-start pulses
    are read once for fingerprinting and reused for the pulse count of
    whichever run claims it.
    """

    date: str
    settings: RawStagingConfig
    pulse_times: Dict[Path, np.ndarray]


def _build_result(
    run: BehaviorRun,
    candidate: Optional[RawSession],
    context: _MatchContext,
    *,
    match_method: str,
    fingerprint: Optional[Tuple[float, float]] = None,
    note: str = "",
) -> MatchResult:
    """Turn one behavior run + its candidate (or ``None``) into a manifest row.

    Always records the candidate's real Start-pulse count when there is
    one -- even for an unambiguous match -- so the manifest is auditable
    for every staged run, not just the disambiguated ones. A count mismatch
    downgrades ``action`` to ``"review"``, except for the runs listed in
    ``KNOWN_TRAILING_TRIAL_MISMATCHES``, where a differing count is the
    expected, independently verified boundary artifact rather than evidence
    that the match itself is wrong.
    """
    meg_count = (
        int(len(context.pulse_times[candidate.path])) if candidate is not None else None
    )
    agreement = _count_agreement(
        meg_count, run.trial_count, tolerance=context.settings.count_tolerance
    )
    action = "stage" if agreement in ("exact", "within_tolerance") else "review"
    if agreement == "mismatch" and (
        run.subject, run.condition, str(run.run)
    ) in KNOWN_TRAILING_TRIAL_MISMATCHES:
        action = "stage"
        note = " ".join(filter(None, [
            note,
            "Start-pulse count differs from a documented "
            "KNOWN_TRAILING_TRIAL_MISMATCHES case (see docs/behavior_qc_report.md "
            "section 3) -- independently verified as boundary-only, staging anyway.",
        ]))
    return MatchResult(
        subject=run.subject,
        kind="run",
        condition=run.condition,
        run=run.run,
        date=context.date,
        source_path=candidate.path if candidate else None,
        match_method=match_method,
        meg_start_pulse_count=meg_count,
        behavior_trial_count=run.trial_count,
        count_agreement=agreement,
        action=action,
        note=note,
        fingerprint_error_ms=fingerprint[0] if fingerprint else None,
        fingerprint_separation=fingerprint[1] if fingerprint else None,
    )


def _fingerprint_result(
    run: BehaviorRun,
    session: RawSession,
    error_ms: float,
    separation: float,
    override_index: Optional[int],
    context: _MatchContext,
) -> MatchResult:
    """A fingerprint-resolved row, cross-checked against any pinned override.

    A fingerprint that contradicts ``KNOWN_SESSION_OVERRIDES`` reports the
    fingerprint's answer but is forced to ``review``: one of the two is
    wrong, and neither gets to win silently.
    """
    note = f"Identified by inter-trial-interval fingerprint: {error_ms:.3f}ms mean error"
    note += (
        "; only one candidate in this duration class."
        if separation == float("inf")
        else f", {separation:.0f}x better than the next candidate."
    )
    contradicts_override = override_index is not None and override_index != session.index
    if contradicts_override:
        note += (
            f" DISAGREES with KNOWN_SESSION_OVERRIDES, which maps this run to session "
            f"{override_index:02d}. One of the two is wrong; staging should not proceed "
            "until it is resolved."
        )
    result = _build_result(
        run, session, context,
        match_method="fingerprint", fingerprint=(error_ms, separation), note=note,
    )
    return replace(result, action="review") if contradicts_override else result


def _chronological_results(
    runs: Sequence[BehaviorRun], candidates: Sequence[RawSession], context: _MatchContext
) -> List[MatchResult]:
    """Pair equal-length run and candidate lists 1:1 in the order both recorded.

    Each row still carries its fingerprint score, so an order-assigned match
    is auditable on the same evidence a fingerprinted one is.
    """
    results = []
    for candidate, run in zip(candidates, runs):
        scores = fingerprint_candidates(run, candidates, context.pulse_times)
        assigned = next((s for s in scores if s.session.path == candidate.path), None)
        scored = assigned is not None and np.isfinite(assigned.mean_abs_error_ms)
        results.append(
            _build_result(
                run, candidate, context, match_method="fast_path",
                fingerprint=(assigned.mean_abs_error_ms, _separation(scores)) if scored else None,
            )
        )
    return results


def _ambiguous_results(
    runs: Sequence[BehaviorRun],
    candidates: Sequence[RawSession],
    context: _MatchContext,
    *,
    run_class: str,
) -> List[MatchResult]:
    """Rows for runs nothing could resolve, carrying the evidence a human needs.

    Each note lists every remaining candidate with its real pulse count, plus
    this run's best fingerprint score and why it was not good enough -- so
    resolving the row by hand means reading the evidence, not re-deriving it.
    """
    candidates_desc = "; ".join(
        f"{candidate.path.name} (start pulses={len(context.pulse_times[candidate.path])})"
        for candidate in candidates
    ) or "none"

    results = []
    for run in runs:
        scores = fingerprint_candidates(run, candidates, context.pulse_times)
        best_desc = ""
        if scores and np.isfinite(scores[0].mean_abs_error_ms):
            best_desc = (
                f" Best fingerprint: {scores[0].session.path.name} at "
                f"{scores[0].mean_abs_error_ms:.3f}ms ({_separation(scores):.1f}x separation)"
                " -- below the acceptance threshold, or contested by another run."
            )
        results.append(
            _build_result(
                run, None, context, match_method="ambiguous",
                note=(
                    f"{len(candidates)} candidate session(s) for {len(runs)} remaining "
                    f"{run_class} run(s) -- not resolved automatically. This run needs "
                    f"{run.trial_count} trials. Candidates: {candidates_desc}.{best_desc}"
                ),
            )
        )
    return results


def _match_duration_class(
    runs: Sequence[BehaviorRun],
    candidates: Sequence[RawSession],
    context: _MatchContext,
    *,
    run_class: str,
) -> List[MatchResult]:
    """Resolve one duration class: fingerprint, then override, then order."""
    unclaimed = {candidate.index: candidate for candidate in candidates}
    fingerprinted = resolve_by_fingerprint(
        runs, candidates, context.pulse_times, settings=context.settings
    )

    results: List[MatchResult] = []
    unresolved: List[BehaviorRun] = []
    for run in runs:
        override_index = KNOWN_SESSION_OVERRIDES.get((run.subject, run.condition, run.run))
        winner = fingerprinted.get((run.condition, run.run))
        if winner is not None:
            session, error_ms, separation = winner
            del unclaimed[session.index]
            results.append(
                _fingerprint_result(run, session, error_ms, separation, override_index, context)
            )
            continue

        candidate = unclaimed.pop(override_index, None) if override_index is not None else None
        if candidate is not None:
            results.append(
                _build_result(
                    run, candidate, context, match_method="known_override",
                    note=(
                        "Fingerprint did not resolve this run; falling back to the "
                        "documented mapping in docs/data_contract.md "
                        "\"H01 and H05\" (KNOWN_SESSION_OVERRIDES)."
                    ),
                )
            )
        else:
            unresolved.append(run)

    if not unresolved:
        return results

    remaining = sorted(
        unclaimed.values(), key=lambda c: (c.start_time or datetime.min, c.index)
    )
    if len(remaining) == len(unresolved):
        results.extend(_chronological_results(unresolved, remaining, context))
    else:
        results.extend(_ambiguous_results(unresolved, remaining, context, run_class=run_class))
    return results


def match_subject(
    raw_sessions: Sequence[RawSession],
    behavior_runs: Sequence[BehaviorRun],
    *,
    date: str,
    settings: RawStagingConfig = RawStagingConfig(),
) -> List[MatchResult]:
    """Match one subject's raw sessions to their behavior runs.

    Both inputs must already be chronologically sorted (as returned by
    ``discover_raw_sessions``/``load_behavior_runs``). Returns one
    ``MatchResult`` per behavior run.

    Duration class (see ``_duration_class``) partitions the problem first:
    a run is only ever matched against sessions recorded under its own
    protocol, so RT runs and Slow/Fast runs never compete for the same
    recording. Within a class, pairing is attempted in three ways, none of
    them a guess:

    1. ``resolve_by_fingerprint`` -- the candidate whose real trial-start
       pulse rhythm reproduces this run's logged inter-trial rhythm, subject
       to the acceptance guards in ``RawStagingConfig``. This identifies the
       recording from its own content and is tried first, because it
       depends on nothing but the two clocks agreeing.
    2. ``KNOWN_SESSION_OVERRIDES`` -- documented per-run mappings for H01
       and H05, retained as a *fallback and cross-check*, not a primary
       input: when the fingerprint resolves a run that also has an override,
       the two are compared and any disagreement is flagged for review
       rather than silently preferring either.
    3. Chronological order over whatever is still unresolved, and only when
       the remaining candidate and run counts are equal -- both lists are
       independently sorted by real timestamps (the session's ``.hist``
       ``DATE:`` and the run's ``nInitialTime``), so this applies an order
       both systems already recorded rather than inferring one. These rows
       still carry their fingerprint score, so a chronological assignment is
       auditable on the same evidence as a fingerprinted one.

    Whenever none applies, nothing is picked -- there is no best-available
    fallback. Every such run is reported ``match_method="ambiguous"`` with
    each remaining candidate's Start-pulse count and fingerprint error in
    its ``note``, so a human has the evidence to fill in the correct
    ``source_path`` by hand (see ``docs/data_contract.md``).
    """
    results: List[MatchResult] = []
    context = _MatchContext(date=date, settings=settings, pulse_times={})

    for run_class in ("RT", "SLOWFAST"):
        runs = [run for run in behavior_runs if run.duration_class == run_class]
        if not runs:
            continue
        candidates = [s for s in raw_sessions if s.duration_class == run_class]
        for candidate in candidates:
            context.pulse_times.setdefault(
                candidate.path, read_start_pulse_times(candidate.path, candidate.subject)
            )
        results.extend(_match_duration_class(runs, candidates, context, run_class=run_class))

    results.sort(key=lambda r: (r.condition or "", r.run or 0))
    return results


def _asset_result(
    subject: str, kind: str, path: Optional[Path], *, date: str, missing_note: str
) -> MatchResult:
    """One manifest row for a per-subject file that is found or isn't.

    Unlike a run, these need no matching -- there is at most one empty-room
    recording, head shape, and anatomical per subject, so discovery *is* the
    decision. Absence is reported as ``not_found``/``review`` rather than
    raised, because a genuine dataset gap (H07/H10 have no FreeSurfer
    reconstruction) is something to see in the manifest alongside every
    other gap, not an error that stops the plan.
    """
    return MatchResult(
        subject=subject,
        kind=kind,
        condition=None,
        run=None,
        date=date,
        source_path=path,
        match_method="found" if path else "not_found",
        meg_start_pulse_count=None,
        behavior_trial_count=None,
        count_agreement="not_applicable",
        action="stage" if path else "review",
        note="" if path else missing_note,
    )


def match_subject_assets(
    raw_root: Path,
    subject: str,
    date: str,
    *,
    subjects_dir: Optional[Path] = None,
) -> List[MatchResult]:
    """Manifest rows for a subject's empty-room, head shape, and anatomical.

    The anatomical row carries no ``date``: the MRI is a separate
    acquisition from the MEG session, so stamping it with the MEG date
    would assert something untrue.
    """
    subject = normalize_subject_id(subject)
    return [
        _asset_result(
            subject, "noise", discover_noise_session(raw_root, subject, date),
            date=date,
            missing_note=(
                "No "
                + _NOISE_TEMPLATE.format(date=date, index=KNOWN_NOISE_OVERRIDES.get(date, 1))
                + " found."
            ),
        ),
        _asset_result(
            subject, "headshape", discover_headshape(raw_root, subject, date),
            date=date, missing_note="No _HEADSHAPE.eeg found.",
        ),
        _asset_result(
            subject, "anat", discover_anat(subjects_dir, subject),
            date="",
            missing_note=(
                f"No FreeSurfer {RAW_T1_RELATIVE_PATH} under subjects_dir"
                + (f" ({subjects_dir})." if subjects_dir else "; subjects_dir is not configured.")
            ),
        ),
    ]


def match_raw_to_behavior(
    raw_root: Path,
    behavior_root: Path,
    subject: str,
    *,
    subjects_dir: Optional[Path] = None,
    settings: RawStagingConfig = RawStagingConfig(),
) -> List[MatchResult]:
    """Every staging decision for one subject: runs, noise, headshape, anat.

    ``behavior_root`` is the raw TDMS root (``ProjectConfig.behavior_root``,
    i.e. ``data_root/tdms``) -- Stage 0 reads TDMS directly and does not
    require ``behavior ingest`` to have run first. ``subjects_dir`` is the
    FreeSurfer reconstruction root; when it is ``None`` the anatomical row
    is still emitted, as ``not_found``.

    Covering all four kinds here is what makes the manifest the whole
    Stage 0 plan: one review pass sees every file that will be staged for a
    subject, and every gap, instead of the anatomical layer being staged by
    a separate command with its own separate reporting.
    """
    subject = normalize_subject_id(subject)
    raw_sessions = discover_raw_sessions(raw_root, subject, settings=settings)
    behavior_runs = load_behavior_runs(behavior_root, subject)
    if not raw_sessions:
        raise FileNotFoundError(
            f"No raw sessions found for {subject} under {raw_root}"
        )
    if not behavior_runs:
        raise FileNotFoundError(
            f"No TDMS runs found for {subject} under {behavior_root}"
        )

    date = raw_sessions[0].date
    results = match_subject(raw_sessions, behavior_runs, date=date, settings=settings)
    results.extend(match_subject_assets(raw_root, subject, date, subjects_dir=subjects_dir))
    return results
