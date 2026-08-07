"""Discovery and matching of raw CTF MEG sessions on external media against
raw TDMS behavior runs.

This is Stage 0 -- it reads only ``tdms/`` and the raw media directly, via
the same low-level TDMS parser ``meg_tokens.behavior.tdms_bids`` uses
(``parse_tdms_file``/``parse_tdms_filename``), not
``derivatives/meg-tokens/sub-*/beh/`` (the ``behavior ingest`` output). It
has no dependency on any other stage having run first.

The raw acquisition media has no condition/run label in its session names
(``H02_DDM-tthiery_20180213_03.ds``); only a subject prefix, an acquisition
date, and a sequential session index. This module works out which numbered
session is which behavioral run (``Slow1``, ``Fast2``, ``RT1``, ...) using
two signals verified against the real dataset (see
``docs/meg_t0_7_raw_bidsification_plan.md``):

1. Nominal trial duration recorded in each session's ``.hist`` file (315s
   for Slow/Fast, 135s for RT, anything else excluded) -- unaffected by
   mid-recording truncation, since it reflects the configured protocol, not
   the measured recording length.
2. Chronological order, cross-checked against the real MEG trial-start
   pulse count whenever the duration bucket alone is ambiguous (i.e. a
   subject has more or fewer same-class sessions than expected runs).

This module never writes anything and never opens the large ``.meg4`` data
files except to count trigger events for validation; it only reads ``.hist``
metadata and (for validation) the trigger channel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from meg_tokens.behavior.tdms import (
    FILENAME_RE,
    find_subject_directory,
    parse_tdms_file,
    parse_tdms_filename,
)
from meg_tokens.core import RawStagingConfig, normalize_subject_id
from meg_tokens.meg.epoching import KNOWN_TRAILING_TRIAL_MISMATCHES

_SESSION_RE = re.compile(
    r"^(?P<prefix>H[0-9]+|Pilot[0-9]+)_DDM-tthiery_(?P<date>[0-9]{8})_(?P<index>[0-9]+)\.ds$",
    re.IGNORECASE,
)
_HIST_DURATION_RE = re.compile(r"Trial duration:\s*(\S+)")
_HIST_DATE_RE = re.compile(
    r"DATE:\s*\n(?P<date>[0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}\s+[0-9]{1,2}:[0-9]{2})"
)
_NOISE_TEMPLATE = "NOISE_noise_{date}_01.ds"
_HEADSHAPE_TEMPLATE = "{prefix}_DDM-tthiery_{date}_HEADSHAPE.eeg"

# Verified, documented resolutions for the two subjects where duration
# class + chronological order + candidate/run count alone cannot
# automatically resolve a match (an extra or missing raw session in that
# duration class). Maps (subject, condition, run) -> the raw session's
# numeric index. See docs/meg_t0_7_raw_bidsification_plan.md
# "H01/H05 resolution" for the full evidence trail; summary:
#
# H01 (Pilot01): the original researcher's own conversion notebook
# (archive/replicated/DDM_scripts/scripts_new/55_CTF_to_FIF.ipynb, cell 6)
# gives the complete session mapping explicitly, including per-session
# trial counts that match this dataset's real Start-pulse counts. Sessions
# 02/04 (75/69 pulses) are confirmed decoys -- not close to any of H01's
# real Slow/Fast trial counts (116-150 per that same notebook) and absent
# from Thomas's own file list.
#
# H05: no equivalent legacy listing exists, but real evidence resolves it
# completely on its own: sessions 03-10 match TDMS Fast1,Slow1,Fast2,
# Slow2,Fast3,Slow3,Fast4,Slow4 both by chronological order *and* Start-
# pulse count (64/56/63/57/61/56/62/56 vs TDMS trial counts 64/56/62/58/
# 61/56/62/56, all exact or within 1). Session 02 (40 pulses) fits no real
# Slow/Fast run and is the decoy. Session 11 (RT-duration, 41 pulses) is
# chronologically last, matching TDMS RT2's last position (RT1 is
# chronologically first, with no raw RT-duration session anywhere near
# that position) -- RT1 has no raw MEG counterpart on the media at all,
# a genuine recording gap, not an unresolved ambiguity.
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


def media_subject_prefix(subject: str) -> str:
    """Media directory prefix for a canonical subject id (``H01`` -> ``Pilot01``)."""
    subject = normalize_subject_id(subject)
    return "Pilot01" if subject == "H01" else subject


def _duration_class(
    nominal_duration: Optional[float],
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> str:
    if nominal_duration is None:
        return "OTHER"
    if abs(nominal_duration - settings.slowfast_nominal_duration) <= settings.duration_tolerance:
        return "SLOWFAST"
    if abs(nominal_duration - settings.rt_nominal_duration) <= settings.duration_tolerance:
        return "RT"
    return "OTHER"


def _run_class(condition: str) -> str:
    return "RT" if condition.upper() == "RT" else "SLOWFAST"


@dataclass(frozen=True)
class RawSession:
    """One numbered raw acquisition session discovered on media."""

    subject: str
    path: Path
    date: str  # YYYYMMDD, as it appears on media
    index: int
    nominal_duration: Optional[float]
    start_time: Optional[datetime]
    duration_class: str


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

    @property
    def duration_class(self) -> str:
        return _run_class(self.condition)


@dataclass(frozen=True)
class MatchResult:
    """One manifest row: a staging decision for a run, noise, or headshape."""

    subject: str
    kind: str  # "run" | "noise" | "headshape"
    condition: Optional[str]
    run: Optional[int]
    date: str  # acquisition date, YYYYMMDD
    media_path: Optional[Path]
    match_method: str  # "fast_path" | "disambiguated" | "unmatched" | "found" | "not_found"
    meg_start_pulse_count: Optional[int]
    behavior_trial_count: Optional[int]
    count_agreement: str
    action: str  # "stage" | "review"
    note: str = ""


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
    media_root: Path,
    subject: str,
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> List[RawSession]:
    """Chronologically sorted numbered sessions for one subject on media."""
    subject = normalize_subject_id(subject)
    prefix = media_subject_prefix(subject)
    media_root = Path(media_root)

    sessions: List[RawSession] = []
    for ds_dir in sorted(media_root.glob(f"{prefix}_DDM-tthiery_*_*.ds")):
        match = _SESSION_RE.match(ds_dir.name)
        if match is None or match.group("prefix").lower() != prefix.lower():
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


def discover_noise_session(media_root: Path, subject: str, date: str) -> Optional[Path]:
    """The one legacy-convention empty-room recording for a subject's date, if any."""
    candidate = Path(media_root) / _NOISE_TEMPLATE.format(date=date)
    return candidate if candidate.exists() else None


def discover_headshape(media_root: Path, subject: str, date: str) -> Optional[Path]:
    """The digitized headshape file for a subject's date, if any."""
    prefix = media_subject_prefix(normalize_subject_id(subject))
    candidate = Path(media_root) / _HEADSHAPE_TEMPLATE.format(prefix=prefix, date=date)
    return candidate if candidate.exists() else None


def load_behavior_runs(behavior_root: Path, subject: str) -> List[BehaviorRun]:
    """Chronologically sorted behavior runs for a subject, read straight from TDMS.

    Parses every strictly-named ``*.tdms`` run file under the subject's
    input directory directly (``parse_tdms_file(..., infer_random_classes=
    False)``, the same low-level parser ``meg_tokens.behavior.tdms_bids``
    uses) rather than reading ``behavior ingest``'s derivatives output --
    Stage 0 has no dependency on any other stage having run first. Files
    not matching the project's TDMS filename contract
    (``meg_tokens.behavior.tdms.FILENAME_RE``, e.g. a `temp_*.tdms` scratch
    export) are skipped, not raised on.
    """
    subject = normalize_subject_id(subject)
    try:
        subject_dir = find_subject_directory(behavior_root, subject)
    except FileNotFoundError:
        return []

    runs: List[BehaviorRun] = []
    for tdms_path in sorted(subject_dir.glob("*.tdms")):
        if not FILENAME_RE.match(tdms_path.name):
            continue
        run_info = parse_tdms_filename(tdms_path.name)
        df = parse_tdms_file(str(tdms_path), infer_random_classes=False)
        runs.append(
            BehaviorRun(
                subject=subject,
                condition=run_info.condition,
                run=int(run_info.run),
                initial_time_min=float(df["nInitialTime"].min()),
                trial_count=int(len(df)),
            )
        )
    runs.sort(key=lambda r: r.initial_time_min)
    return runs


def count_start_pulses(ds_path: Path, subject: str) -> int:
    """Real trial-start trigger-pulse count for one raw session.

    Opens only the trigger channel via MNE (no full preload). Start pulses
    fire even for trials whose go-cue later dropped out, so this is a more
    reliable audit signal than the go-cue count (see
    ``meg_tokens.meg.epoching``).
    """
    import mne

    from meg_tokens.meg.epoching import get_event_id

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_ctf(str(ds_path), preload=False, verbose=False)
    events = mne.find_events(raw, verbose=False)
    start_code = get_event_id("start", subject)["Start"]
    return int((events[:, 2] == start_code).sum())


def _count_agreement(meg_count: Optional[int], behavior_count: Optional[int], *, tolerance: int) -> str:
    if meg_count is None or behavior_count is None:
        return "not_checked"
    diff = abs(meg_count - behavior_count)
    if diff == 0:
        return "exact"
    if diff <= tolerance:
        return "within_tolerance"
    return "mismatch"


def _known_trailing_mismatch(subject: str, condition: str, run: int) -> bool:
    """Whether this run is one of the 114 documented trailing-boundary
    mismatches in ``meg_tokens.meg.epoching.KNOWN_TRAILING_TRIAL_MISMATCHES``
    -- independently verified (per-run, sub-ms timing residuals, see
    ``docs/behavior_qc_report.md`` section 3) to differ from its MEG
    trigger-pulse count only at the trailing boundary, never in the middle.
    A Start-pulse-count "mismatch" against a documented run like this is
    expected, not a sign the match itself is wrong.
    """
    return (subject, condition, str(run)) in KNOWN_TRAILING_TRIAL_MISMATCHES


def _build_result(
    run: BehaviorRun,
    candidate: Optional[RawSession],
    *,
    date: str,
    match_method: str,
    count_tolerance: int,
    note: str = "",
) -> MatchResult:
    meg_count = count_start_pulses(candidate.path, candidate.subject) if candidate else None
    agreement = _count_agreement(meg_count, run.trial_count, tolerance=count_tolerance)
    action = "stage" if agreement in ("exact", "within_tolerance") else "review"
    if action == "review" and agreement == "mismatch" and _known_trailing_mismatch(
        run.subject, run.condition, run.run
    ):
        action = "stage"
        note = (
            note + " " if note else ""
        ) + (
            "Start-pulse count differs from a documented "
            "KNOWN_TRAILING_TRIAL_MISMATCHES case (see docs/behavior_qc_report.md "
            "section 3) -- independently verified as boundary-only, staging anyway."
        )
    return MatchResult(
        subject=run.subject,
        kind="run",
        condition=run.condition,
        run=run.run,
        date=date,
        media_path=candidate.path if candidate else None,
        match_method=match_method,
        meg_start_pulse_count=meg_count,
        behavior_trial_count=run.trial_count,
        count_agreement=agreement,
        action=action,
        note=note,
    )


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

    Automatic pairing happens in two ways, neither a guess:

    1. ``KNOWN_SESSION_OVERRIDES`` -- a documented, per-run mapping for the
       two real subjects (H01, H05) where an extra or missing raw session
       makes duration-class + order insufficient on their own. Every entry
       is backed by real evidence (a legacy notebook's own session list, or
       this project's own count/position analysis) recorded in
       ``docs/meg_t0_7_raw_bidsification_plan.md``, not inferred here.
    2. Within whatever is left in a duration class (RT or SLOWFAST) after
       overrides are applied, if the remaining candidate-session count
       equals the remaining run count, pair them 1:1 in chronological
       order -- both lists are independently sorted by real, objective
       timestamps (the raw session's own ``.hist`` ``DATE:`` and the
       behavior run's ``nInitialTime``), so this is applying the
       acquisition order both systems already recorded, not inferring one.

    Whenever neither applies, nothing is picked automatically -- there is
    no scoring/best-fit step. Every such run is reported
    ``match_method="ambiguous"`` with every real remaining candidate
    session's Start-pulse count listed in its `note`, so a human has the
    evidence to fill in the correct ``media_path`` by hand (see
    ``docs/meg_t0_7_raw_bidsification_plan.md``).
    """
    count_tolerance = settings.count_tolerance
    results: List[MatchResult] = []
    for run_class in ("RT", "SLOWFAST"):
        class_candidates = {c.index: c for c in raw_sessions if c.duration_class == run_class}
        class_runs = [r for r in behavior_runs if r.duration_class == run_class]
        if not class_runs:
            continue

        remaining_runs: List[BehaviorRun] = []
        for run in class_runs:
            override_index = KNOWN_SESSION_OVERRIDES.get((run.subject, run.condition, run.run))
            candidate = class_candidates.pop(override_index, None) if override_index is not None else None
            if candidate is not None:
                results.append(
                    _build_result(
                        run,
                        candidate,
                        date=date,
                        match_method="known_override",
                        count_tolerance=count_tolerance,
                        note=(
                            "Resolved via docs/meg_t0_7_raw_bidsification_plan.md "
                            "\"H01/H05 resolution\" (KNOWN_SESSION_OVERRIDES), not "
                            "duration/order/count alone."
                        ),
                    )
                )
            else:
                remaining_runs.append(run)

        if not remaining_runs:
            continue

        remaining_candidates = sorted(
            class_candidates.values(), key=lambda c: (c.start_time or datetime.min, c.index)
        )
        if len(remaining_candidates) == len(remaining_runs):
            for candidate, run in zip(remaining_candidates, remaining_runs):
                results.append(
                    _build_result(
                        run, candidate, date=date, match_method="fast_path", count_tolerance=count_tolerance
                    )
                )
        else:
            candidate_counts = {
                candidate.path: count_start_pulses(candidate.path, candidate.subject)
                for candidate in remaining_candidates
            }
            candidates_desc = "; ".join(
                f"{candidate.path.name} (start pulses={candidate_counts[candidate.path]})"
                for candidate in remaining_candidates
            ) or "none"
            for run in remaining_runs:
                results.append(
                    _build_result(
                        run,
                        None,
                        date=date,
                        match_method="ambiguous",
                        count_tolerance=count_tolerance,
                        note=(
                            f"{len(remaining_candidates)} candidate session(s) for "
                            f"{len(remaining_runs)} remaining {run_class} run(s) -- not "
                            f"resolved automatically. This run needs {run.trial_count} "
                            f"trials. Candidates: {candidates_desc}."
                        ),
                    )
                )

    results.sort(key=lambda r: (r.condition or "", r.run or 0))
    return results


def match_noise_and_headshape(
    media_root: Path, subject: str, date: str
) -> List[MatchResult]:
    """Manifest rows for the empty-room noise recording and headshape file."""
    subject = normalize_subject_id(subject)
    results = []
    noise_path = discover_noise_session(media_root, subject, date)
    results.append(
        MatchResult(
            subject=subject,
            kind="noise",
            condition=None,
            run=None,
            date=date,
            media_path=noise_path,
            match_method="found" if noise_path else "not_found",
            meg_start_pulse_count=None,
            behavior_trial_count=None,
            count_agreement="not_applicable",
            action="stage" if noise_path else "review",
            note="" if noise_path else f"No NOISE_noise_{date}_01.ds found.",
        )
    )
    headshape_path = discover_headshape(media_root, subject, date)
    results.append(
        MatchResult(
            subject=subject,
            kind="headshape",
            condition=None,
            run=None,
            date=date,
            media_path=headshape_path,
            match_method="found" if headshape_path else "not_found",
            meg_start_pulse_count=None,
            behavior_trial_count=None,
            count_agreement="not_applicable",
            action="stage" if headshape_path else "review",
            note="" if headshape_path else "No _HEADSHAPE.eeg found.",
        )
    )
    return results


def match_media_to_behavior(
    media_root: Path,
    behavior_root: Path,
    subject: str,
    *,
    settings: RawStagingConfig = RawStagingConfig(),
) -> List[MatchResult]:
    """Full staging decision for one subject: runs + noise + headshape.

    ``behavior_root`` is the raw TDMS root (``ProjectConfig.behavior_root``,
    i.e. ``data_root/tdms``) -- Stage 0 reads TDMS directly and does not
    require ``behavior ingest`` to have run first.
    """
    subject = normalize_subject_id(subject)
    raw_sessions = discover_raw_sessions(media_root, subject, settings=settings)
    behavior_runs = load_behavior_runs(behavior_root, subject)
    if not raw_sessions:
        raise FileNotFoundError(
            f"No raw sessions found for {subject} under {media_root}"
        )
    if not behavior_runs:
        raise FileNotFoundError(
            f"No TDMS runs found for {subject} under {behavior_root}"
        )

    date = raw_sessions[0].date
    results = match_subject(raw_sessions, behavior_runs, date=date, settings=settings)
    results.extend(match_noise_and_headshape(media_root, subject, date))
    return results
