"""Stage 0 discovery and matching, tested against the real dataset.

These tests read genuine recordings and behavioral logs rather than
constructed stand-ins. The properties under test -- that two independent
clocks agree to sub-millisecond precision on the same run and disagree by
orders of magnitude on any other, that nominal durations fall into three
clean protocol clusters, that a subject's sessions are numbered in
acquisition order -- are facts about the acquisition hardware and this
dataset. A constructed input would only restate the assumption being
checked, so it could not fail when the assumption is wrong.

Point ``MEG_TOKENS_DATA_ROOT`` at the project's ``data_root`` -- the same one
the project TOML declares, holding ``raw/`` and ``tdms/`` -- to run them:

    MEG_TOKENS_DATA_ROOT=<data-root> python -m pytest tests/test_meg_raw_staging.py

Everything skips when it is unset, so a checkout without the dataset still
runs the rest of the suite.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from meg_tokens.core import ProjectConfig
from meg_tokens.meg.raw_staging import (
    KNOWN_SESSION_OVERRIDES,
    _best_window_error,
    discover_headshape,
    discover_noise_session,
    discover_raw_sessions,
    load_behavior_runs,
    match_raw_to_behavior,
    match_subject,
    read_hist_metadata,
    read_start_pulse_times,
)

_DATA_ROOT = os.environ.get("MEG_TOKENS_DATA_ROOT")
_PROJECT = (
    ProjectConfig(
        data_root=_DATA_ROOT,
        # The project TOML declares this; the real layout keeps the FreeSurfer
        # reconstructions alongside raw/ and tdms/ under data_root.
        subjects_dir=os.environ.get("MEG_TOKENS_SUBJECTS_DIR", Path(_DATA_ROOT) / "IRM"),
    )
    if _DATA_ROOT
    else None
)

requires_real_data = pytest.mark.skipif(
    _PROJECT is None
    or not _PROJECT.raw_meg_root.is_dir()
    or not _PROJECT.behavior_root.is_dir(),
    reason=(
        "Set MEG_TOKENS_DATA_ROOT to the project's data_root (containing raw/ "
        "and tdms/) to run Stage 0 real-data tests."
    ),
)


@pytest.fixture()
def project() -> ProjectConfig:
    return _PROJECT


# --------------------------------------------------------------------------
# Pure arithmetic: the alignment scorer, pinned on plain number sequences.
# These describe no recording, so there is nothing to fabricate.
# --------------------------------------------------------------------------

def test_best_window_error_handles_extra_leading_pulses():
    run = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    candidate = np.array([900.0, 800.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    error, shift = _best_window_error(candidate, run)
    assert error == pytest.approx(0.0)
    assert shift == -2  # the recording has two extra leading pulses


def test_best_window_error_handles_surplus_at_both_ends():
    # The shape that broke the first implementation: surplus at both ends at
    # once, so neither sequence is contained in the other.
    run = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    candidate = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 999.0])
    error, shift = _best_window_error(candidate, run)
    assert error == pytest.approx(0.0)
    assert shift == 3  # the recording missed the run's first three trials


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

@requires_real_data
def test_read_hist_metadata_parses_a_real_session_header(project):
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")
    assert sessions, "H02 should have discoverable sessions"

    metadata = read_hist_metadata(sessions[0].path)

    assert metadata["nominal_duration"] > 0
    assert metadata["start_time"] is not None


@requires_real_data
def test_discover_raw_sessions_returns_one_subject_in_acquisition_order(project):
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")

    assert all(session.subject == "H02" for session in sessions)
    assert all(session.path.name.startswith("H02_") for session in sessions)
    start_times = [s.start_time for s in sessions if s.start_time is not None]
    assert start_times == sorted(start_times)
    # A subject's sessions are numbered in acquisition order by the scanner,
    # so chronological sorting must not reorder them.
    assert [s.index for s in sessions] == sorted(s.index for s in sessions)


@requires_real_data
def test_discover_raw_sessions_buckets_durations_into_protocol_classes(project):
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")
    by_class = {}
    for session in sessions:
        by_class.setdefault(session.duration_class, []).append(session.nominal_duration)

    # Eight Slow/Fast runs and two RT runs per complete subject.
    assert len(by_class.get("SLOWFAST", [])) == 8
    assert len(by_class.get("RT", [])) == 2
    assert all(abs(d - 315.0) < 20.0 for d in by_class["SLOWFAST"])
    assert all(abs(d - 135.0) < 20.0 for d in by_class["RT"])


@requires_real_data
def test_discover_noise_session_and_headshape_find_the_real_companions(project):
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")
    date = sessions[0].date

    assert discover_noise_session(project.raw_meg_root, "H02", date) is not None
    assert discover_headshape(project.raw_meg_root, "H02", date) is not None
    # A date with no acquisition has neither.
    assert discover_noise_session(project.raw_meg_root, "H02", "19000101") is None
    assert discover_headshape(project.raw_meg_root, "H02", "19000101") is None


@requires_real_data
def test_load_behavior_runs_reads_tdms_without_any_ingest_output(project):
    runs = load_behavior_runs(project.behavior_root, "H02")

    assert len(runs) == 10  # 4 Slow + 4 Fast + 2 RT
    assert {run.condition for run in runs} == {"Slow", "Fast", "RT"}
    assert [run.initial_time_min for run in runs] == sorted(
        run.initial_time_min for run in runs
    )
    for run in runs:
        assert run.trial_count == len(run.initial_times_ms)
        assert len(run.intervals_ms) == run.trial_count - 1
        # nInitialTime is a monotonic clock within a run, so every gap is positive.
        assert np.all(run.intervals_ms > 0)


@requires_real_data
def test_load_behavior_runs_returns_empty_for_a_subject_with_no_logs(project):
    assert load_behavior_runs(project.behavior_root, "H99") == []


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

@requires_real_data
def test_read_start_pulse_times_returns_a_real_pulse_train(project):
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")
    task_sessions = [s for s in sessions if s.duration_class != "OTHER"]

    times = read_start_pulse_times(task_sessions[0].path, "H02")

    assert len(times) > 0
    assert times[0] == 0.0  # relative to the first pulse
    assert np.all(np.diff(times) > 0)


@requires_real_data
def test_match_subject_fingerprints_every_run_of_an_unambiguous_subject(project):
    # H02 has exactly the expected candidates, so order alone would suffice --
    # the fingerprint must still identify each run, and agree with that order.
    sessions = discover_raw_sessions(project.raw_meg_root, "H02")
    runs = load_behavior_runs(project.behavior_root, "H02")

    results = match_subject(sessions, runs, date=sessions[0].date)

    assert len(results) == 10
    assert all(result.match_method == "fingerprint" for result in results)
    assert all(result.action == "stage" for result in results)
    assert all(result.fingerprint_error_ms < 1.0 for result in results)
    assert all(result.fingerprint_separation > 20.0 for result in results)
    # Distinct runs never share a recording.
    assert len({result.source_path for result in results}) == len(results)


@requires_real_data
def test_fingerprint_reproduces_every_pinned_override_for_h05(project):
    # H05 has an extra Slow/Fast candidate and only one RT session, so neither
    # order nor counts can resolve it -- every pinned override has to come from
    # the recordings themselves.
    sessions = discover_raw_sessions(project.raw_meg_root, "H05")
    runs = load_behavior_runs(project.behavior_root, "H05")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20180220")}

    pinned = {
        (condition, run): index
        for (subject, condition, run), index in KNOWN_SESSION_OVERRIDES.items()
        if subject == "H05"
    }
    assert pinned, "H05 should have pinned overrides to reproduce"
    for key, expected_index in pinned.items():
        result = results[key]
        assert result.match_method == "fingerprint", key
        assert result.source_path.name.endswith(f"_{expected_index:02d}.ds"), key
        assert result.fingerprint_error_ms < 1.0, key
        assert "DISAGREES" not in result.note, key


@requires_real_data
def test_fingerprint_rejects_the_only_candidate_when_the_run_was_not_recorded(project):
    # H05 RT1 has no recording at all; its one RT-duration candidate belongs to
    # RT2. A best-available matcher would take it -- the guards must not.
    sessions = discover_raw_sessions(project.raw_meg_root, "H05")
    runs = load_behavior_runs(project.behavior_root, "H05")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20180220")}

    assert results[("RT", 1)].match_method == "ambiguous"
    assert results[("RT", 1)].source_path is None
    assert results[("RT", 1)].action == "review"
    assert "0 candidate session(s)" in results[("RT", 1)].note


@requires_real_data
def test_fingerprint_resolves_a_run_whose_recording_started_late(project):
    # H12 Slow2's recording begins several trials into the run and carries an
    # extra trailing pulse, so its pulse train is not a sub-sequence of the log
    # in either direction.
    sessions = discover_raw_sessions(project.raw_meg_root, "H12")
    runs = load_behavior_runs(project.behavior_root, "H12")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20181010")}

    slow2 = results[("Slow", 2)]
    assert slow2.match_method == "fingerprint"
    assert slow2.fingerprint_error_ms < 1.0
    assert slow2.fingerprint_separation > 20.0


@requires_real_data
def test_documented_trailing_mismatch_still_stages_with_an_explanatory_note(project):
    # H01 Slow3's pulse count differs from its logged trial count by more than
    # count_tolerance, but it is a documented boundary-only case, so it must
    # stage rather than drop to review.
    sessions = discover_raw_sessions(project.raw_meg_root, "H01")
    runs = load_behavior_runs(project.behavior_root, "H01")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20180131")}

    slow3 = results[("Slow", 3)]
    assert slow3.count_agreement == "mismatch"
    assert slow3.action == "stage"
    assert "KNOWN_TRAILING_TRIAL_MISMATCHES" in slow3.note


@requires_real_data
def test_match_raw_to_behavior_covers_every_kind_of_staged_file(project):
    # The manifest is the whole Stage 0 plan for a subject: runs plus every
    # per-subject asset, anatomical included, so one review pass sees it all.
    results = match_raw_to_behavior(
        project.raw_meg_root,
        project.behavior_root,
        "H02",
        subjects_dir=project.subjects_dir,
    )

    kinds = [result.kind for result in results]
    assert kinds.count("run") == 10
    assert kinds.count("noise") == 1
    assert kinds.count("headshape") == 1
    assert kinds.count("anat") == 1
    assert all(result.action == "stage" for result in results)
    assert all(result.source_path is not None for result in results)

    anat = next(result for result in results if result.kind == "anat")
    assert anat.match_method == "found"
    assert anat.source_path.name == "rawavg.mgz"
    # The MRI is a separate acquisition from the MEG session, so stamping it
    # with the MEG date would assert something untrue.
    assert anat.date == ""


@requires_real_data
def test_missing_reconstruction_becomes_a_review_row_not_an_error(project):
    # H10 has MEG but no FreeSurfer reconstruction. That gap belongs in the
    # manifest next to every other gap, not in a separate command's output.
    results = match_raw_to_behavior(
        project.raw_meg_root,
        project.behavior_root,
        "H10",
        subjects_dir=project.subjects_dir,
    )

    anat = next(result for result in results if result.kind == "anat")
    assert anat.match_method == "not_found"
    assert anat.source_path is None
    assert anat.action == "review"
    assert "rawavg.mgz" in anat.note


@requires_real_data
def test_anat_row_is_emitted_even_when_subjects_dir_is_unconfigured(project):
    results = match_raw_to_behavior(
        project.raw_meg_root, project.behavior_root, "H02", subjects_dir=None
    )

    anat = next(result for result in results if result.kind == "anat")
    assert anat.match_method == "not_found"
    assert "not configured" in anat.note


@requires_real_data
def test_match_raw_to_behavior_raises_for_a_subject_with_no_logs(project):
    with pytest.raises(FileNotFoundError):
        match_raw_to_behavior(project.raw_meg_root, project.behavior_root, "H99")
