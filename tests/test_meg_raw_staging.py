"""Stage 0 discovery and matching, tested against the real dataset.

These tests read genuine recordings and behavioral logs rather than
constructed stand-ins. The properties under test -- that two independent
clocks agree to sub-millisecond precision on the same run and disagree by
orders of magnitude on any other, that nominal durations fall into three
clean protocol clusters, that a subject's sessions are numbered in
acquisition order -- are facts about the acquisition hardware and this
dataset. A constructed input would only restate the assumption being
checked, so it could not fail when the assumption is wrong.

They read ``raw/`` and ``tdms/`` under the ``data_root`` of this checkout's
``tokens.toml`` (see ``tests/conftest.py``), and skip where those are absent.
"""

import numpy as np
import pytest

from meg_tokens.meg.raw_staging import (
    KNOWN_SESSION_OVERRIDES,
    _best_window_error,
    discover_noise_session,
    discover_raw_sessions,
    load_behavior_runs,
    match_raw_to_behavior,
    match_subject,
    read_start_pulse_times,
)

# --------------------------------------------------------------------------
# Pure arithmetic: the alignment scorer, pinned on plain number sequences.
# These describe no recording, so there is nothing to fabricate.
# --------------------------------------------------------------------------


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


def test_discover_noise_session_redirects_the_known_corrupt_recording(raw_data):
    # NOISE_noise_20181206_01.ds (H26/H27's shared empty-room) has a corrupt
    # .hc file -- reading it raises RuntimeError("HPI information not
    # available") -- so KNOWN_NOISE_OVERRIDES points this date at the valid
    # operator retake, _02.ds, instead.
    found = discover_noise_session(raw_data.raw_meg_root, "H26", "20181206")
    assert found is not None
    assert found.name == "NOISE_noise_20181206_02.ds"


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def test_read_start_pulse_times_returns_a_real_pulse_train(raw_data):
    sessions = discover_raw_sessions(raw_data.raw_meg_root, "H02")
    task_sessions = [s for s in sessions if s.duration_class != "OTHER"]

    times = read_start_pulse_times(task_sessions[0].path, "H02")

    assert len(times) > 0
    assert times[0] == 0.0  # relative to the first pulse
    assert np.all(np.diff(times) > 0)


def test_fingerprint_reproduces_every_pinned_override_for_h05(raw_data):
    # H05 has an extra Slow/Fast candidate and only one RT session, so neither
    # order nor counts can resolve it -- every pinned override has to come from
    # the recordings themselves.
    sessions = discover_raw_sessions(raw_data.raw_meg_root, "H05")
    runs = load_behavior_runs(raw_data.behavior_root, "H05")
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


def test_fingerprint_rejects_the_only_candidate_when_the_run_was_not_recorded(raw_data):
    # H05 RT1 has no recording at all; its one RT-duration candidate belongs to
    # RT2. A best-available matcher would take it -- the guards must not.
    sessions = discover_raw_sessions(raw_data.raw_meg_root, "H05")
    runs = load_behavior_runs(raw_data.behavior_root, "H05")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20180220")}

    assert results[("RT", 1)].match_method == "ambiguous"
    assert results[("RT", 1)].source_path is None
    assert results[("RT", 1)].action == "review"
    assert "0 candidate session(s)" in results[("RT", 1)].note


def test_fingerprint_resolves_a_run_whose_recording_started_late(raw_data):
    # H12 Slow2's recording begins several trials into the run and carries an
    # extra trailing pulse, so its pulse train is not a sub-sequence of the log
    # in either direction.
    sessions = discover_raw_sessions(raw_data.raw_meg_root, "H12")
    runs = load_behavior_runs(raw_data.behavior_root, "H12")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20181010")}

    slow2 = results[("Slow", 2)]
    assert slow2.match_method == "fingerprint"
    assert slow2.fingerprint_error_ms < 1.0
    assert slow2.fingerprint_separation > 20.0


def test_documented_trailing_mismatch_still_stages_with_an_explanatory_note(raw_data):
    # H01 Slow3's pulse count differs from its logged trial count by more than
    # count_tolerance, but it is a documented boundary-only case, so it must
    # stage rather than drop to review.
    sessions = discover_raw_sessions(raw_data.raw_meg_root, "H01")
    runs = load_behavior_runs(raw_data.behavior_root, "H01")
    results = {(r.condition, r.run): r for r in match_subject(sessions, runs, date="20180131")}

    slow3 = results[("Slow", 3)]
    assert slow3.count_agreement == "mismatch"
    assert slow3.action == "stage"
    assert "KNOWN_TRAILING_TRIAL_MISMATCHES" in slow3.note


def test_match_raw_to_behavior_covers_every_kind_of_staged_file(raw_data):
    # The manifest is the whole Stage 0 plan for a subject: runs plus every
    # per-subject asset, anatomical included, so one review pass sees it all.
    results = match_raw_to_behavior(
        raw_data.raw_meg_root,
        raw_data.behavior_root,
        "H02",
        subjects_dir=raw_data.subjects_dir,
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
