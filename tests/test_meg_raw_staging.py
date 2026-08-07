import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from meg_tokens.core import RawStagingConfig
from meg_tokens.meg.raw_staging import (
    BehaviorRun,
    RawSession,
    discover_headshape,
    discover_noise_session,
    discover_raw_sessions,
    load_behavior_runs,
    match_media_to_behavior,
    match_subject,
    media_subject_prefix,
    read_hist_metadata,
)

_EVENTS_TEMPLATE = """sTaskType: 'TokensMvt'
dDate: '2018/02/13 02:20:52.738 PM'
nInitialTime: {initial_time}
nTrialIndex: {trial_index}
sNeuralFilename: 'H2Slow1'
nChoiceMade: 2
nCorrectChoice: 2
nSelectedTarget: 4
nUnselectedTarget: 1
tEnterCenter: 3
tClick: 3
tGO: 1041
tExitCenter: 2484
tEnterTarget: 2484
tTrialEnd: 3676
nOutcome: 0
sTrialClass: 'x'
sTokenDirs: '221221122212211'
Tokens.Data: (
[tTime: 1260, nTokenNum: 1, nTokenDir: 2, nProb: 0.70947265625000, nTokenX: 575, nTokenY: 474],
[tTime: 1460, nTokenNum: 2, nTokenDir: 2, nProb: 0.81234567890123, nTokenX: 576, nTokenY: 475]
)
"""


class _FakeGroup:
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties


class _FakeTdmsFile:
    def __init__(self, groups):
        self._groups = groups

    def groups(self):
        return self._groups


def _fake_tdms_groups(trial_specs):
    return [
        _FakeGroup(
            f"TrialData{i}",
            {"Events": _EVENTS_TEMPLATE.format(initial_time=initial_time, trial_index=i + 1)},
        )
        for i, initial_time in enumerate(trial_specs)
    ]


def _write_hist(ds_dir: Path, *, duration, start_time: str) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    name = ds_dir.name[: -len(".ds")]
    text = f"DATE:\n{start_time}\n\nOPERATOR:\nmeg\n\nSample rate: 1200\n"
    if duration is not None:
        text += f"Trial duration: {duration}\n"
    (ds_dir / f"{name}.hist").write_text(text)


def _make_session(media_root: Path, prefix: str, date: str, index: str, *, duration, start_time: str) -> Path:
    ds_dir = media_root / f"{prefix}_DDM-tthiery_{date}_{index}.ds"
    _write_hist(ds_dir, duration=duration, start_time=start_time)
    return ds_dir


def test_media_subject_prefix_pilot_mapping():
    assert media_subject_prefix("H01") == "Pilot01"
    assert media_subject_prefix("H1") == "Pilot01"
    assert media_subject_prefix("H02") == "H02"


def test_read_hist_metadata_parses_duration_and_start_time(tmp_path):
    ds_dir = _make_session(
        tmp_path, "H02", "20180213", "03", duration=315.0, start_time="13-Feb-2018 14:17"
    )
    meta = read_hist_metadata(ds_dir)
    assert meta["nominal_duration"] == 315.0
    assert meta["start_time"] == datetime(2018, 2, 13, 14, 17)


def test_read_hist_metadata_missing_hist_file_returns_none(tmp_path):
    ds_dir = tmp_path / "H02_DDM-tthiery_20180213_99.ds"
    ds_dir.mkdir()
    assert read_hist_metadata(ds_dir) is None


def test_discover_raw_sessions_sorts_chronologically_and_buckets_duration(tmp_path):
    # Deliberately out of filename order -- chronological (.hist DATE:) order
    # must win, matching real acquisition-time ordering, not lexical sort.
    _make_session(tmp_path, "H02", "20180213", "02", duration=135.0, start_time="13-Feb-2018 09:05")
    _make_session(tmp_path, "H02", "20180213", "01", duration=360.0, start_time="13-Feb-2018 09:00")
    _make_session(tmp_path, "H02", "20180213", "03", duration=315.0, start_time="13-Feb-2018 09:10")

    sessions = discover_raw_sessions(tmp_path, "H02")

    assert [s.index for s in sessions] == [1, 2, 3]
    assert [s.duration_class for s in sessions] == ["OTHER", "RT", "SLOWFAST"]


def test_discover_raw_sessions_uses_pilot01_prefix_for_h01(tmp_path):
    _make_session(tmp_path, "Pilot01", "20180131", "01", duration=135.0, start_time="31-Jan-2018 09:00")
    _make_session(tmp_path, "H02", "20180213", "01", duration=135.0, start_time="13-Feb-2018 09:00")

    sessions = discover_raw_sessions(tmp_path, "H01")

    assert len(sessions) == 1
    assert sessions[0].date == "20180131"


def test_discover_raw_sessions_honors_settings_tolerance(tmp_path):
    _make_session(tmp_path, "H02", "20180213", "01", duration=310.0, start_time="13-Feb-2018 09:00")
    default = discover_raw_sessions(tmp_path, "H02")
    assert default[0].duration_class == "SLOWFAST"

    strict = discover_raw_sessions(
        tmp_path, "H02", settings=RawStagingConfig(duration_tolerance=2.0)
    )
    assert strict[0].duration_class == "OTHER"


def test_discover_noise_session_matches_exact_legacy_pattern_only(tmp_path):
    (tmp_path / "NOISE_noise_20180213_01.ds").mkdir()
    (tmp_path / "NOISE1Trial5min_noise_20180213_01.ds").mkdir()

    assert discover_noise_session(tmp_path, "H02", "20180213") == (
        tmp_path / "NOISE_noise_20180213_01.ds"
    )
    assert discover_noise_session(tmp_path, "H02", "20180214") is None


def test_discover_headshape_uses_media_prefix(tmp_path):
    (tmp_path / "H02_DDM-tthiery_20180213_HEADSHAPE.eeg").write_text("")

    assert discover_headshape(tmp_path, "H02", "20180213") is not None
    assert discover_headshape(tmp_path, "H02", "20180214") is None
    assert discover_headshape(tmp_path, "H01", "20180213") is None


def test_load_behavior_runs_reads_tdms_directly_no_ingest_dependency(tmp_path, monkeypatch):
    # Stage 0 must work with only tdms/ populated -- no
    # derivatives/meg-tokens/sub-*/beh/ output from `behavior ingest`
    # anywhere on disk.
    fake = _FakeTdmsFile(_fake_tdms_groups([300, 100, 200]))
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    tdms_dir = tmp_path / "tdms" / "H02"
    tdms_dir.mkdir(parents=True)
    (tdms_dir / "H2Slow1_180213.tdms").write_text("")

    runs = load_behavior_runs(tmp_path / "tdms", "H02")

    assert len(runs) == 1
    assert runs[0].condition == "Slow"
    assert runs[0].run == 1
    assert runs[0].trial_count == 3
    assert runs[0].initial_time_min == 100.0
    assert runs[0].duration_class == "SLOWFAST"


def test_load_behavior_runs_resolves_h01s_unpadded_h1_folder(tmp_path, monkeypatch):
    fake = _FakeTdmsFile(_fake_tdms_groups([100]))
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    tdms_dir = tmp_path / "tdms" / "H1"
    tdms_dir.mkdir(parents=True)
    (tdms_dir / "H1Slow1_180131.tdms").write_text("")

    runs = load_behavior_runs(tmp_path / "tdms", "H01")

    assert len(runs) == 1
    assert runs[0].subject == "H01"


def test_load_behavior_runs_skips_non_standard_filenames(tmp_path, monkeypatch):
    fake = _FakeTdmsFile(_fake_tdms_groups([100]))
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake)

    tdms_dir = tmp_path / "tdms" / "H02"
    tdms_dir.mkdir(parents=True)
    (tdms_dir / "temp_180213.tdms").write_text("")

    assert load_behavior_runs(tmp_path / "tdms", "H02") == []


def test_load_behavior_runs_returns_empty_for_missing_subject_dir(tmp_path):
    (tmp_path / "tdms").mkdir()
    assert load_behavior_runs(tmp_path / "tdms", "H99") == []


def _session(
    name: str, duration_class: str, index: int, start: datetime, *, subject: str = "H02"
) -> RawSession:
    return RawSession(
        subject=subject,
        path=Path(name),
        date="20180213",
        index=index,
        nominal_duration=None,
        start_time=start,
        duration_class=duration_class,
    )


def _run(
    condition: str, run: int, trial_count: int, initial_time: float, *, subject: str = "H02"
) -> BehaviorRun:
    return BehaviorRun(
        subject=subject,
        condition=condition,
        run=run,
        initial_time_min=initial_time,
        trial_count=trial_count,
    )


def _patch_counts(monkeypatch, counts: dict) -> None:
    monkeypatch.setattr(
        "meg_tokens.meg.raw_staging.count_start_pulses",
        lambda path, subject: counts[str(path)],
    )


def test_match_subject_fast_path_pairs_in_chronological_order(monkeypatch):
    sessions = [
        _session("s1.ds", "SLOWFAST", 1, datetime(2018, 1, 1, 9, 0)),
        _session("s2.ds", "SLOWFAST", 2, datetime(2018, 1, 1, 9, 10)),
    ]
    runs = [_run("Slow", 1, 56, 100), _run("Slow", 2, 57, 200)]
    _patch_counts(monkeypatch, {"s1.ds": 56, "s2.ds": 57})

    results = match_subject(sessions, runs, date="20180213")

    assert [r.match_method for r in results] == ["fast_path", "fast_path"]
    assert [r.action for r in results] == ["stage", "stage"]
    assert [r.media_path for r in results] == [Path("s1.ds"), Path("s2.ds")]
    assert [r.count_agreement for r in results] == ["exact", "exact"]


def test_match_subject_never_auto_picks_among_extra_candidates_h01_shape(monkeypatch):
    # H01-shape: one extra SLOWFAST candidate (a non-canonical/decoy
    # session) beyond the two real runs, verified against the real dataset.
    # Even though the real Start-pulse counts would make an unambiguous
    # best-fit choice possible, this must NOT be auto-picked -- no scoring
    # heuristic, ever. Every run in the ambiguous class is flagged for a
    # human to resolve by hand, with the real counts given as evidence.
    sessions = [
        _session("s1.ds", "SLOWFAST", 1, datetime(2018, 1, 1, 9, 0)),
        _session("decoy.ds", "SLOWFAST", 2, datetime(2018, 1, 1, 9, 5)),
        _session("s3.ds", "SLOWFAST", 3, datetime(2018, 1, 1, 9, 10)),
    ]
    runs = [_run("Slow", 1, 56, 100), _run("Slow", 2, 60, 200)]
    _patch_counts(monkeypatch, {"s1.ds": 56, "decoy.ds": 999, "s3.ds": 60})

    results = match_subject(sessions, runs, date="20180213")

    assert len(results) == 2
    assert all(r.match_method == "ambiguous" for r in results)
    assert all(r.action == "review" for r in results)
    assert all(r.media_path is None for r in results)
    by_run = {r.run: r for r in results}
    assert "s1.ds (start pulses=56)" in by_run[1].note
    assert "decoy.ds (start pulses=999)" in by_run[1].note
    assert "s3.ds (start pulses=60)" in by_run[1].note
    assert "3 candidate session(s) for 2 remaining SLOWFAST run(s)" in by_run[1].note


def test_match_subject_flags_both_runs_ambiguous_when_a_candidate_is_missing_h05_shape(monkeypatch):
    # H05-shape: only one RT-duration candidate for two RT runs, verified
    # against the real dataset. Both runs are reported ambiguous (not one
    # auto-matched and one left unmatched) -- there is no basis to prefer
    # RT1 over RT2 for the sole candidate without guessing.
    sessions = [_session("s1.ds", "RT", 1, datetime(2018, 1, 1, 9, 0))]
    runs = [_run("RT", 1, 40, 100), _run("RT", 2, 40, 200)]
    _patch_counts(monkeypatch, {"s1.ds": 40})

    results = match_subject(sessions, runs, date="20180213")

    assert len(results) == 2
    assert all(r.match_method == "ambiguous" for r in results)
    assert all(r.action == "review" for r in results)
    assert all(r.media_path is None for r in results)
    by_run = {r.run: r for r in results}
    assert "1 candidate session(s) for 2 remaining RT run(s)" in by_run[1].note
    assert "s1.ds (start pulses=40)" in by_run[2].note


def test_match_subject_flags_count_mismatch_for_review_not_force_stage(monkeypatch):
    sessions = [_session("s1.ds", "SLOWFAST", 1, datetime(2018, 1, 1, 9, 0))]
    runs = [_run("Slow", 1, 56, 100)]
    _patch_counts(monkeypatch, {"s1.ds": 20})  # far outside tolerance

    results = match_subject(sessions, runs, date="20180213")

    assert results[0].match_method == "fast_path"
    assert results[0].count_agreement == "mismatch"
    assert results[0].action == "review"
    # a mismatch is still reported with its media_path -- reviewable, not
    # silently dropped.
    assert results[0].media_path == Path("s1.ds")


def test_match_subject_stages_known_trailing_mismatch_without_review(monkeypatch):
    # H10 Fast2 is a real entry in
    # meg_tokens.meg.epoching.KNOWN_TRAILING_TRIAL_MISMATCHES, independently
    # verified (docs/behavior_qc_report.md section 3) as a boundary-only
    # discrepancy. A Start-pulse-count "mismatch" against it must not block
    # staging the way an undocumented mismatch does.
    sessions = [_session("s1.ds", "SLOWFAST", 1, datetime(2018, 1, 1, 9, 0), subject="H10")]
    runs = [_run("Fast", 2, 67, 100, subject="H10")]
    _patch_counts(monkeypatch, {"s1.ds": 63})  # diff=4, beyond default tolerance=3

    results = match_subject(sessions, runs, date="20180302")

    assert results[0].count_agreement == "mismatch"
    assert results[0].action == "stage"
    assert "KNOWN_TRAILING_TRIAL_MISMATCHES" in results[0].note


def test_match_subject_applies_known_session_override_for_h01(monkeypatch):
    # H01's real ambiguity (10 SLOWFAST-duration candidates for 8 runs) is
    # resolved via KNOWN_SESSION_OVERRIDES (see
    # docs/meg_t0_7_raw_bidsification_plan.md "H01/H05 resolution"), not a
    # scoring guess -- the two decoy sessions (indices 2 and 4 here) are
    # simply never claimed and never appear in the output.
    sessions = [
        _session("decoy2.ds", "SLOWFAST", 2, datetime(2018, 1, 31, 9, 0), subject="H01"),
        _session("s05.ds", "SLOWFAST", 5, datetime(2018, 1, 31, 9, 10), subject="H01"),
        _session("decoy4.ds", "SLOWFAST", 4, datetime(2018, 1, 31, 9, 5), subject="H01"),
        _session("s06.ds", "SLOWFAST", 6, datetime(2018, 1, 31, 9, 15), subject="H01"),
    ]
    runs = [
        _run("Slow", 1, 117, 100, subject="H01"),
        _run("Fast", 1, 144, 200, subject="H01"),
    ]
    _patch_counts(monkeypatch, {"decoy2.ds": 75, "decoy4.ds": 69, "s05.ds": 117, "s06.ds": 144})

    results = match_subject(sessions, runs, date="20180131")

    assert len(results) == 2
    by_run = {(r.condition, r.run): r for r in results}
    assert by_run[("Slow", 1)].media_path == Path("s05.ds")
    assert by_run[("Fast", 1)].media_path == Path("s06.ds")
    assert all(r.match_method == "known_override" for r in results)
    assert all(r.action == "stage" for r in results)


def test_match_subject_h05_override_leaves_rt1_genuinely_unmatched(monkeypatch):
    # H05's real shape: one RT-duration session, claimed by the RT2
    # override (position + count evidence specific to H05, see the plan
    # doc); RT1 has no raw counterpart at all and must be reported with
    # zero candidates, not offered the already-claimed session.
    sessions = [_session("s11.ds", "RT", 11, datetime(2018, 2, 20, 14, 47), subject="H05")]
    runs = [
        _run("RT", 1, 39, 100, subject="H05"),
        _run("RT", 2, 40, 200, subject="H05"),
    ]
    _patch_counts(monkeypatch, {"s11.ds": 41})

    results = match_subject(sessions, runs, date="20180220")

    by_run = {r.run: r for r in results}
    assert by_run[2].match_method == "known_override"
    assert by_run[2].media_path == Path("s11.ds")
    assert by_run[2].action == "stage"
    assert by_run[1].match_method == "ambiguous"
    assert by_run[1].media_path is None
    assert "0 candidate session(s)" in by_run[1].note
    assert "Candidates: none" in by_run[1].note


def test_match_media_to_behavior_end_to_end_with_synthetic_media_and_tdms(tmp_path, monkeypatch):
    # No derivatives/meg-tokens/sub-*/beh/ anywhere -- Stage 0 must work from
    # raw media + tdms/ alone.
    media_root = tmp_path / "media"
    behavior_root = tmp_path / "tdms"
    _make_session(media_root, "H02", "20180213", "01", duration=135.0, start_time="13-Feb-2018 09:00")
    _make_session(media_root, "H02", "20180213", "02", duration=315.0, start_time="13-Feb-2018 09:10")
    (media_root / "NOISE_noise_20180213_01.ds").mkdir()
    (media_root / "H02_DDM-tthiery_20180213_HEADSHAPE.eeg").write_text("")

    tdms_dir = behavior_root / "H02"
    tdms_dir.mkdir(parents=True)
    (tdms_dir / "H2RT1_180213.tdms").write_text("")
    (tdms_dir / "H2Slow1_180213.tdms").write_text("")

    fake_by_path = {
        str(tdms_dir / "H2RT1_180213.tdms"): _FakeTdmsFile(_fake_tdms_groups([50])),
        str(tdms_dir / "H2Slow1_180213.tdms"): _FakeTdmsFile(_fake_tdms_groups([150])),
    }
    monkeypatch.setattr("meg_tokens.behavior.tdms.TdmsFile", lambda path: fake_by_path[str(path)])
    monkeypatch.setattr("meg_tokens.meg.raw_staging.count_start_pulses", lambda path, subject: 1)

    results = match_media_to_behavior(media_root, behavior_root, "H02")

    kinds = {(r.kind, r.condition, r.run) for r in results}
    assert ("run", "RT", 1) in kinds
    assert ("run", "Slow", 1) in kinds
    assert ("noise", None, None) in kinds
    assert ("headshape", None, None) in kinds
    assert all(r.action == "stage" for r in results)


def test_match_media_to_behavior_raises_without_any_tdms_runs(tmp_path):
    media_root = tmp_path / "media"
    _make_session(media_root, "H02", "20180213", "01", duration=315.0, start_time="13-Feb-2018 09:00")
    with pytest.raises(FileNotFoundError):
        match_media_to_behavior(media_root, tmp_path / "tdms", "H02")


def test_count_start_pulses_real_media_integration():
    media_root = os.environ.get("MEG_TOKENS_MEDIA_ROOT", "/media/karim/Hamza/DDM-tthiery")
    real_path = os.path.join(media_root, "H02_DDM-tthiery_20180213_03.ds")
    if not os.path.exists(real_path):
        pytest.skip(f"Real raw MEG media not available at {real_path}")

    from meg_tokens.meg.raw_staging import count_start_pulses

    count = count_start_pulses(real_path, "H02")
    assert count > 0
