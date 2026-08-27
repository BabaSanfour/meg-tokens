import json

import pytest

from meg_tokens.meg.meg_bids import associated_emptyroom, raw_meg_path


def _stage(bids_root, subject, condition, run, associated):
    path = raw_meg_path(bids_root, subject, condition, run)
    path.mkdir(parents=True)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"AssociatedEmptyRoom": associated} if associated else {}),
        encoding="utf-8",
    )
    return path


def _stage_noise(bids_root, date):
    path = (
        bids_root
        / "sub-emptyroom"
        / f"ses-{date}"
        / "meg"
        / f"sub-emptyroom_ses-{date}_task-noise_meg.ds"
    )
    path.mkdir(parents=True)
    return path


def test_associated_emptyroom_reads_what_stage_0_recorded(tmp_path):
    noise = _stage_noise(tmp_path, "20180213")
    associated = "/sub-emptyroom/ses-20180213/meg/sub-emptyroom_ses-20180213_task-noise_meg.ds"
    _stage(tmp_path, "H01", "Fast", 1, associated)
    _stage(tmp_path, "H01", "Slow", 1, associated)

    assert associated_emptyroom(tmp_path, "H01") == noise


def test_associated_emptyroom_refuses_runs_that_disagree(tmp_path):
    for date in ("20180213", "20180214"):
        _stage_noise(tmp_path, date)
    _stage(tmp_path, "H01", "Fast", 1, "/sub-emptyroom/ses-20180213/meg/a_meg.ds")
    _stage(tmp_path, "H01", "Slow", 1, "/sub-emptyroom/ses-20180214/meg/b_meg.ds")

    with pytest.raises(ValueError, match="2 different empty-room sessions"):
        associated_emptyroom(tmp_path, "H01")


def test_associated_emptyroom_reports_a_subject_with_no_noise_day(tmp_path):
    """H05 is the real case: no empty-room session was acquired that day."""
    _stage(tmp_path, "H05", "Fast", 1, None)

    with pytest.raises(FileNotFoundError, match="no AssociatedEmptyRoom"):
        associated_emptyroom(tmp_path, "H05")


def test_associated_emptyroom_reports_an_unstaged_noise_session(tmp_path):
    _stage(tmp_path, "H01", "Fast", 1, "/sub-emptyroom/ses-20180213/meg/a_meg.ds")

    with pytest.raises(FileNotFoundError, match="is not staged"):
        associated_emptyroom(tmp_path, "H01")


def test_epoched_meg_runs_walks_the_grid_in_protocol_order(tmp_path):
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.meg.meg_bids import epoched_meg_runs

    layout = DerivativeLayout(tmp_path)
    for condition, run in (("Slow", 2), ("Fast", 1)):
        path = layout.epochs(
            subject="H01", run=str(run), condition=condition, alignment="go"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    # A different alignment is a different derivative and must not be picked up.
    other = layout.epochs(subject="H01", run="3", condition="Fast", alignment="enter")
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("")

    found = [(spec.condition, spec.run) for spec, _ in epoched_meg_runs(layout, "H01", alignment="go")]
    assert found == [("Fast", "1"), ("Slow", "2")]
