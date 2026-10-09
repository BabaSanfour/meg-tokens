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
