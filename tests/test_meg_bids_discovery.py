import json

import pytest

from meg_tokens.io import DerivativeLayout
from meg_tokens.meg.meg_bids import epoched_meg_runs, nearest_emptyroom


def _scans(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename\tacq_time", *(f"{name}\t{time}" for name, time in rows)]
    # mne-bids writes a byte-order mark.
    path.write_text("\ufeff" + "\n".join(lines) + "\n", encoding="utf-8")


def _stage_noise(bids_root, session, time):
    name = f"sub-emptyroom_ses-{session}_task-noise_meg.ds"
    _scans(bids_root / "sub-emptyroom" / f"ses-{session}" / f"sub-emptyroom_ses-{session}_scans.tsv", [(f"meg/{name}", time)])
    return bids_root / "sub-emptyroom" / f"ses-{session}" / "meg" / name


def test_nearest_emptyroom_is_the_recording_days_own(tmp_path):
    _stage_noise(tmp_path, "20180130", "2018-01-30T19:00:00.000000Z")
    noise = _stage_noise(tmp_path, "20180131", "2018-01-31T19:35:00.000000Z")
    # Runs late in the local evening cross midnight in UTC.
    _scans(tmp_path / "sub-H01" / "sub-H01_scans.tsv", [
        ("meg/sub-H01_task-tokens_acq-fast_run-1_meg.ds", "2018-01-31T23:19:00.000000Z"),
        ("meg/sub-H01_task-tokens_acq-fast_run-4_meg.ds", "2018-02-01T00:00:00.000000Z"),
    ])

    path, hours = nearest_emptyroom(tmp_path, "H01")
    assert path == noise
    assert hours == pytest.approx(3 + 44 / 60)


def test_nearest_emptyroom_takes_the_nearest_day_when_the_day_has_none(tmp_path):
    _stage_noise(tmp_path, "20180215", "2018-02-15T12:00:00.000000Z")
    noise = _stage_noise(tmp_path, "20180221", "2018-02-21T12:48:00.000000Z")
    _scans(tmp_path / "sub-H05" / "sub-H05_scans.tsv", [
        ("meg/sub-H05_task-tokens_acq-fast_run-1_meg.ds", "2018-02-20T13:55:00.000000Z"),
    ])

    path, hours = nearest_emptyroom(tmp_path, "H05")
    assert path == noise
    assert hours == pytest.approx(22 + 53 / 60)


def test_a_run_that_kept_no_epoch_is_not_discovered(tmp_path):
    layout = DerivativeLayout(tmp_path)
    for run, n_epochs in ((1, 60), (2, 0)):
        path = layout.epochs(subject="H10", run=str(run), condition="Fast", alignment="go", stream="dss")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        path.with_suffix(".json").write_text(json.dumps({"n_epochs": n_epochs}), encoding="utf-8")

    runs = epoched_meg_runs(layout, "H10", alignment="go", stream="dss")

    assert [(spec.condition, spec.run) for spec, _ in runs] == [("Fast", "1")]
