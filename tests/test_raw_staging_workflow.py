from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from meg_tokens.core import ProjectConfig
from meg_tokens.workflows.raw_staging import (
    MANIFEST_COLUMNS,
    _matching_tdms_files,
    apply_raw_staging,
)


def _project(tmp_path):
    return ProjectConfig(data_root=tmp_path)


def _manifest_row(**overrides):
    row = {
        "subject": "H02",
        "kind": "run",
        "condition": "Slow",
        "run": 1,
        "date": "20180213",
        "source_path": "s1.ds",
        "match_method": "fast_path",
        "meg_start_pulse_count": 10,
        "behavior_trial_count": 10,
        "count_agreement": "exact",
        "action": "stage",
        "note": "",
    }
    row.update(overrides)
    return row


def _patch_writers(monkeypatch):
    written = {"meg": [], "noise": [], "headshape": [], "anat": [], "tdms_calls": []}

    def fake_write_meg_bids(
        source_path, *, subject, condition, run, bids_root, task, empty_room_bids_path, overwrite
    ):
        written["meg"].append((subject, condition, run))
        return SimpleNamespace(fpath=Path(f"{subject}-{condition}-{run}_meg.ds"))

    def fake_write_emptyroom_bids(source_path, *, date, bids_root, overwrite):
        written["noise"].append(date)
        return SimpleNamespace(fpath=Path("emptyroom_meg.ds"))

    def fake_write_headshape(project, subject, eeg_path):
        written["headshape"].append(subject)
        return Path(f"{subject}_headshape.eeg")

    def fake_write_anat_bids(subject, t1_path, *, bids_root, overwrite):
        written["anat"].append((subject, str(t1_path)))
        return SimpleNamespace(fpath=Path(f"{subject}_T1w.nii.gz"))

    monkeypatch.setattr("meg_tokens.workflows.raw_staging.write_meg_bids", fake_write_meg_bids)
    monkeypatch.setattr(
        "meg_tokens.workflows.raw_staging.write_emptyroom_bids", fake_write_emptyroom_bids
    )
    monkeypatch.setattr("meg_tokens.workflows.raw_staging._write_headshape", fake_write_headshape)
    monkeypatch.setattr("meg_tokens.workflows.raw_staging.write_anat_bids", fake_write_anat_bids)
    monkeypatch.setattr(
        "meg_tokens.workflows.raw_staging._matching_tdms_files", lambda project, subject: []
    )
    return written


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(path, sep="\t", index=False)


def test_apply_raw_staging_respects_hand_edited_action_column(tmp_path, monkeypatch):
    written = _patch_writers(monkeypatch)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest_path,
        [_manifest_row(run=1, count_agreement="mismatch", action="review")],
    )

    # Nothing applied while still flagged for review.
    apply_raw_staging(_project(tmp_path), manifest_path=manifest_path)
    assert written["meg"] == []

    # A human reviews the flagged row and approves it by hand-editing the
    # manifest -- re-applying from that same file must now stage it. This
    # is the whole point of apply reading the manifest as saved instead of
    # recomputing the plan.
    table = pd.read_csv(manifest_path, sep="\t")
    table.loc[0, "action"] = "stage"
    table.to_csv(manifest_path, sep="\t", index=False)

    apply_raw_staging(_project(tmp_path), manifest_path=manifest_path)
    assert written["meg"] == [("H02", "Slow", 1)]


def _subject_logs(tmp_path, *names) -> Path:
    subject_dir = tmp_path / "tdms" / "H01"
    subject_dir.mkdir(parents=True)
    for name in names:
        (subject_dir / name).write_text("")
    return subject_dir


# The two guards below used to sit downstream in `behavior ingest`. Stage 1
# now reads the staged BIDS tables, so this is the only gate between the raw
# logs and the dataset: a file dropped here never reaches any analysis.


def test_matching_tdms_files_refuses_to_guess_at_a_non_canonical_name(tmp_path):
    """A misnamed real run and a scratch export look identical to a glob, and
    only one of them is safe to skip."""
    _subject_logs(tmp_path, "H1Slow1_180131.tdms", "temp_180131.tdms")

    with pytest.raises(ValueError, match="temp_180131.tdms"):
        _matching_tdms_files(_project(tmp_path), "H01")


def test_matching_tdms_files_rejects_two_logs_claiming_one_bids_path(tmp_path):
    """Both resolve to the same staged file, so the second would overwrite the
    first and the loss would be invisible afterwards."""
    _subject_logs(tmp_path, "H1Slow1_180131.tdms", "H1Slow1_190101.tdms")

    with pytest.raises(ValueError, match="Duplicate TDMS run"):
        _matching_tdms_files(_project(tmp_path), "H01")
