from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from meg_tokens.core import ProjectConfig
from meg_tokens.meg.raw_staging import MatchResult
from meg_tokens.workflows.raw_staging import (
    MANIFEST_COLUMNS,
    _matching_tdms_files,
    apply_raw_staging,
    plan_raw_staging,
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
        "media_path": "s1.ds",
        "match_method": "fast_path",
        "meg_start_pulse_count": 10,
        "behavior_trial_count": 10,
        "count_agreement": "exact",
        "action": "stage",
        "note": "",
    }
    row.update(overrides)
    return row


def test_plan_raw_staging_writes_manifest_and_settings_summary(tmp_path, monkeypatch):
    def fake_match(media_root, behavior_root, subject, *, settings):
        return [
            MatchResult(
                subject=subject, kind="run", condition="Slow", run=1, date="20180213",
                media_path=Path("s1.ds"), match_method="fast_path", meg_start_pulse_count=10,
                behavior_trial_count=10, count_agreement="exact", action="stage",
            ),
            MatchResult(
                subject=subject, kind="run", condition="Slow", run=2, date="20180213",
                media_path=None, match_method="unmatched", meg_start_pulse_count=None,
                behavior_trial_count=12, count_agreement="not_applicable", action="review",
                note="No raw session matched this run.",
            ),
        ]

    monkeypatch.setattr("meg_tokens.workflows.raw_staging.match_media_to_behavior", fake_match)
    (tmp_path / "raw").mkdir()

    result = plan_raw_staging(_project(tmp_path), subjects=["H02"])

    manifest_path = result.outputs[0]
    assert manifest_path.exists()
    table = pd.read_csv(manifest_path, sep="\t")
    assert list(table.columns) == MANIFEST_COLUMNS
    assert len(table) == 2
    assert result.settings["summary_by_subject"]["H02"] == {"stage": 1, "review": 1}
    assert len(result.settings["review_rows"]) == 1
    assert result.settings["review_rows"][0]["note"] == "No raw session matched this run."


def test_plan_raw_staging_replan_overwrites_previous_manifest(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_match(media_root, behavior_root, subject, *, settings):
        calls["n"] += 1
        run = calls["n"]
        return [
            MatchResult(
                subject=subject, kind="run", condition="Slow", run=run, date="20180213",
                media_path=Path(f"s{run}.ds"), match_method="fast_path", meg_start_pulse_count=10,
                behavior_trial_count=10, count_agreement="exact", action="stage",
            )
        ]

    monkeypatch.setattr("meg_tokens.workflows.raw_staging.match_media_to_behavior", fake_match)
    project = _project(tmp_path)

    first = plan_raw_staging(project, subjects=["H02"])
    second = plan_raw_staging(project, subjects=["H02"])

    assert first.outputs[0] == second.outputs[0]
    table = pd.read_csv(second.outputs[0], sep="\t")
    assert len(table) == 1
    assert table.iloc[0]["run"] == 2


def _patch_writers(monkeypatch):
    written = {"meg": [], "noise": [], "headshape": [], "tdms_calls": []}

    def fake_write_meg_bids(
        media_path, *, subject, condition, run, bids_root, task, empty_room_bids_path, overwrite
    ):
        written["meg"].append((subject, condition, run))
        return SimpleNamespace(fpath=Path(f"{subject}-{condition}-{run}_meg.ds"))

    def fake_write_emptyroom_bids(media_path, *, date, bids_root, overwrite):
        written["noise"].append(date)
        return SimpleNamespace(fpath=Path("emptyroom_meg.ds"))

    def fake_write_headshape(project, subject, eeg_path):
        written["headshape"].append(subject)
        return Path(f"{subject}_headshape.pos")

    monkeypatch.setattr("meg_tokens.workflows.raw_staging.write_meg_bids", fake_write_meg_bids)
    monkeypatch.setattr(
        "meg_tokens.workflows.raw_staging.write_emptyroom_bids", fake_write_emptyroom_bids
    )
    monkeypatch.setattr("meg_tokens.workflows.raw_staging._write_headshape", fake_write_headshape)
    monkeypatch.setattr(
        "meg_tokens.workflows.raw_staging._matching_tdms_files", lambda project, subject: []
    )
    return written


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(path, sep="\t", index=False)


def test_apply_raw_staging_only_materializes_action_stage_rows(tmp_path, monkeypatch):
    written = _patch_writers(monkeypatch)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest_path,
        [
            _manifest_row(run=1, action="stage"),
            _manifest_row(run=2, media_path="s2.ds", count_agreement="mismatch", action="review"),
            _manifest_row(kind="noise", condition="", run="", media_path="noise.ds", action="stage"),
            _manifest_row(kind="headshape", condition="", run="", media_path="hs.eeg", action="stage"),
        ],
    )

    result = apply_raw_staging(_project(tmp_path), manifest_path=manifest_path)

    assert written["meg"] == [("H02", "Slow", 1)]  # run 2 (review) never applied
    assert written["noise"] == ["20180213"]
    assert written["headshape"] == ["H02"]
    assert len(result.outputs) == 3
    assert manifest_path in result.inputs


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


def test_apply_raw_staging_filters_by_requested_subjects(tmp_path, monkeypatch):
    written = _patch_writers(monkeypatch)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest_path,
        [
            _manifest_row(subject="H02", run=1),
            _manifest_row(subject="H03", run=1, media_path="s1.ds"),
        ],
    )

    apply_raw_staging(_project(tmp_path), manifest_path=manifest_path, subjects=["H02"])

    assert written["meg"] == [("H02", "Slow", 1)]


def test_apply_raw_staging_raises_without_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        apply_raw_staging(_project(tmp_path), manifest_path=tmp_path / "missing.tsv")


def test_matching_tdms_files_finds_h01s_unpadded_h1_folder(tmp_path):
    # Real dataset regression: H01's TDMS folder on disk is literally "H1",
    # not "H01" -- a naive `behavior_root / "H01"` glob silently returns
    # nothing (found via a real apply-raw-staging run producing zero beh.tsv
    # files for H01). _matching_tdms_files must resolve the same way
    # workflows.behavior._subject_input_dir already does for ingestion.
    tdms_dir = tmp_path / "tdms" / "H1"
    tdms_dir.mkdir(parents=True)
    (tdms_dir / "H1Slow1_180131.tdms").write_text("")
    (tdms_dir / "temp_180131.tdms").write_text("")  # must be skipped

    found = _matching_tdms_files(_project(tmp_path), "H01")

    assert [path.name for path in found] == ["H1Slow1_180131.tdms"]


def test_matching_tdms_files_returns_empty_for_missing_subject_dir(tmp_path):
    (tmp_path / "tdms").mkdir()
    assert _matching_tdms_files(_project(tmp_path), "H99") == []
