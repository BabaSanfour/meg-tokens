import mne
import numpy as np
import pandas as pd
import pytest

from meg_tokens.core import EpochingConfig, ProjectConfig, RunSpec, WorkflowResult
from meg_tokens.io import DerivativeLayout, save_table
from meg_tokens.meg.meg_bids import raw_meg_path
from meg_tokens.workflows import epoch_subjects
from meg_tokens.workflows import preprocessing as preprocessing_workflow
from meg_tokens.workflows.preprocessing import preprocess_run, preprocess_subjects


def _stage_recording(bids_root, subject, condition, run):
    """A CTF session is a directory, so staging one is a mkdir."""
    path = raw_meg_path(bids_root, subject, condition, run)
    path.mkdir(parents=True)
    return path


def test_preprocess_run_resolves_its_own_input_from_the_run_identity(tmp_path, monkeypatch):
    project = ProjectConfig(data_root=tmp_path)
    staged = _stage_recording(project.bids_root, "H01", "Fast", 1)

    class _Resolved(Exception):
        """Raised by the stub to stop the stage once the path is known."""

    seen = []

    def _capture(path, **kwargs):
        seen.append(path)
        raise _Resolved

    monkeypatch.setattr(preprocessing_workflow, "load_and_filter_raw", _capture)
    with pytest.raises(_Resolved):
        preprocess_run(project, RunSpec(subject="H01", run="Fast1"))

    assert seen == [str(staged)]


def test_preprocess_run_names_the_missing_prerequisite(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    project.bids_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="apply-raw-staging"):
        preprocess_run(project, RunSpec(subject="H01", run="Fast1"))


def test_preprocess_run_refuses_a_run_number_without_a_condition(tmp_path):
    """acq-fast run 1 and acq-slow run 1 are two different recordings."""
    project = ProjectConfig(data_root=tmp_path)

    with pytest.raises(ValueError, match="needs a condition"):
        preprocess_run(project, RunSpec(subject="H01", run="1"))


def test_preprocess_subjects_defaults_to_every_staged_subject_and_run(tmp_path, monkeypatch):
    project = ProjectConfig(data_root=tmp_path)
    for subject, condition, run in (
        ("H01", "Fast", 1),
        ("H01", "Slow", 2),
        ("H02", "RT", 1),
    ):
        _stage_recording(project.bids_root, subject, condition, run)

    dispatched = []

    def _record(project, run, *, settings):
        dispatched.append((run.subject, run.condition, run.run))
        return WorkflowResult(stage="meg_preprocessing", inputs=(), outputs=())

    monkeypatch.setattr(preprocessing_workflow, "preprocess_run", _record)
    result = preprocess_subjects(project)

    # Protocol order within a subject, not glob order.
    assert dispatched == [
        ("H01", "Fast", "1"),
        ("H01", "Slow", "2"),
        ("H02", "RT", "1"),
    ]
    assert result.settings["subjects"] == ["H01", "H02"]


def test_preprocess_subjects_rejects_a_subject_with_nothing_staged(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    _stage_recording(project.bids_root, "H01", "Fast", 1)

    with pytest.raises(FileNotFoundError, match="No staged raw MEG for H02"):
        preprocess_subjects(project, subjects=["H02"])


def test_epoch_subjects_consumes_staged_raw_and_behavior(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    layout = DerivativeLayout(project.bids_root)
    raw_path = layout.preprocessed_raw(
        subject="H1",
        run="1",
        condition="Fast",
        processing="filt",
    )
    raw_path.parent.mkdir(parents=True)
    info = mne.create_info(
        ["MEG001", "STI 014"],
        sfreq=1000.0,
        ch_types=["mag", "stim"],
    )
    data = np.zeros((2, 3000))
    data[1, 1000] = 524288
    data[1, 2000] = 524288
    mne.io.RawArray(data, info).save(raw_path, overwrite=True)

    behavior_path = layout.behavior(subject="H1", run="1", condition="Fast")
    behavior = pd.DataFrame(
        {
            "subject": ["H01", "H01"],
            "condition": ["Fast", "Fast"],
            "run": [1, 1],
            "source_file": ["H1Fast1_recording.tdms"] * 2,
            "nTrialIndex": [1, 2],
            "sTrialClass": [1, 2],
            "sTrialClassRaw": ["e", "a"],
            "trial_class_source": ["design", "design"],
            "trial_class_rule": ["recorded_label", "recorded_label"],
            "sp_design_correct": ["[0.6]", "[0.6]"],
            "nInitialTime": [0, 0],
            "nChoiceMade": [1, 2],
            "nCorrectChoice": [1, 2],
            "tGO": [1000, 2000],
            "tEnterCenter": [0, 0],
            "tExitCenter": [1400, 2400],
            "tEnterTarget": [1400, 2400],
            "tTrialEnd": [1600, 2600],
            "sTokenDirs": ["121", "212"],
            "nTokenNum": ["[1]", "[1]"],
            "nTokenDir": ["[1]", "[2]"],
            "tTime": ["[1100]", "[2100]"],
            "nProb": ["[0.6]", "[0.6]"],
            "token_log_rows": [1, 1],
            "token_log_short": [False, False],
            "nOutcome": [0, 0],
            "rawRT": [400, 400],
            "isCorrect": [True, True],
        }
    )
    save_table(behavior_path, behavior, metadata={"stage": "behavior_parsing"})

    result = epoch_subjects(
        project,
        subjects=["H1"],
        settings=EpochingConfig(tmin=-0.1, tmax=0.2, alignment="go"),
    )

    assert result.stage == "meg_epoching"
    assert layout.epochs(
        subject="H1",
        run="1",
        condition="Fast",
        alignment="go",
    ) in result.outputs
    epochs = mne.read_epochs(
        layout.epochs(
            subject="H1",
            run="1",
            condition="Fast",
            alignment="go",
        ),
        preload=False,
    )
    assert len(epochs) == 2
    assert epochs.metadata["nTrialIndex"].tolist() == [1, 2]
