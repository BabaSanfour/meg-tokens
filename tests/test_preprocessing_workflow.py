import pytest

from meg_tokens.core import ProjectConfig, RunSpec
from meg_tokens.meg.meg_bids import raw_meg_path
from meg_tokens.workflows import preprocessing as preprocessing_workflow
from meg_tokens.workflows.preprocessing import preprocess_run


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
        seen.append(str(path))
        raise _Resolved

    monkeypatch.setattr(preprocessing_workflow, "read_staged_raw", _capture)
    with pytest.raises(_Resolved):
        preprocess_run(project, RunSpec(subject="H01", run="Fast1"))

    assert seen == [str(staged)]
