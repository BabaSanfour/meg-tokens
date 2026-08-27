import pytest

from meg_tokens.core import ProjectConfig, SourceConfig
from meg_tokens.io import DerivativeLayout
from meg_tokens.workflows.sources import _source_runs, reconstruct_sources


def test_source_workflow_names_the_subject_with_no_staged_meg(tmp_path):
    """The cov stage resolves noise through the run sidecars, so no staged
    MEG means no association to read -- reported as the staging gap it is."""
    project = ProjectConfig(data_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="No staged raw MEG for H01"):
        reconstruct_sources(
            project,
            subjects=["H01"],
            settings=SourceConfig(stages=("cov",)),
        )


def _stage_epochs(layout, subject, cells, alignment="go"):
    for condition, run in cells:
        path = layout.epochs(
            subject=subject, run=str(run), condition=condition, alignment=alignment
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def test_source_runs_discovers_every_epoched_run_of_a_subject(tmp_path):
    """Without a declared run the per-run stages cover what the subject has."""
    layout = DerivativeLayout(tmp_path)
    _stage_epochs(layout, "H01", (("Fast", 1), ("Fast", 2), ("Slow", 1)))

    found = [
        (spec.condition, spec.run)
        for spec, _ in _source_runs(layout, "H01", SourceConfig(stages=("fwd",)))
    ]

    assert found == [("Fast", "1"), ("Fast", "2"), ("Slow", "1")]


def test_source_runs_narrows_to_one_declared_run(tmp_path):
    layout = DerivativeLayout(tmp_path)
    _stage_epochs(layout, "H01", (("Fast", 1), ("Fast", 2), ("Slow", 1)))

    runs = _source_runs(layout, "H01", SourceConfig(stages=("fwd",), run="Fast2"))

    assert [(spec.condition, spec.run) for spec, _ in runs] == [("Fast", "2")]


def test_source_runs_names_the_missing_epochs(tmp_path):
    layout = DerivativeLayout(tmp_path)

    with pytest.raises(FileNotFoundError, match="No go-locked epochs for H01"):
        _source_runs(layout, "H01", SourceConfig(stages=("fwd",)))
