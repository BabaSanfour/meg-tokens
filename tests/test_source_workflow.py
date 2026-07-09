import pytest

from meg_tokens.core import ProjectConfig, SourceConfig
from meg_tokens.workflows.sources import reconstruct_sources


def test_source_workflow_requires_declared_noise_root(tmp_path):
    project = ProjectConfig(bids_root=tmp_path)

    with pytest.raises(ValueError, match="noise_dir"):
        reconstruct_sources(
            project,
            subjects=["H01"],
            settings=SourceConfig(stages=("cov",)),
        )
