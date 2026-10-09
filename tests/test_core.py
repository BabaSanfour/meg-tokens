import pytest

from meg_tokens.core import ProjectConfig


def test_project_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown project configuration fields"):
        ProjectConfig.from_mapping({"data_root": "/data", "unexpected": "value"})


def test_project_config_derives_roots_from_data_root(tmp_path):
    config = ProjectConfig(data_root=tmp_path / "meg-tokens")

    assert config.raw_meg_root == tmp_path / "meg-tokens" / "raw"
    assert config.behavior_root == tmp_path / "meg-tokens" / "tdms"
    assert config.bids_root == tmp_path / "meg-tokens" / "BIDS"
    assert config.subjects_dir == tmp_path / "meg-tokens" / "IRM"
