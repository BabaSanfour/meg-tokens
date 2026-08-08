from pathlib import Path

import pytest

from meg_tokens.core import (
    EpochingConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunSpec,
    SourceConfig,
    WorkflowResult,
    normalize_subject_id,
    parse_run_label,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("H1", "H01"), ("h01", "H01"), ("H032", "H32")],
)
def test_normalize_subject_id(value, expected):
    assert normalize_subject_id(value) == expected


def test_normalize_subject_id_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid subject ID"):
        normalize_subject_id("subject-1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Slow01", ("1", "Slow")),
        ("RT2", ("2", "Rt")),
        ("run-03", ("03", None)),
        ("custom-run", ("custom-run", None)),
    ],
)
def test_parse_run_label_preserves_compatibility(value, expected):
    assert parse_run_label(value) == expected


def test_run_spec_canonicalizes_identity():
    run = RunSpec(subject="h1", run="Slow01", alignment="GO")
    assert run == RunSpec(subject="H01", run="1", condition="Slow", alignment="go")


def test_run_spec_rejects_unknown_alignment():
    with pytest.raises(ValueError, match="Unknown alignment"):
        RunSpec(subject="H01", run="1", alignment="commitment")


def test_project_config_loads_relative_toml_paths(tmp_path):
    config_path = tmp_path / "tokens.toml"
    config_path.write_text(
        "\n".join(
            [
                "[project]",
                'data_root = "meg-tokens"',
                'subjects_dir = "freesurfer"',
                'task = "tokens"',
                'subject_exclusions = ["H1", "h02"]',
            ]
        ),
        encoding="utf-8",
    )

    config = ProjectConfig.from_toml(config_path)

    assert config.data_root == tmp_path / "meg-tokens"
    assert config.subjects_dir == tmp_path / "freesurfer"
    assert config.subject_exclusions == ("H01", "H02")


def test_project_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown project configuration fields"):
        ProjectConfig.from_mapping({"data_root": "/data", "unexpected": "value"})


def test_project_config_derives_roots_from_data_root(tmp_path):
    config = ProjectConfig(data_root=tmp_path / "meg-tokens")

    assert config.raw_meg_root == tmp_path / "meg-tokens" / "raw"
    assert config.behavior_root == tmp_path / "meg-tokens" / "tdms"
    assert config.bids_root == tmp_path / "meg-tokens" / "BIDS"
    assert config.subjects_dir == tmp_path / "meg-tokens" / "IRM"


def test_project_config_requires_data_root():
    with pytest.raises(TypeError):
        ProjectConfig()

    with pytest.raises(ValueError, match="requires 'data_root'"):
        ProjectConfig.from_mapping({"task": "tokens"})


def test_workflow_result_normalizes_paths():
    result = WorkflowResult(
        stage="epochs",
        inputs=("raw.fif",),
        outputs=("epochs.fif",),
        settings={"alignment": "go"},
    )
    assert result.inputs == (Path("raw.fif"),)
    assert result.outputs == (Path("epochs.fif"),)


def test_preprocessing_and_epoching_settings_validate_ranges():
    assert PreprocessingConfig().high_pass == 0.5
    assert EpochingConfig(alignment="ENTER").alignment == "enter"
    with pytest.raises(ValueError, match="low_pass"):
        PreprocessingConfig(high_pass=10.0, low_pass=5.0)
    with pytest.raises(ValueError, match="tmax"):
        EpochingConfig(tmin=1.0, tmax=0.0)


def test_source_settings_require_explicit_run_for_run_level_stages():
    assert SourceConfig(stages=("cov",)).run is None
    with pytest.raises(ValueError, match="run is required"):
        SourceConfig(stages=("fwd",))
    with pytest.raises(ValueError, match="Unknown source stages"):
        SourceConfig(stages=("unknown",))
