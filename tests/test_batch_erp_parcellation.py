import numpy as np
import pandas as pd
import pytest
import mne

from meg_tokens.core import ERPConfig, ProjectConfig
from meg_tokens.io import derivative_path, load_array, save_table
from meg_tokens.meg.sources import source_derivative_path
from meg_tokens.workflows.erp import (
    erp_derivative_path,
    extract_parcellated_erp_from_manifest,
    run_erp_parcellation_pipeline,
)
from meg_tokens.workflows.erp import extract_erp_features


def _write_stage_inputs(root, n_trials=2):
    manifest_path = source_derivative_path(
        root,
        "H1",
        suffix="stcmanifest",
        extension=".tsv",
        run_id="Slow1",
        description="go-dSPM",
    )
    vertices = [np.array([0]), np.array([1])]
    rows = []
    sfreq = 100.0

    for trial in range(1, n_trials + 1):
        data = np.vstack([
            np.full(500, float(trial)),
            np.full(500, float(trial + 10)),
        ])
        stc = mne.SourceEstimate(data, vertices=vertices, tmin=-1.0, tstep=1.0 / sfreq)
        base = source_derivative_path(
            root,
            "H1",
            suffix=f"trial{trial:04d}stc",
            extension="",
            run_id="Slow1",
            description="go-dSPM",
        )
        base.parent.mkdir(parents=True, exist_ok=True)
        stc.save(str(base), ftype="stc", overwrite=True)
        rows.append({
            "trial": trial,
            "stc_base": str(base),
            "subject": "H01",
            "run": "1",
            "condition": "Slow",
            "alignment": "go",
            "method": "dSPM",
        })

    save_table(
        manifest_path,
        pd.DataFrame(rows),
        metadata={"stage": "source_reconstruction", "kind": "source_estimate_manifest"},
    )

    behavior_path = derivative_path(
        root,
        subject="H01",
        datatype="beh",
        task="tokens",
        run="1",
        description="slow",
        suffix="beh",
        extension=".tsv",
    )
    behavior = pd.DataFrame({
        "subject": ["H01"] * n_trials,
        "condition": ["Slow"] * n_trials,
        "run": [1] * n_trials,
        "source_file": ["H1Slow1_180131.tdms"] * n_trials,
        "nTrialIndex": list(range(1, n_trials + 1)),
        "sTrialClass": [1] * n_trials,
        "nInitialTime": [0] * n_trials,
        "nChoiceMade": [1] * n_trials,
        "nCorrectChoice": [1] * n_trials,
        "tGO": [1000.0] * n_trials,
        "tEnterTarget": [2500.0, 1500.0][:n_trials],
        "tTrialEnd": [3000.0] * n_trials,
        "sTokenDirs": ["0"] * n_trials,
        "tTime": ["[]"] * n_trials,
        "nProb": ["[]"] * n_trials,
        "rawRT": [1500.0, 500.0][:n_trials],
        "isCorrect": [True] * n_trials,
    })
    save_table(
        behavior_path,
        behavior,
        metadata={"stage": "behavior_parsing", "subject": "H01", "condition": "Slow", "run": "1"},
    )
    return manifest_path, behavior_path


def test_erp_derivative_path_uses_bids_contract(tmp_path):
    path = erp_derivative_path(
        tmp_path,
        subject="H1",
        run="Slow1",
        condition=None,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )

    assert path.name == "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erp.npy"


def test_extract_parcellated_erp_from_manifest_writes_array_and_trials(tmp_path, monkeypatch):
    manifest, behavior = _write_stage_inputs(tmp_path)

    def simple_parcellation(stc, **kwargs):
        return ["Label_1-lh", "Label_2-rh"], stc.data

    monkeypatch.setattr(
        "meg_tokens.workflows.erp.parcellate_source_estimates",
        simple_parcellation,
    )

    outputs = extract_parcellated_erp_from_manifest(
        manifest,
        behavior,
        subjects_dir=str(tmp_path / "subjects"),
        out_dir=str(tmp_path),
        parc="HCPMMP1",
        max_duration_samples=300,
        min_rt_ms=100.0,
    )

    loaded = load_array(outputs["data"], require_sidecar=True)
    trials = pd.read_csv(outputs["trials"], sep="\t")

    assert outputs["data"].name == "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-HCPMMP1_erp.npy"
    assert loaded.data.shape == (2, 2, 300)
    assert loaded.metadata["dims"] == ["trial", "label", "time"]
    assert loaded.metadata["coords"]["trial"] == [1, 2]
    assert loaded.metadata["coords"]["label"] == ["Label_1-lh", "Label_2-rh"]
    assert np.all(np.isnan(loaded.data[0, :, 220:]))
    assert np.all(np.isnan(loaded.data[1, :, 120:]))
    assert loaded.metadata["metadata"]["trial_table"] == str(outputs["trials"])
    assert trials["manifest_trial"].tolist() == [1, 2]
    assert trials["output_trial_index"].tolist() == [1, 2]


def test_extract_parcellated_erp_rejects_trial_count_mismatch(tmp_path, monkeypatch):
    manifest, behavior = _write_stage_inputs(tmp_path)
    table = pd.read_csv(behavior, sep="\t").iloc[:1]
    save_table(behavior, table, metadata={"stage": "behavior_parsing"})

    monkeypatch.setattr(
        "meg_tokens.workflows.erp.parcellate_source_estimates",
        lambda stc, **kwargs: (["Label"], stc.data[:1]),
    )

    with pytest.raises(ValueError, match="Trial count mismatch"):
        extract_parcellated_erp_from_manifest(
            manifest,
            behavior,
            subjects_dir=str(tmp_path / "subjects"),
            out_dir=str(tmp_path),
        )


def test_extract_all_source_erp_from_manifest_writes_source_coordinates(tmp_path):
    manifest, behavior = _write_stage_inputs(tmp_path)

    outputs = extract_parcellated_erp_from_manifest(
        manifest,
        behavior,
        subjects_dir=None,
        out_dir=str(tmp_path),
        feature_space="all_source",
        max_duration_samples=300,
        min_rt_ms=100.0,
    )

    loaded = load_array(outputs["data"], require_sidecar=True)

    assert outputs["data"].name == "sub-H01_task-tokens_run-1_desc-slow-go-dSPM-all-source_erp.npy"
    assert loaded.data.shape == (2, 2, 300)
    assert loaded.metadata["dims"] == ["trial", "source", "time"]
    assert loaded.metadata["coords"]["source"] == ["lh-0", "rh-1"]
    meta = loaded.metadata["metadata"]
    assert meta["feature_space"] == "all_source"
    assert meta["parcellation"] == "all-source"
    assert meta["atlas"] is None
    assert meta["label_subject"] is None
    assert meta["source_vertices"] == [[0], [1]]


def test_extract_volume_erp_rejects_surface_only_manifest(tmp_path):
    manifest, behavior = _write_stage_inputs(tmp_path)

    with pytest.raises(ValueError, match="does not contain volume source groups"):
        extract_parcellated_erp_from_manifest(
            manifest,
            behavior,
            subjects_dir=None,
            out_dir=str(tmp_path),
            feature_space="volume",
            max_duration_samples=300,
            min_rt_ms=100.0,
        )


def test_run_erp_parcellation_pipeline_finds_stage_inputs(tmp_path, monkeypatch):
    _write_stage_inputs(tmp_path)

    monkeypatch.setattr(
        "meg_tokens.workflows.erp.parcellate_source_estimates",
        lambda stc, **kwargs: (["Label_1-lh", "Label_2-rh"], stc.data),
    )

    outputs = run_erp_parcellation_pipeline(
        ["H1"],
        source_dir=str(tmp_path),
        behavior_dir=str(tmp_path),
        subjects_dir=str(tmp_path / "subjects"),
        out_dir=str(tmp_path),
        run="Slow1",
        align_to="go",
        source_method="dSPM",
        max_duration_samples=300,
    )

    assert "H01" in outputs
    assert outputs["H01"]["data"].is_file()


def test_erp_workflow_declares_inputs_and_outputs(tmp_path):
    manifest, behavior = _write_stage_inputs(tmp_path)

    result = extract_erp_features(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H1"],
        settings=ERPConfig(
            run="Slow1",
            feature_space="all_source",
            max_duration_samples=300,
        ),
    )

    assert result.stage == "erp_features"
    assert result.inputs == (manifest, behavior)
    assert len(result.outputs) == 2
    assert all(path.is_file() for path in result.outputs)
