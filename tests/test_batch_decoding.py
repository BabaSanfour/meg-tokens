import numpy as np
import pandas as pd

from meg_tokens.core import DecodingConfig, ProjectConfig
from meg_tokens.io import load_array, save_array, save_table, sidecar_path
from meg_tokens.workflows.decoding import (
    _load_feature_array,
    build_decoding_dataset,
    find_feature_arrays,
    run_batch_decoding,
)
from meg_tokens.workflows.erp import erp_derivative_path
from meg_tokens.workflows.decoding import run_decoding


LABELS = ["Pair-lh", "Pair-rh"]
TIMES = [-0.1, 0.0, 0.1]


def _write_erp(root, subject, run, condition, data, class_values=None):
    path = erp_derivative_path(
        root,
        subject=subject,
        run=run,
        condition=condition,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )
    trial_path = erp_derivative_path(
        root,
        subject=subject,
        run=run,
        condition=condition,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erptrials",
        extension=".tsv",
    )
    data = np.asarray(data, dtype=float)
    if class_values is None:
        class_values = [1] * data.shape[0]
    trials = pd.DataFrame({
        "nTrialIndex": list(range(1, data.shape[0] + 1)),
        "sTrialClass": class_values,
        "manifest_trial": list(range(1, data.shape[0] + 1)),
    })
    save_table(trial_path, trials, metadata={"stage": "erp_parcellation"})
    save_array(
        path,
        data,
        dims=("trial", "label", "time"),
        coords={"trial": list(range(1, data.shape[0] + 1)), "label": LABELS, "time_sec": TIMES},
        metadata={
            "stage": "erp_parcellation",
            "subject": subject,
            "condition": condition,
            "run": run,
            "alignment": "go",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
            "trial_table": str(trial_path),
        },
    )
    return path


def _write_decoding_inputs(root):
    base = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    _write_erp(root, "H01", "Fast1", "Fast", base + 10)
    _write_erp(root, "H01", "Slow1", "Slow", base + 20)
    _write_erp(root, "H02", "Fast1", "Fast", base + 30)
    _write_erp(root, "H02", "Slow1", "Slow", base + 40)


def test_find_feature_arrays_and_build_dataset_selects_labels(tmp_path):
    _write_decoding_inputs(tmp_path)

    paths = find_feature_arrays(
        tmp_path,
        "H1",
        "Fast",
        feature_source="erp",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
    )
    dataset = build_decoding_dataset(
        tmp_path,
        conditions=("Fast", "Slow"),
        feature_source="erp",
        subjects=["H01", "H02"],
        labels=["Pair-lh"],
    )

    assert len(paths) == 1
    assert dataset["X"].shape == (8, 1, 3)
    assert dataset["y"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert dataset["groups"].tolist() == ["H01", "H01", "H01", "H01", "H02", "H02", "H02", "H02"]
    assert dataset["feature_names"] == ["Pair-lh"]


def test_build_dataset_can_label_from_trial_metadata(tmp_path):
    data = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    _write_erp(tmp_path, "H01", "Fast1", "Fast", data, class_values=[1, 2, 1, 2])
    _write_erp(tmp_path, "H02", "Fast1", "Fast", data + 100, class_values=[1, 2, 1, 2])

    dataset = build_decoding_dataset(
        tmp_path,
        conditions=("Easy", "Ambiguous"),
        input_conditions=("Fast",),
        feature_source="erp",
        subjects=["H01", "H02"],
        class_column="sTrialClass",
        class_values=("1", "2"),
        labels=["Pair-lh"],
    )

    assert dataset["X"].shape == (8, 1, 3)
    assert dataset["y"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_build_dataset_lateralizes_label_pairs(tmp_path):
    data = np.zeros((2, 2, 3), dtype=float)
    data[:, 0, :] = 5.0
    data[:, 1, :] = 2.0
    _write_erp(tmp_path, "H01", "Fast1", "Fast", data)
    _write_erp(tmp_path, "H01", "Slow1", "Slow", data + 1)

    dataset = build_decoding_dataset(
        tmp_path,
        conditions=("Fast", "Slow"),
        feature_source="erp",
        subjects=["H01"],
        lateralize=True,
    )

    assert dataset["X"].shape == (4, 1, 3)
    assert np.allclose(dataset["X"][:2], 3.0)
    assert dataset["feature_names"] == ["Pair"]


def test_load_feature_array_uses_source_coordinate_names(tmp_path):
    path = erp_derivative_path(
        tmp_path,
        subject="H01",
        run="Fast1",
        condition="Fast",
        align_to="go",
        source_method="dSPM",
        parc="all-source",
        suffix="erp",
        extension=".npy",
    )
    save_array(
        path,
        np.zeros((2, 2, 3)),
        dims=("trial", "source", "time"),
        coords={"trial": [1, 2], "source": ["lh-0", "rh-1"], "time_sec": TIMES},
        metadata={"stage": "erp_parcellation", "feature_space": "all_source"},
    )

    X, times, feature_names, _ = _load_feature_array(path)

    assert X.shape == (2, 2, 3)
    assert times.tolist() == TIMES
    assert feature_names == ["lh-0", "rh-1"]


def test_run_batch_decoding_writes_derivatives_and_preserves_invalid_times(tmp_path, monkeypatch):
    _write_decoding_inputs(tmp_path)
    slow_path = erp_derivative_path(
        tmp_path,
        subject="H02",
        run="Slow1",
        condition="Slow",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )
    loaded = load_array(slow_path, require_sidecar=True)
    data = loaded.data
    data[:, :, -1] = np.nan
    save_array(
        slow_path,
        data,
        dims=loaded.metadata["dims"],
        coords=loaded.metadata["coords"],
        metadata=loaded.metadata["metadata"],
    )
    captured = {}

    def fixed_decoding(X, y, groups, balance, n_jobs):
        captured["shape"] = X.shape
        return np.array([[0.50, 0.60], [0.55, 0.65]])

    monkeypatch.setattr(
        "meg_tokens.workflows.decoding.compute_time_resolved_decoding",
        fixed_decoding,
    )

    outputs = run_batch_decoding(
        feature_dir=str(tmp_path),
        output_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        feature_source="erp",
        subjects=["H01", "H02"],
        labels=["Pair-lh"],
        n_jobs=1,
    )

    scores = load_array(outputs["score"], require_sidecar=True)
    splits = load_array(outputs["splits"], require_sidecar=True)

    assert captured["shape"] == (8, 1, 2)
    assert np.allclose(scores.data[:2], [0.525, 0.625])
    assert np.isnan(scores.data[2])
    assert splits.data.shape == (2, 3)
    assert sidecar_path(outputs["plot"]).is_file()
    assert scores.metadata["metadata"]["valid_time_mask"] == [True, True, False]


def test_decoding_workflow_declares_features_and_outputs(tmp_path, monkeypatch):
    _write_decoding_inputs(tmp_path)
    monkeypatch.setattr(
        "meg_tokens.workflows.decoding.compute_time_resolved_decoding",
        lambda X, y, groups, balance, n_jobs: np.array([[0.5, 0.6, 0.7]]),
    )

    result = run_decoding(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H01", "H02"],
        settings=DecodingConfig(labels=("Pair-lh",), n_jobs=1),
    )

    assert result.stage == "decoding"
    assert len(result.inputs) == 4
    assert any(path.name.endswith("_decoding.npy") for path in result.outputs)
    assert any(path.name.endswith("_decodingsplits.npy") for path in result.outputs)
    assert any(path.name.endswith("_decodingplot.png") for path in result.outputs)
