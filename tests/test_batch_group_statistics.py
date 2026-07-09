import numpy as np
import pytest

from meg_tokens.core import (
    LateralizedStatisticsConfig,
    ProjectConfig,
    StatisticsConfig,
)
from meg_tokens.io import load_array, save_array, sidecar_path
from meg_tokens.workflows.erp import erp_derivative_path
from meg_tokens.workflows.statistics import (
    discover_subjects,
    find_erp_arrays,
    run_group_statistics_contrast,
    stats_derivative_path,
)
from meg_tokens.workflows.statistics import (
    run_group_statistics,
    run_lateralized_statistics,
    run_lateralized_statistics_test,
)


LABELS = ["Label_1-lh", "Label_2-rh"]
TIMES = [-0.1, 0.0, 0.1]


def _write_erp(root, subject, run, condition, data, labels=LABELS):
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
    save_array(
        path,
        np.asarray(data, dtype=float),
        dims=("trial", "label", "time"),
        coords={
            "trial": list(range(1, np.asarray(data).shape[0] + 1)),
            "label": labels,
            "time_sec": TIMES,
        },
        metadata={
            "stage": "erp_parcellation",
            "subject": subject,
            "condition": condition,
            "run": run,
            "alignment": "go",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
        },
    )
    return path


def _write_group_inputs(root):
    base = np.arange(12, dtype=float).reshape(2, 2, 3)
    _write_erp(root, "H01", "Fast1", "Fast", base + 5.0)
    _write_erp(root, "H01", "Slow1", "Slow", base + 1.0)
    _write_erp(root, "H02", "Fast1", "Fast", base + 7.0)
    _write_erp(root, "H02", "Slow1", "Slow", base + 2.0)


def test_stats_derivative_path_uses_group_subject(tmp_path):
    path = stats_derivative_path(
        tmp_path,
        conditions=("Fast", "Slow"),
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="tstat",
        extension=".npy",
    )

    assert path.name == "sub-group_task-tokens_desc-fast-vs-slow-go-dSPM-HCPMMP1_tstat.npy"


def test_find_and_discover_erp_arrays(tmp_path):
    _write_group_inputs(tmp_path)

    arrays = find_erp_arrays(
        tmp_path,
        "H1",
        "Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
    )
    subjects = discover_subjects(
        tmp_path,
        ("Fast", "Slow"),
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
    )

    assert len(arrays) == 1
    assert arrays[0].name.startswith("sub-H01")
    assert subjects == ["H01", "H02"]


def test_run_group_statistics_contrast_saves_derivatives(tmp_path, monkeypatch):
    _write_group_inputs(tmp_path)
    captured = {}

    def fixed_t_test(data, n_permutations, tail, n_jobs):
        captured["data"] = data.copy()
        return np.arange(data.shape[1]), np.linspace(0.01, 0.2, data.shape[1]), np.array([1.0, 2.0])

    monkeypatch.setattr(
        "meg_tokens.workflows.statistics.compute_permutation_t_test",
        fixed_t_test,
    )

    outputs = run_group_statistics_contrast(
        erp_dir=str(tmp_path),
        out_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        subjects_list=["H01", "H02"],
        n_permutations=2,
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        alpha=0.05,
    )

    tstat = load_array(outputs["tstat"], require_sidecar=True)
    pvalue = load_array(outputs["pvalue"], require_sidecar=True)
    contrast = load_array(outputs["contrast"], require_sidecar=True)
    h0 = load_array(outputs["h0"], require_sidecar=True)

    assert captured["data"].shape == (2, 6)
    assert np.allclose(captured["data"][0], 4.0)
    assert np.allclose(captured["data"][1], 5.0)
    assert tstat.data.shape == (2, 3)
    assert pvalue.data.shape == (2, 3)
    assert contrast.data.shape == (2, 2, 3)
    assert h0.data.tolist() == [1.0, 2.0]
    assert tstat.metadata["dims"] == ["label", "time"]
    assert tstat.metadata["coords"]["label"] == LABELS
    assert tstat.metadata["metadata"]["conditions"] == ["Fast", "Slow"]
    assert sidecar_path(outputs["windows"]).is_file()


def test_run_group_statistics_requires_two_subjects(tmp_path):
    _write_erp(tmp_path, "H01", "Fast1", "Fast", np.ones((2, 2, 3)))
    _write_erp(tmp_path, "H01", "Slow1", "Slow", np.ones((2, 2, 3)))

    with pytest.raises(ValueError, match="At least two subjects"):
        run_group_statistics_contrast(
            erp_dir=str(tmp_path),
            out_dir=str(tmp_path),
            conditions=("Fast", "Slow"),
            subjects_list=["H01"],
        )


def test_statistics_workflow_declares_group_inputs_and_outputs(tmp_path, monkeypatch):
    _write_group_inputs(tmp_path)
    monkeypatch.setattr(
        "meg_tokens.workflows.statistics.compute_permutation_t_test",
        lambda data, n_permutations, tail, n_jobs: (
            np.ones(data.shape[1]),
            np.full(data.shape[1], 0.5),
            np.ones(2),
        ),
    )

    result = run_group_statistics(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H01", "H02"],
        settings=StatisticsConfig(permutations=2),
    )

    assert result.stage == "group_statistics"
    assert len(result.inputs) == 4
    assert len(result.outputs) == 5


def test_lateralized_statistics_uses_homologous_label_difference(tmp_path, monkeypatch):
    labels = ["Motor-lh", "Motor-rh"]
    subject_1 = np.stack(
        [np.vstack([np.full(3, 5.0), np.full(3, 2.0)]) for _ in range(2)]
    )
    subject_2 = np.stack(
        [np.vstack([np.full(3, 7.0), np.full(3, 3.0)]) for _ in range(2)]
    )
    _write_erp(tmp_path, "H01", "Fast1", "Fast", subject_1, labels=labels)
    _write_erp(tmp_path, "H02", "Fast1", "Fast", subject_2, labels=labels)
    captured = {}

    def fixed_t_test(data, n_permutations, tail, n_jobs):
        captured["data"] = data.copy()
        return np.ones(data.shape[1]), np.full(data.shape[1], 0.01), np.ones(2)

    monkeypatch.setattr(
        "meg_tokens.workflows.statistics.compute_permutation_t_test",
        fixed_t_test,
    )
    outputs = run_lateralized_statistics_test(
        str(tmp_path),
        str(tmp_path),
        "Fast",
        subjects_list=["H01", "H02"],
        n_permutations=2,
    )

    lateralization = load_array(outputs["lateralization"], require_sidecar=True)
    assert captured["data"].shape == (2, 3)
    assert np.allclose(captured["data"][0], 3.0)
    assert np.allclose(captured["data"][1], 4.0)
    assert lateralization.data.shape == (2, 1, 3)
    assert lateralization.metadata["coords"]["label"] == ["Motor"]
    assert lateralization.metadata["metadata"]["contrast"] == "left-right"


def test_lateralized_statistics_workflow_declares_inputs(tmp_path, monkeypatch):
    labels = ["Motor-lh", "Motor-rh"]
    values = np.ones((2, 2, 3))
    _write_erp(tmp_path, "H01", "Fast1", "Fast", values, labels=labels)
    _write_erp(tmp_path, "H02", "Fast1", "Fast", values, labels=labels)
    monkeypatch.setattr(
        "meg_tokens.workflows.statistics.compute_permutation_t_test",
        lambda data, n_permutations, tail, n_jobs: (
            np.ones(data.shape[1]),
            np.ones(data.shape[1]),
            np.ones(2),
        ),
    )

    result = run_lateralized_statistics(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H01", "H02"],
        settings=LateralizedStatisticsConfig(condition="Fast", permutations=2),
    )

    assert result.stage == "lateralized_statistics"
    assert len(result.inputs) == 2
    assert len(result.outputs) == 5
