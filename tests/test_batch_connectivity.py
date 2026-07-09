import numpy as np
import pandas as pd

from meg_tokens.core import ConnectivityConfig, ProjectConfig
from meg_tokens.io import load_array, save_array, save_table
from meg_tokens.workflows.connectivity import connectivity_derivative_path, run_batch_connectivity
from meg_tokens.workflows.erp import erp_derivative_path
from meg_tokens.reports.connectivity import load_connectivity_pairs_with_metadata
from meg_tokens.reports.seed_connectivity import run_batch_plot_seed_connectivity
from meg_tokens.workflows.connectivity import extract_connectivity_features


LABELS = ["NodeA-lh", "NodeB-rh"]
TIMES = [0.0, 0.5, 1.0, 1.5, 2.0]


def _write_erp(root, subject, run, condition, data):
    data = np.asarray(data, dtype=float)
    path = erp_derivative_path(
        root,
        subject=subject,
        run=run,
        condition=condition,
        align_to="enter",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )
    trials_path = erp_derivative_path(
        root,
        subject=subject,
        run=run,
        condition=condition,
        align_to="enter",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erptrials",
        extension=".tsv",
    )
    save_table(
        trials_path,
        pd.DataFrame({"nTrialIndex": list(range(1, data.shape[0] + 1))}),
        metadata={"stage": "erp_parcellation"},
    )
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
            "alignment": "enter",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
            "trial_table": str(trials_path),
        },
    )
    return path


def _write_connectivity(root, subject, run, condition, before_value, after_value):
    data = np.zeros((2, 1, 2, 2), dtype=float)
    data[0, 0] = np.array([[0.0, before_value], [before_value, 0.0]])
    data[1, 0] = np.array([[0.0, after_value], [after_value, 0.0]])
    path = connectivity_derivative_path(
        root,
        subject=subject,
        run=run,
        condition=condition,
        align_to="enter",
        source_method="dSPM",
        parc="HCPMMP1",
        method="imcoh",
    )
    return save_array(
        path,
        data,
        dims=("window", "band", "node_from", "node_to"),
        coords={"window": ["before", "after"], "band": ["alpha"], "node_from": LABELS, "node_to": LABELS},
        metadata={
            "stage": "connectivity",
            "subject": subject,
            "run": run,
            "condition": condition,
            "alignment": "enter",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
            "method": "imcoh",
        },
    )


def test_run_batch_connectivity_writes_window_band_derivative(tmp_path, monkeypatch):
    data = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
    _write_erp(tmp_path, "H01", "Fast1", "Fast", data)

    def fixed_connectivity(window_data, **kwargs):
        value = float(window_data.shape[-1])
        return np.array([[[0.0, value], [value, 0.0]]])

    monkeypatch.setattr(
        "meg_tokens.workflows.connectivity.compute_spectral_connectivity",
        fixed_connectivity,
    )

    outputs = run_batch_connectivity(
        feature_dir=str(tmp_path),
        output_dir=str(tmp_path),
        subjects=["H01"],
        conditions=["Fast"],
        fmin=[8.0],
        fmax=[15.0],
        freq_names=["alpha"],
        before_window=(0.0, 1.0),
        after_window=(1.0, 2.0),
    )

    loaded = load_array(outputs["H01"][0], require_sidecar=True)
    assert loaded.data.shape == (2, 1, 2, 2)
    assert loaded.metadata["dims"] == ["window", "band", "node_from", "node_to"]
    assert loaded.metadata["coords"]["band"] == ["alpha"]
    assert loaded.metadata["coords"]["node_from"] == LABELS
    assert loaded.metadata["metadata"]["input_feature"].endswith("_erp.npy")


def test_connectivity_workflow_declares_erp_and_output(tmp_path, monkeypatch):
    data = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
    input_path = _write_erp(tmp_path, "H01", "Fast1", "Fast", data)

    monkeypatch.setattr(
        "meg_tokens.workflows.connectivity.compute_spectral_connectivity",
        lambda window_data, **kwargs: np.zeros((1, 2, 2)),
    )
    result = extract_connectivity_features(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H01"],
        settings=ConnectivityConfig(
            conditions=("Fast",),
            bands=(("alpha", 8.0, 15.0),),
            before_window=(0.0, 1.0),
            after_window=(1.0, 2.0),
        ),
    )

    assert result.stage == "connectivity_features"
    assert result.inputs == (input_path,)
    assert len(result.outputs) == 1


def test_load_connectivity_pairs_averages_runs_within_subject(tmp_path):
    _write_connectivity(tmp_path, "H01", "Fast1", "Fast", before_value=1.0, after_value=3.0)
    _write_connectivity(tmp_path, "H01", "Fast2", "Fast", before_value=3.0, after_value=5.0)
    _write_connectivity(tmp_path, "H02", "Fast1", "Fast", before_value=10.0, after_value=20.0)

    before, after, node_names, subjects, input_paths = load_connectivity_pairs_with_metadata(
        tmp_path,
        "Fast",
        "alpha",
    )

    assert subjects == ["H01", "H02"]
    assert node_names == LABELS
    assert len(input_paths) == 3
    assert before.shape == (2, 2, 2)
    assert before[0, 0, 1] == 2.0
    assert after[0, 0, 1] == 4.0
    assert before[1, 0, 1] == 10.0


def test_seed_connectivity_writes_node_vector_derivatives(tmp_path, monkeypatch):
    _write_connectivity(tmp_path, "H01", "Fast1", "Fast", before_value=1.0, after_value=4.0)
    _write_connectivity(tmp_path, "H02", "Fast1", "Fast", before_value=2.0, after_value=6.0)

    def fixed_ttest(diff, n_permutations, n_jobs):
        return np.array([0.0, 5.0]), np.array([1.0, 0.01]), np.array([0.0, 1.0])

    monkeypatch.setattr(
        "meg_tokens.reports.seed_connectivity.permutation_t_test",
        fixed_ttest,
    )

    outputs = run_batch_plot_seed_connectivity(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        condition="Fast",
        seed_roi="NodeA-lh",
        band="alpha",
        p_threshold=0.05,
        n_permutations=10,
    )

    seed_map = load_array(outputs["seed_map"], require_sidecar=True)
    assert seed_map.data.tolist() == [1.0, 3.5]
    assert seed_map.metadata["coords"]["node"] == LABELS
    assert seed_map.metadata["metadata"]["seed_index"] == 0
