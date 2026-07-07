import numpy as np
import pandas as pd

from meg_tokens.io import load_array, save_array, save_table
from meg_tokens.utils.batch_dpca import build_dpca_tensor, run_pca_trajectory
from meg_tokens.utils.batch_erp_parcellation import erp_derivative_path


LABELS = ["Pair-lh", "Pair-rh"]
TIMES = [-0.1, 0.0, 0.1]


def _write_erp(root, subject, run, condition, data, trial_classes=None, choices=None):
    data = np.asarray(data, dtype=float)
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
    n_trials = data.shape[0]
    trial_classes = trial_classes or [1] * n_trials
    choices = choices or ["left"] * n_trials
    save_table(
        trial_path,
        pd.DataFrame({
            "nTrialIndex": list(range(1, n_trials + 1)),
            "sTrialClass": trial_classes,
            "nChoiceMade": choices,
            "manifest_trial": list(range(1, n_trials + 1)),
        }),
        metadata={"stage": "erp_parcellation"},
    )
    save_array(
        path,
        data,
        dims=("trial", "label", "time"),
        coords={"trial": list(range(1, n_trials + 1)), "label": LABELS, "time_sec": TIMES},
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


def test_run_pca_trajectory_writes_sidecar_derivatives(tmp_path):
    base = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    _write_erp(tmp_path, "H01", "Fast1", "Fast", base + 1)
    _write_erp(tmp_path, "H01", "Slow1", "Slow", base + 11)
    _write_erp(tmp_path, "H02", "Fast1", "Fast", base + 21)
    _write_erp(tmp_path, "H02", "Slow1", "Slow", base + 31)

    outputs = run_pca_trajectory(
        feature_dir=str(tmp_path),
        output_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        subjects=("H01", "H02"),
        n_components=2,
    )

    trajectory = load_array(outputs["trajectory"], require_sidecar=True)
    loadings = load_array(outputs["loadings"], require_sidecar=True)
    variance = load_array(outputs["variance"], require_sidecar=True)
    observations = pd.read_csv(outputs["observations"], sep="\t")

    assert trajectory.data.shape == (2, 2, 3)
    assert trajectory.metadata["dims"] == ["condition", "component", "time"]
    assert trajectory.metadata["coords"]["condition"] == ["Fast", "Slow"]
    assert loadings.data.shape == (2, 2)
    assert variance.data.shape == (2,)
    assert observations["condition"].tolist() == ["Fast", "Slow", "Fast", "Slow"]
    assert observations["n_trials"].tolist() == [2, 2, 2, 2]


def test_run_pca_trajectory_can_use_label_selection(tmp_path):
    data = np.zeros((2, 2, 3), dtype=float)
    data[:, 0, :] = 5.0
    data[:, 1, :] = 1.0
    _write_erp(tmp_path, "H01", "Fast1", "Fast", data)
    _write_erp(tmp_path, "H01", "Slow1", "Slow", data + 2.0)

    outputs = run_pca_trajectory(
        feature_dir=str(tmp_path),
        output_dir=str(tmp_path),
        conditions=("Fast", "Slow"),
        subjects=("H01",),
        labels=("Pair-lh",),
        n_components=1,
    )

    loadings = load_array(outputs["loadings"], require_sidecar=True)
    assert loadings.metadata["coords"]["feature"] == ["Pair-lh"]
    assert loadings.data.shape == (1, 1)


def test_build_dpca_tensor_uses_real_trial_metadata_cells():
    X = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    metadata = pd.DataFrame({
        "sTrialClass": [1, 1, 2, 2],
        "nChoiceMade": ["left", "right", "left", "right"],
    })

    trial_tensor, mean_tensor, values, counts = build_dpca_tensor(
        X,
        metadata,
        marginalize_cols=("sTrialClass", "nChoiceMade"),
    )

    assert trial_tensor.shape == (1, 2, 2, 2, 3)
    assert mean_tensor.shape == (2, 2, 2, 3)
    assert values == {"sTrialClass": ["1", "2"], "nChoiceMade": ["left", "right"]}
    assert counts["n_trials"].tolist() == [1, 1, 1, 1]
    np.testing.assert_array_equal(mean_tensor[:, 0, 0, :], X[0])
