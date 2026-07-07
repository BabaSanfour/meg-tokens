import numpy as np

from meg_tokens.io import load_array, save_array
from meg_tokens.utils.batch_erp_parcellation import erp_derivative_path
from meg_tokens.utils.batch_hilbert_features import (
    hilbert_feature_derivative_path,
    run_batch_hilbert_features,
)


def _write_erp(root):
    sfreq = 100.0
    n_times = 400
    times = np.arange(n_times) / sfreq
    path = erp_derivative_path(
        root,
        subject="H01",
        run="Fast1",
        condition="Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        suffix="erp",
        extension=".npy",
    )
    base = np.stack([
        np.sin(2 * np.pi * 10.0 * times),
        np.cos(2 * np.pi * 10.0 * times),
    ])
    data = np.stack([base, base * 0.5], axis=0)
    save_array(
        path,
        data,
        dims=("trial", "label", "time"),
        coords={"trial": [1, 2], "label": ["Alpha-lh", "Alpha-rh"], "time_sec": times},
        metadata={
            "stage": "erp_parcellation",
            "subject": "H01",
            "condition": "Fast",
            "run": "Fast1",
            "alignment": "go",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
        },
    )
    return path


def test_run_batch_hilbert_features_writes_sidecar_backed_arrays(tmp_path):
    input_path = _write_erp(tmp_path)

    outputs = run_batch_hilbert_features(
        feature_dir=str(tmp_path),
        output_dir=str(tmp_path),
        subjects=["H1"],
        conditions=["Fast"],
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        labels=["Alpha-lh"],
        freq_bands={"alpha": (8.0, 12.0)},
        features=("amplitude", "phase"),
        n_jobs=1,
    )

    expected = hilbert_feature_derivative_path(
        tmp_path,
        subject="H01",
        run="Fast1",
        condition="Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band="alpha",
        feature="amplitude",
    )
    loaded = load_array(expected, require_sidecar=True)

    assert len(outputs["H01"]) == 2
    assert expected in outputs["H01"]
    assert expected.name == "sub-H01_task-tokens_run-1_desc-fast-go-dSPM-HCPMMP1-alpha-amplitude_hilbertfeature.npy"
    assert loaded.data.shape == (2, 1, 400)
    assert loaded.metadata["dims"] == ["trial", "feature", "time"]
    assert loaded.metadata["coords"]["trial"] == [1, 2]
    assert loaded.metadata["coords"]["feature"] == ["Alpha-lh"]
    assert loaded.metadata["metadata"]["stage"] == "hilbert_features"
    assert loaded.metadata["metadata"]["feature"] == "amplitude"
    assert loaded.metadata["metadata"]["input_feature"] == str(input_path)
