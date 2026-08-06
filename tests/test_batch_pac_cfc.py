import numpy as np

from meg_tokens.core import PACConfig, ProjectConfig
from meg_tokens.io import load_array, save_array
from meg_tokens.workflows.hilbert import hilbert_feature_derivative_path
from meg_tokens.workflows.pac import (
    find_hilbert_feature_arrays,
    pac_derivative_path,
    run_batch_pac_cfc,
)
from meg_tokens.workflows.pac import extract_pac_features


def _write_hilbert(root, *, band, feature, data):
    path = hilbert_feature_derivative_path(
        root,
        subject="H01",
        run="Fast1",
        condition="Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band=band,
        feature=feature,
    )
    save_array(
        path,
        data,
        dims=("trial", "feature", "time"),
        coords={
            "trial": [1, 2],
            "feature": ["ROI-lh", "ROI-rh"],
            "time_sec": np.arange(data.shape[-1], dtype=float) / 100.0,
        },
        metadata={
            "stage": "hilbert_features",
            "kind": "erp_hilbert_feature",
            "subject": "H01",
            "run": "1",
            "condition": "Fast",
            "alignment": "go",
            "source_method": "dSPM",
            "parcellation": "HCPMMP1",
            "band": band,
            "feature": feature,
            "sfreq_hz": 100.0,
        },
    )
    return path


def _write_inputs(root):
    phase_line = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    phase = np.tile(phase_line, (2, 2, 1))
    amplitude = np.ones_like(phase)
    amplitude[:, 0, :] = 1.5 + np.cos(phase[:, 0, :])
    phase_path = _write_hilbert(root, band="theta", feature="phase", data=phase)
    amp_path = _write_hilbert(root, band="gamma_low", feature="amplitude", data=amplitude)
    return phase_path, amp_path


def test_find_hilbert_feature_arrays_matches_stage11_paths(tmp_path):
    phase_path, _ = _write_inputs(tmp_path)

    found = find_hilbert_feature_arrays(
        tmp_path,
        "H1",
        "Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        band="theta",
        feature="phase",
    )

    assert found == [phase_path]


def test_run_batch_pac_cfc_writes_modulation_index_derivative(tmp_path):
    phase_path, amp_path = _write_inputs(tmp_path)

    outputs = run_batch_pac_cfc(
        feature_dir=tmp_path,
        output_dir=tmp_path,
        subjects=["H01"],
        conditions=["Fast"],
        phase_bands=["theta"],
        amplitude_bands=["gamma_low"],
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        n_bins=18,
    )
    expected = pac_derivative_path(
        tmp_path,
        subject="H01",
        run="1",
        condition="Fast",
        align_to="go",
        source_method="dSPM",
        parc="HCPMMP1",
        phase_bands=["theta"],
        amplitude_bands=["gamma_low"],
        method="modulation_index",
    )
    loaded = load_array(expected, require_sidecar=True)

    assert outputs["H01"] == [expected]
    assert loaded.data.shape == (1, 1, 2)
    assert loaded.data[0, 0, 0] > loaded.data[0, 0, 1]
    assert loaded.metadata["dims"] == ["phase_band", "amplitude_band", "feature"]
    assert loaded.metadata["coords"]["phase_band"] == ["theta"]
    assert loaded.metadata["coords"]["amplitude_band"] == ["gamma_low"]
    assert loaded.metadata["coords"]["feature"] == ["ROI-lh", "ROI-rh"]
    meta = loaded.metadata["metadata"]
    assert meta["stage"] == "pac_cfc"
    assert meta["method"] == "modulation_index"
    assert meta["input_phase_features"] == [str(phase_path)]
    assert meta["input_amplitude_features"] == [str(amp_path)]


def test_pac_workflow_declares_hilbert_inputs_and_output(tmp_path):
    project = ProjectConfig(data_root=tmp_path)
    phase_path, amplitude_path = _write_inputs(project.bids_root)

    result = extract_pac_features(
        project,
        subjects=["H1"],
        settings=PACConfig(
            conditions=("Fast",),
            phase_bands=("theta",),
            amplitude_bands=("gamma_low",),
        ),
    )

    assert result.stage == "pac_features"
    assert set(result.inputs) == {phase_path, amplitude_path}
    assert len(result.outputs) == 1
