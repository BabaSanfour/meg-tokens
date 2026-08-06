import numpy as np

from meg_tokens.core import ProjectConfig, SpatialDecodingConfig
from meg_tokens.io import load_array, save_array
from meg_tokens.workflows.spatial_decoding import (
    load_spatial_decoding_inputs,
    run_spatial_decoding,
)


def _write_inputs(root):
    save_array(
        root / "X.npy",
        np.arange(24, dtype=float).reshape(8, 3),
        dims=("trial", "sensor"),
        coords={"sensor": ["MEG001", "MEG002", "MEG003"]},
    )
    save_array(
        root / "y.npy",
        np.asarray(["Fast", "Slow"] * 4),
        dims=("trial",),
    )
    save_array(
        root / "groups.npy",
        np.asarray(["H01"] * 4 + ["H02"] * 4),
        dims=("trial",),
    )


def test_load_spatial_decoding_inputs_preserves_sensor_coordinates(tmp_path):
    _write_inputs(tmp_path)

    X, y, groups, channels, inputs = load_spatial_decoding_inputs(
        tmp_path, ("Fast", "Slow")
    )

    assert X.shape == (8, 3)
    assert set(y) == {0, 1}
    assert groups.shape == (8,)
    assert channels == ["MEG001", "MEG002", "MEG003"]
    assert len(inputs) == 3


def test_spatial_decoding_workflow_saves_labeled_scores(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr(
        "meg_tokens.workflows.spatial_decoding.compute_spatial_decoding",
        lambda **kwargs: np.asarray([0.6, 0.7, 0.8]),
    )

    result = run_spatial_decoding(
        ProjectConfig(data_root=tmp_path),
        settings=SpatialDecodingConfig(),
        data_dir=tmp_path,
    )
    loaded = load_array(result.outputs[0], require_sidecar=True)

    assert result.stage == "spatial_decoding"
    assert len(result.inputs) == 3
    assert loaded.metadata["dims"] == ["sensor"]
    assert loaded.metadata["coords"]["sensor"] == ["MEG001", "MEG002", "MEG003"]
