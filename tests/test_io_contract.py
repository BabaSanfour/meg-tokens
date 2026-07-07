import json

import numpy as np

from meg_tokens.io import derivative_path, load_array, save_array, sidecar_path


def test_derivative_path_builds_bids_derivative_name(tmp_path):
    path = derivative_path(
        tmp_path,
        subject="sub-H01",
        task="tokens",
        description="go alpha fast",
        suffix="power",
        extension="npy",
    )

    assert path == (
        tmp_path
        / "derivatives"
        / "meg-tokens"
        / "sub-H01"
        / "meg"
        / "sub-H01_task-tokens_desc-goalphafast_power.npy"
    )


def test_save_and_load_array_with_sidecar(tmp_path):
    path = tmp_path / "sub-H01_task-tokens_desc-goalpha_power.npy"
    data = np.arange(6, dtype=float).reshape(2, 3)

    save_array(
        path,
        data,
        dims=("roi", "time"),
        coords={"roi": ["M1", "S1"], "time": [0.0, 0.01, 0.02]},
        metadata={"stage": "power", "subject": "H01", "band": "alpha"},
    )

    loaded = load_array(path, expected_ndim=2, require_sidecar=True)

    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.metadata["dims"] == ["roi", "time"]
    assert loaded.metadata["coords"]["roi"] == ["M1", "S1"]
    assert loaded.metadata["metadata"]["band"] == "alpha"

    with sidecar_path(path).open("r", encoding="utf-8") as f:
        raw_sidecar = json.load(f)
    assert raw_sidecar["shape"] == [2, 3]


def test_save_array_rejects_mismatched_dims(tmp_path):
    data = np.zeros((2, 3))

    try:
        save_array(tmp_path / "bad.npy", data, dims=("roi",))
    except ValueError as exc:
        assert "dims has length" in str(exc)
    else:
        raise AssertionError("save_array accepted mismatched dims")
