import json

import numpy as np
import xarray as xr

from meg_tokens.io import (
    derivative_path,
    load_array,
    load_dataarray,
    save_array,
    save_dataarray,
    sidecar_path,
)


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
    assert raw_sidecar["schema_version"] == 1


def test_save_array_rejects_mismatched_dims(tmp_path):
    data = np.zeros((2, 3))

    try:
        save_array(tmp_path / "bad.npy", data, dims=("roi",))
    except ValueError as exc:
        assert "dims has length" in str(exc)
    else:
        raise AssertionError("save_array accepted mismatched dims")


def test_dataarray_round_trip_uses_existing_storage_contract(tmp_path):
    path = tmp_path / "sub-H01_task-tokens_desc-go_erp.npy"
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 2, 3),
        dims=("trial", "label", "time"),
        coords={
            "trial": [1, 2],
            "label": ["M1-lh", "M1-rh"],
            "time": [-0.1, 0.0, 0.1],
        },
        attrs={"stage": "erp", "subject": "H01"},
        name="source_erp",
    )

    save_dataarray(path, data, metadata={"alignment": "go"})
    loaded = load_dataarray(path)

    xr.testing.assert_identical(loaded, data.assign_attrs(alignment="go"))
    assert path.is_file()
    assert sidecar_path(path).is_file()


def test_load_dataarray_rejects_coordinate_length_mismatch(tmp_path):
    path = tmp_path / "invalid.npy"
    save_array(
        path,
        np.zeros((2, 3)),
        dims=("label", "time"),
        coords={"label": ["only-one"], "time": [0.0, 0.1, 0.2]},
    )

    try:
        load_dataarray(path)
    except ValueError as exc:
        assert "Coordinate 'label' has length 1" in str(exc)
    else:
        raise AssertionError("load_dataarray accepted a mismatched coordinate")


def test_save_dataarray_rejects_coordinates_that_cannot_round_trip(tmp_path):
    data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("label", "time"),
        coords={"label": ["M1", "S1"], "time": [0.0, 0.1, 0.2], "condition": "Slow"},
    )

    try:
        save_dataarray(tmp_path / "unsupported.npy", data)
    except ValueError as exc:
        assert "unsupported: ['condition']" in str(exc)
    else:
        raise AssertionError("save_dataarray dropped an unsupported coordinate")
