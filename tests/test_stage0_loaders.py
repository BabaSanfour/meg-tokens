import numpy as np
import pytest

from meg_tokens.io import save_array
from meg_tokens.workflows.connectivity import find_roi_timeseries
from meg_tokens.workflows.decoding import load_decoding_inputs
from meg_tokens.reports.connectivity import load_connectivity_pairs


def test_load_decoding_inputs_filters_string_conditions(tmp_path):
    X = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    y = np.array(["Fast", "Slow", "Fast", "Other", "Slow"], dtype=object)
    groups = np.array(["H01", "H01", "H02", "H03", "H02"], dtype=object)
    times = np.array([-100.0, 0.0, 100.0])

    save_array(tmp_path / "X.npy", X, dims=("epoch", "feature", "time"))
    save_array(tmp_path / "y.npy", y, dims=("epoch",))
    save_array(tmp_path / "groups.npy", groups, dims=("epoch",))
    save_array(tmp_path / "times_ms.npy", times, dims=("time",))

    loaded_X, loaded_y, loaded_groups, loaded_times = load_decoding_inputs(tmp_path, ["Fast", "Slow"])

    assert loaded_X.shape == (4, 2, 3)
    assert loaded_y.tolist() == [0, 1, 0, 1]
    assert loaded_groups.tolist() == ["H01", "H01", "H02", "H02"]
    np.testing.assert_array_equal(loaded_times, times)


def test_load_decoding_inputs_requires_matching_epoch_counts(tmp_path):
    save_array(tmp_path / "X.npy", np.zeros((2, 3, 4)), dims=("epoch", "feature", "time"))
    save_array(tmp_path / "y.npy", np.array(["Fast"]), dims=("epoch",))

    with pytest.raises(ValueError, match="X and y disagree"):
        load_decoding_inputs(tmp_path, ["Fast"])


def test_find_roi_timeseries_uses_subject_condition_contract(tmp_path):
    target = tmp_path / "H01" / "H01_Fast_roi_timeseries.npy"
    save_array(target, np.zeros((2, 3, 4)), dims=("epoch", "roi", "time"))

    assert find_roi_timeseries(tmp_path, "H01", "Fast") == target


def test_load_connectivity_pairs_reads_staged_window_array(tmp_path):
    path = tmp_path / "sub-H01_task-tokens_desc-fast_connectivity.npy"
    data = np.stack([np.eye(2), np.eye(2) * 2], axis=0)[:, np.newaxis]
    save_array(
        path,
        data,
        dims=("window", "band", "node_from", "node_to"),
        coords={
            "window": ["before", "after"],
            "band": ["alpha"],
            "node_from": ["A", "B"],
            "node_to": ["A", "B"],
        },
        metadata={"subject": "H01", "condition": "Fast"},
    )

    before_group, after_group = load_connectivity_pairs(tmp_path, "Fast", "alpha")

    assert before_group.shape == (1, 2, 2)
    assert after_group.shape == (1, 2, 2)
    np.testing.assert_array_equal(before_group[0], np.eye(2))
    np.testing.assert_array_equal(after_group[0], np.eye(2) * 2)
