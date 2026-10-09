import numpy as np
import mne
import pytest

from meg_tokens.meg.coregistration import compose_trans, head_to_mri, session_alignment


def _rotation(angle, axis=2):
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.eye(3)
    other = [index for index in range(3) if index != axis]
    rotation[np.ix_(other, other)] = [[cos, -sin], [sin, cos]]
    return rotation


def _info(fiducials):
    info = mne.create_info(["MEG0111"], 1000.0, "mag")
    montage = mne.channels.make_dig_montage(coord_frame="head", **fiducials)
    with info._unlock():
        info["dig"] = montage.dig
    return info


def _shifted_info(offset):
    return _info(
        {
            "nasion": np.array([0.0, 0.1, 0.0]) + offset,
            "lpa": np.array([-0.07, 0.0, 0.0]) + offset,
            "rpa": np.array([0.07, 0.0, 0.0]) + offset,
        }
    )


# --- session geometry ---------------------------------------------------------


def test_session_alignment_recovers_a_translated_head_frame():
    reference = _shifted_info(np.zeros(3))
    offset = np.array([0.003, -0.001, 0.002])

    transform, residual_mm = session_alignment(reference, _shifted_info(offset))

    assert transform[:3, 3] == pytest.approx(offset)
    assert residual_mm == pytest.approx(np.zeros(3), abs=1e-9)
    assert session_alignment(reference, reference)[0] == pytest.approx(np.eye(4))


def test_compose_trans_sends_a_moved_session_to_the_same_mri_point():
    """A physical point must land in one MRI location whichever session saw it."""
    reference = _shifted_info(np.zeros(3))
    offset = np.array([0.004, -0.002, 0.001])
    session = _shifted_info(offset)

    head_to_mri_t = mne.transforms.Transform(
        "head", "mri", np.block([[_rotation(0.2), np.array([[0.01], [0.02], [0.03]])],
                                 [np.zeros((1, 3)), np.ones((1, 1))]])
    )
    alignment, _ = session_alignment(reference, session)
    composed = compose_trans(head_to_mri_t, alignment)

    point_in_reference = np.array([[0.02, 0.05, 0.06]])
    point_in_session = point_in_reference + offset
    assert mne.transforms.apply_trans(composed, point_in_session) == pytest.approx(
        mne.transforms.apply_trans(head_to_mri_t, point_in_reference)
    )
    assert (composed["from"], composed["to"]) == (
        head_to_mri_t["from"],
        head_to_mri_t["to"],
    )


def test_head_to_mri_normalises_either_direction():
    forward = mne.transforms.Transform("head", "mri", _rotation_trans(0.3))
    assert head_to_mri(forward)["trans"] == pytest.approx(forward["trans"])

    inverse = mne.transforms.invert_transform(forward)
    assert inverse["from"] != forward["from"]
    assert head_to_mri(inverse)["trans"] == pytest.approx(forward["trans"])


def _rotation_trans(angle):
    transform = np.eye(4)
    transform[:3, :3] = _rotation(angle)
    transform[:3, 3] = [0.01, 0.02, 0.03]
    return transform
