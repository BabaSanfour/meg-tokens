import numpy as np
import pytest

from meg_tokens.meg.coregistration import read_ctf_headshape, rigid_transform


def _write_headshape(path, points):
    """A CTF .eeg export: each landmark traced twice, then the bare points."""
    lines = [
        "NZ\t10.0\t0.0\t0.0", "OG\t0.0\t7.0\t0.0", "OD\t0.0\t-7.0\t0.0",
        "NZ\t12.0\t0.0\t0.0", "OG\t0.0\t7.0\t0.0", "OD\t0.0\t-7.0\t0.0",
    ]
    lines += ["\t".join(f"{value}" for value in point) for point in points]
    path.write_text("\n".join(lines) + "\n")
    return path


def _rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    )


def test_read_ctf_headshape_converts_ctf_als_cm_to_head_ras_metres(tmp_path):
    headshape = read_ctf_headshape(
        _write_headshape(tmp_path / "hs.eeg", [[10.5, 2.0, 7.0], [-3.0, -1.0, 4.0]])
    )

    # CTF ALS centimetres -> MNE head RAS metres: x <- -y, y <- x, z <- z, /100.
    assert headshape.points == pytest.approx(
        np.array([[-0.02, 0.105, 0.07], [0.01, -0.03, 0.04]])
    )
    # Nasion is the mean of its two traces, 10 and 12 cm.
    assert headshape.fiducials["nasion"] == pytest.approx([0.0, 0.11, 0.0])
    assert headshape.fiducials["lpa"] == pytest.approx([-0.07, 0.0, 0.0])
    assert headshape.fiducials["rpa"] == pytest.approx([0.07, 0.0, 0.0])


def test_rigid_transform_recovers_a_known_pose():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(12, 3))
    rotation = _rotation(0.3, -0.2)
    offset = np.array([0.01, -0.02, 0.03])

    transform = rigid_transform(source, source @ rotation.T + offset)

    assert transform[:3, :3] == pytest.approx(rotation)
    assert transform[:3, 3] == pytest.approx(offset)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)


def test_rigid_transform_never_reflects():
    """A mirrored point set must be fitted by a rotation, not a reflection."""
    rng = np.random.default_rng(1)
    source = rng.normal(size=(20, 3))
    mirrored = source * np.array([1.0, 1.0, -1.0])

    transform = rigid_transform(source, mirrored)

    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
