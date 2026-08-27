import numpy as np
import mne
import pytest

from meg_tokens.meg.coregistration import (
    FIDUCIAL_ORDER,
    head_fiducials,
    headshape_in_head_frame,
    info_with_headshape,
    read_ctf_headshape,
    rigid_transform,
    scalp_distances,
)


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


def _info_with_fiducials(fiducials):
    info = mne.create_info(["MEG0111"], 1000.0, "mag")
    montage = mne.channels.make_dig_montage(coord_frame="head", **fiducials)
    with info._unlock():
        info["dig"] = montage.dig
    return info


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


def test_read_ctf_headshape_ignores_trailing_columns_on_a_landmark(tmp_path):
    """Real exports pad some landmark lines with trailing zeros."""
    path = tmp_path / "hs.eeg"
    path.write_text(
        "NZ\t10.0\t0.0\t0.0\nOG\t0.0\t7.0\t0.0\n"
        "OD\t0.0\t-7.0\t0.0\t0\t0\t0\n"
        "10.5\t2.0\t7.0\n"
    )

    headshape = read_ctf_headshape(path)

    assert headshape.fiducials["rpa"] == pytest.approx([0.07, 0.0, 0.0])
    assert headshape.points == pytest.approx(np.array([[-0.02, 0.105, 0.07]]))


def test_read_ctf_headshape_refuses_an_unusable_file(tmp_path):
    """Landmarks are what place the points in a frame; without them, nothing works."""
    path = tmp_path / "hs.eeg"
    path.write_text("NZ\t10.0\t0.0\t0.0\n1.0\t1.0\t1.0\n")
    with pytest.raises(ValueError, match="missing landmarks"):
        read_ctf_headshape(path)

    with pytest.raises(ValueError, match="no head-shape points"):
        read_ctf_headshape(_write_headshape(tmp_path / "empty.eeg", []))


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


def test_headshape_in_head_frame_carries_points_and_reports_no_residual(tmp_path):
    """A recording that localized the same landmarks leaves nothing unexplained."""
    headshape = read_ctf_headshape(
        _write_headshape(tmp_path / "hs.eeg", [[10.5, 2.0, 7.0], [-3.0, -1.0, 4.0]])
    )
    rotation = _rotation(0.15, 0.1)
    offset = np.array([0.002, -0.001, 0.004])
    moved = {
        name: rotation @ headshape.fiducials[name] + offset for name in FIDUCIAL_ORDER
    }

    points, residual_mm = headshape_in_head_frame(headshape, _info_with_fiducials(moved))

    assert residual_mm == pytest.approx(np.zeros(3), abs=1e-9)
    assert points == pytest.approx(headshape.points @ rotation.T + offset)


def test_headshape_in_head_frame_reports_landmark_disagreement(tmp_path):
    """A 3 mm disagreement must surface as residual, not be absorbed by the fit.

    It surfaces as 2 mm, not 3: the best rigid match spreads one landmark's
    error across all three rather than leaving it all on the nasion.
    """
    headshape = read_ctf_headshape(
        _write_headshape(tmp_path / "hs.eeg", [[10.5, 2.0, 7.0]])
    )
    nudged = dict(headshape.fiducials)
    nudged["nasion"] = nudged["nasion"] + np.array([0.0, 0.003, 0.0])

    _, residual_mm = headshape_in_head_frame(headshape, _info_with_fiducials(nudged))

    assert residual_mm.max() == pytest.approx(2.0, abs=0.1)


def test_info_with_headshape_keeps_the_recordings_own_landmarks():
    fiducials = {
        "nasion": [0.0, 0.1, 0.0], "lpa": [-0.07, 0.0, 0.0], "rpa": [0.07, 0.0, 0.0],
    }
    info = _info_with_fiducials(fiducials)
    points = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])

    updated = info_with_headshape(info, points)

    assert head_fiducials(updated)["nasion"] == pytest.approx(fiducials["nasion"])
    extra = np.array(
        [point["r"] for point in updated["dig"] if point["kind"] == 4]
    )
    assert extra == pytest.approx(points)
    assert len(info["dig"]) == 3, "the input info must not be mutated"


def _flat_scalp(spacing=0.002, n=40):
    """A dense patch of surface in the MRI z = 0 plane.

    Flat and gridded so the expected distance is exact: a point raised straight
    off a vertex by less than `spacing` has that vertex as its nearest, with no
    discretization error to absorb into a tolerance.
    """
    axis = (np.arange(n) - n / 2) * spacing
    x, y = np.meshgrid(axis, axis)
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def test_scalp_distances_measures_head_frame_points_against_the_mri_surface():
    """The points arrive in head coordinates; the surface is in MRI ones."""
    scalp = _flat_scalp()
    offset = np.array([0.03, -0.02, 0.01])
    trans = np.eye(4)
    trans[:3, 3] = offset

    # 1 mm clear of the scalp in MRI, handed over in the head frame.
    raised = scalp[::37] + np.array([0.0, 0.0, 0.001]) - offset

    distances = scalp_distances(
        raised, mne.transforms.Transform("head", "mri", trans), scalp
    )

    assert distances == pytest.approx(np.full(len(raised), 0.001))
