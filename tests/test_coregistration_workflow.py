import numpy as np
import mne
import pytest

from meg_tokens.meg.coregistration import (
    compose_trans,
    has_own_anatomy,
    head_to_mri,
    reference_session,
    session_alignment,
)
from meg_tokens.meg.meg_bids import RAW_MEG_RUNS, raw_meg_sessions, staged_subjects


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


# --- BIDS raw-layer discovery -------------------------------------------------


def _stage(root, subject, sessions):
    meg = root / f"sub-{subject}" / "meg"
    meg.mkdir(parents=True)
    for condition, run in sessions:
        (meg / f"sub-{subject}_task-tokens_acq-{condition}_run-{run}_meg.ds").mkdir()
    return meg


def test_raw_meg_sessions_names_each_recording_and_skips_a_missing_one(tmp_path):
    """H05's absent RT run 1 must leave the other nine identified as themselves."""
    _stage(tmp_path, "H05", [(c.lower(), r) for c, r in RAW_MEG_RUNS if (c, r) != ("RT", 1)])

    found = raw_meg_sessions(tmp_path, "H05")

    assert [(spec.condition, spec.run) for spec, _ in found] == [
        ("Fast", "1"), ("Fast", "2"), ("Fast", "3"), ("Fast", "4"),
        ("RT", "2"),
        ("Slow", "1"), ("Slow", "2"), ("Slow", "3"), ("Slow", "4"),
    ]
    assert found[4][1].name == "sub-H05_task-tokens_acq-rt_run-2_meg.ds"
    assert raw_meg_sessions(tmp_path, "H03") == []


def test_staged_subjects_skips_emptyroom_and_unstaged_trees(tmp_path):
    _stage(tmp_path, "H02", [("fast", 1)])
    _stage(tmp_path, "H03", [("fast", 1)])
    # Empty-room is session-nested, so it has no subject-level meg/.
    (tmp_path / "sub-emptyroom" / "ses-20180213" / "meg").mkdir(parents=True)
    (tmp_path / "sub-H04" / "beh").mkdir(parents=True)

    assert staged_subjects(tmp_path) == ["H02", "H03"]


def test_has_own_anatomy_requires_a_real_reconstruction(tmp_path):
    """A scaled template must not be relabelled as reconstructed on a re-run."""
    (tmp_path / "H02" / "bem").mkdir(parents=True)
    (tmp_path / "H07" / "bem").mkdir(parents=True)
    (tmp_path / "H07" / "MRI scaling parameters.cfg").write_text("")

    assert has_own_anatomy(tmp_path, "H02")
    assert not has_own_anatomy(tmp_path, "H07")  # scaled template
    assert not has_own_anatomy(tmp_path, "H10")  # no anatomy at all


# --- session geometry ---------------------------------------------------------


def test_reference_session_picks_the_most_typical_head_frame():
    """The outlier session must not become the anchor everything is carried from."""
    infos = [
        _shifted_info(np.zeros(3)),
        _shifted_info(np.array([0.0005, 0.0, 0.0])),
        _shifted_info(np.array([0.001, 0.0, 0.0])),
        _shifted_info(np.array([0.05, 0.0, 0.0])),  # the outlier
    ]
    assert reference_session(infos) == 1


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


# --- the QC summary is a view of the sidecars, not of one run ------------------


def _write_trans(layout, subject, condition, run, **metadata):
    from meg_tokens.io import ensure_dir, save_sidecar

    path = layout.source(
        subject=subject, run=run, condition=condition, suffix="trans", extension=".fif"
    )
    ensure_dir(path.parent)
    path.write_bytes(b"")
    save_sidecar(
        path,
        {
            "kind": "meg_mri_transform",
            "subject": subject,
            "condition": condition,
            "run": run,
            "flagged": False,
            **metadata,
        },
    )
    return path


def test_qc_table_reads_every_persisted_transform(tmp_path):
    """Coregistering one subject must not shrink a cohort-wide summary."""
    from meg_tokens.io import DerivativeLayout
    from meg_tokens.workflows.coregistration import qc_table

    layout = DerivativeLayout(tmp_path)
    _write_trans(layout, "H02", "Fast", "1", dig_mri_median_mm=3.0)
    _write_trans(layout, "H02", "Slow", "1", dig_mri_median_mm=3.0)
    _write_trans(layout, "H03", "Fast", "1", dig_mri_median_mm=2.5)

    table = qc_table(layout)

    assert len(table) == 3
    assert sorted(set(table["subject"])) == ["H02", "H03"]


def test_qc_table_ignores_derivatives_that_are_not_transforms(tmp_path):
    from meg_tokens.io import DerivativeLayout, ensure_dir, save_sidecar
    from meg_tokens.workflows.coregistration import qc_table

    layout = DerivativeLayout(tmp_path)
    _write_trans(layout, "H02", "Fast", "1", dig_mri_median_mm=3.0)
    stray = layout.source(
        subject="H02", run="2", condition="Fast", suffix="trans", extension=".fif"
    )
    ensure_dir(stray.parent)
    stray.write_bytes(b"")
    save_sidecar(stray, {"kind": "something_else", "subject": "H02"})

    assert len(qc_table(layout)) == 1
