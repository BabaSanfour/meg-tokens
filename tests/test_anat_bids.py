"""BIDS-anat raw layer, tested against the real FreeSurfer reconstructions.

Point ``MEG_TOKENS_DATA_ROOT`` at the project's ``data_root`` to run --
``subjects_dir`` defaults to ``data_root/IRM``, same as ``ProjectConfig``.
"""

import os
from pathlib import Path

import pytest

from meg_tokens.meg.anat_bids import RAW_T1_RELATIVE_PATH, discover_anat, write_anat_bids


@pytest.fixture()
def real_subjects_dir():
    data_root = os.environ.get("MEG_TOKENS_DATA_ROOT")
    if not data_root:
        pytest.skip("Set MEG_TOKENS_DATA_ROOT to the project's data_root to run this test.")
    root = os.path.join(data_root, "IRM")
    if not os.path.isdir(os.path.join(root, "H02", "mri")):
        pytest.skip(f"Real FreeSurfer subjects_dir not available under {root}")
    return root


def test_discover_anat_finds_the_raw_volume(real_subjects_dir):
    found = discover_anat(real_subjects_dir, "H02")
    assert found == Path(real_subjects_dir) / "H02" / RAW_T1_RELATIVE_PATH
    assert found.is_file()


def test_discover_anat_reports_a_missing_reconstruction_without_raising(real_subjects_dir):
    # H07 and H10 have no FreeSurfer reconstruction in the real dataset. That
    # is a dataset gap to report in the manifest, not an error -- and never a
    # reason to substitute another subject's or a template's volume.
    assert discover_anat(real_subjects_dir, "H07") is None
    assert discover_anat(real_subjects_dir, "H10") is None


def test_discover_anat_reports_an_unconfigured_subjects_dir():
    assert discover_anat(None, "H02") is None


def test_write_anat_bids_copies_the_native_volume_losslessly(tmp_path, real_subjects_dir):
    import nibabel as nib
    import numpy as np

    source_path = discover_anat(real_subjects_dir, "H02")
    bids_root = tmp_path / "BIDS"

    bids_path = write_anat_bids("H02", source_path, bids_root=bids_root)

    assert bids_path.fpath == bids_root / "sub-H02" / "anat" / "sub-H02_T1w.nii.gz"
    assert bids_path.fpath.is_file()

    source = nib.load(str(source_path))
    written = nib.load(str(bids_path.fpath))
    # Native (unconformed) geometry -- confirms this is rawavg.mgz, not
    # FreeSurfer's uniformly-resampled 256x256x256 T1.mgz/orig.mgz.
    assert written.shape == source.shape != (256, 256, 256)
    # The .mgz -> .nii.gz conversion must preserve both the samples and where
    # they sit in scanner space, or the exported volume is not the raw scan.
    assert np.array_equal(np.asarray(written.dataobj), np.asarray(source.dataobj))
    assert np.allclose(written.affine, source.affine, atol=1e-6)
    assert written.header["sform_code"] == 2  # SCANNER_ANAT


def test_write_anat_bids_accepts_unnormalized_subject_id(tmp_path, real_subjects_dir):
    source_path = discover_anat(real_subjects_dir, "H02")
    bids_path = write_anat_bids("h2", source_path, bids_root=tmp_path / "BIDS")
    assert bids_path.subject == "H02"


def test_write_anat_bids_raises_for_a_source_that_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_anat_bids("H02", tmp_path / "no-such.mgz", bids_root=tmp_path / "BIDS")
