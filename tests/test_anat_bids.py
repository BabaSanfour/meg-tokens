"""BIDS-anat raw layer, tested against the real FreeSurfer reconstructions.

Reads ``subjects_dir`` from this checkout's ``tokens.toml`` (see
``tests/conftest.py``) and skips where it is absent.
"""

from meg_tokens.meg.anat_bids import discover_anat, write_anat_bids


def test_write_anat_bids_copies_the_native_volume_losslessly(tmp_path, subjects_dir):
    import nibabel as nib
    import numpy as np

    source_path = discover_anat(subjects_dir, "H02")
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
