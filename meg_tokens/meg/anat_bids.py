"""BIDS-anat raw layer writer.

Copies each subject's raw T1 (``mri/rawavg.mgz`` -- native per-subject voxel
geometry and intensity range, unlike ``mri/T1.mgz``, which every subject
shares as a uniformly resampled/intensity-normalized 256x256x256 8-bit
conformed volume) into ``BIDS/sub-*/anat`` via ``mne_bids.write_anat``, a
real copy/format-conversion (``.mgz`` -> ``.nii.gz``).
``ProjectConfig.subjects_dir`` itself is untouched and stays the direct input
to ``meg_tokens.meg.sources``' BEM/source-space stages (which use the
conformed volumes, e.g. ``T1.mgz``) -- this BIDS layer is a documented,
portable export of the raw scan, not a replacement for it. Nothing here
reads or writes a coregistration: the head-to-MRI transform this project
uses lives in a separately-managed ``-trans.fif``, which is why
``write_anat`` is called without ``landmarks`` (no fiducial sidecar) and
without ``deface`` (defacing would alter the very voxels this layer exists
to preserve verbatim).

Staging is planned like everything else in Stage 0: ``discover_anat``
supplies each subject's ``anat`` manifest row (via
``meg_tokens.meg.raw_staging.match_subject_assets``), and
``meg_tokens.workflows.raw_staging`` copies the reviewed rows -- so a
missing or hand-corrected T1 shows up in the same manifest as every other
Stage 0 gap rather than in a separate command's own reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mne
from mne_bids import BIDSPath, write_anat

from meg_tokens.core import normalize_subject_id

mne.set_log_level("ERROR")

# Where FreeSurfer keeps the volume this layer exports, relative to a
# subject's reconstruction directory.
RAW_T1_RELATIVE_PATH = Path("mri") / "rawavg.mgz"


def discover_anat(subjects_dir: Optional[str | Path], subject: str) -> Optional[Path]:
    """The subject's raw T1 under ``subjects_dir``, or ``None`` if absent.

    Absent is a real, expected outcome -- a subject either has a FreeSurfer
    reconstruction or doesn't (30 of this dataset's 32 subjects do; H07 and
    H10 have none) -- so this reports rather than raises, the same way
    ``discover_noise_session`` and ``discover_headshape`` do: two known
    gaps must not stop the other 30 subjects' plan from being written.
    ``None`` for ``subjects_dir`` (unconfigured) is likewise reported, not
    an error.
    """
    if subjects_dir is None:
        return None
    subject = normalize_subject_id(subject)
    candidate = Path(subjects_dir) / subject / RAW_T1_RELATIVE_PATH
    return candidate if candidate.is_file() else None


def write_anat_bids(
    subject: str,
    t1_path: str | Path,
    *,
    bids_root: str | Path,
    overwrite: bool = False,
) -> BIDSPath:
    """Copy one subject's raw T1 into the BIDS-anat raw layer.

    Takes the source volume explicitly, rather than re-deriving it from
    ``subjects_dir``, so that -- like every other staged file -- what gets
    copied is whatever the manifest names, including a hand-edited
    ``source_path``.

    ``mne_bids.write_anat`` converts ``.mgz`` to ``.nii.gz`` via nibabel --
    a format change, not a resampling. Verified on this dataset's real
    volumes: identical shape, bit-identical voxel data, and an identical
    affine; the only differences are the dtype's byte order (normalised) and
    the NIfTI header's frame codes, where ``sform_code`` becomes 2
    (``SCANNER_ANAT``) and ``qform_code`` stays 0 -- the correct claim for a
    volume still in its own scanner space, which ``rawavg.mgz`` is. That is
    what makes this layer safe to treat as the raw scan rather than a
    derived one.

    One T1w exists per subject (no ``acq``/``run`` entities on this path), so
    re-running with ``overwrite=True`` can only replace a file with the same
    file -- unlike the MEG layer, there is no shared sidecar for a second
    entity's write to contend over.
    """
    subject = normalize_subject_id(subject)
    t1_path = Path(t1_path)
    if not t1_path.is_file():
        raise FileNotFoundError(f"No raw T1 for {subject}: {t1_path}")
    bids_path = BIDSPath(subject=subject, datatype="anat", suffix="T1w", root=Path(bids_root))
    write_anat(t1_path, bids_path=bids_path, overwrite=overwrite)
    return bids_path
