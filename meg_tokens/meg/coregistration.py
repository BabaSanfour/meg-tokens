"""MEG-MRI coregistration from CTF Polhemus head shapes.

Three coordinate frames meet here, and keeping them apart is the whole job.

*MRI* is the FreeSurfer surface frame. *Head* is the frame MNE derives from
the head-coil positions a recording localized at its start, and it is the frame
``info["dev_head_t"]`` maps the sensors into -- so it is the frame a ``trans``
must target. *Digitizer* is the frame of the Polhemus session that traced the
dense head shape, defined by the nasion and pre-auricular points recorded in
that same file.

Head and digitizer are both "CTF head coordinates" and are *not* the same: the
two sessions localize the same three landmarks minutes apart and disagree by a
few millimetres (2-8 mm on this dataset). The bridge between them is the three
shared landmarks, so the head shape is carried into a recording's head frame by
the rigid transform that matches its own fiducials onto that recording's, and
the residual of that match is reported as a digitization-quality number.

This reads the native CTF ``.eeg`` head-shape export that Stage 0 stages
verbatim, landmarks included: they are the only record of which frame the dense
points are expressed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import mne
import numpy as np
from mne.coreg import Coregistration


#: CTF head-shape landmark codes and the MNE fiducial names they carry.
CTF_FIDUCIAL_CODES = {"NZ": "nasion", "OG": "lpa", "OD": "rpa"}

#: Fiducial order used for every point-correspondence array in this module.
FIDUCIAL_ORDER = ("lpa", "nasion", "rpa")

#: MNE dig identifiers for the cardinal landmarks.
_CARDINAL_IDENTS = {1: "lpa", 2: "nasion", 3: "rpa"}


def _ordered(fiducials: Mapping[str, np.ndarray]) -> np.ndarray:
    """Three landmarks as a ``(3, 3)`` array in :data:`FIDUCIAL_ORDER`."""
    return np.array([fiducials[name] for name in FIDUCIAL_ORDER], dtype=float)


@dataclass(frozen=True)
class HeadShape:
    """A dense Polhemus head shape in its own digitizer frame, in metres."""

    fiducials: Mapping[str, np.ndarray]
    points: np.ndarray
    source: Path


@dataclass(frozen=True)
class CoregistrationResult:
    """One fitted MEG-MRI transform and the numbers that justify trusting it."""

    fs_subject: str
    trans: mne.transforms.Transform
    fiducial_source: str
    fiducial_residual_mm: np.ndarray
    dig_mri_distances_mm: np.ndarray
    n_points: int
    n_omitted: int
    scale: Optional[np.ndarray] = None

    @property
    def quality(self) -> dict[str, float]:
        """Scalar QC summary, in millimetres."""
        distances = self.dig_mri_distances_mm
        return {
            "fiducial_residual_max_mm": float(self.fiducial_residual_mm.max()),
            "dig_mri_median_mm": float(np.median(distances)),
            "dig_mri_mean_mm": float(distances.mean()),
            "dig_mri_p95_mm": float(np.percentile(distances, 95)),
            "dig_mri_max_mm": float(distances.max()),
            "points_used": float(self.n_points - self.n_omitted),
            "points_omitted": float(self.n_omitted),
        }


def _ctf_als_cm_to_head_ras_m(points: np.ndarray) -> np.ndarray:
    """Convert CTF ALS centimetres to the MNE head RAS metres MNE expects."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.column_stack([-points[:, 1], points[:, 0], points[:, 2]]) / 100.0


def read_ctf_headshape(path: str | Path) -> HeadShape:
    """Read a CTF Polhemus ``.eeg`` head-shape export.

    Landmark lines are ``CODE x y z`` and are traced several times per session,
    so each landmark is averaged over its repeats; every other line is a bare
    ``x y z`` triple.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Head shape file does not exist: {source}")

    repeats: dict[str, list[list[float]]] = {code: [] for code in CTF_FIDUCIAL_CODES}
    points: list[list[float]] = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in repeats:
            repeats[fields[0]].append([float(value) for value in fields[1:4]])
        else:
            points.append([float(value) for value in fields[:3]])

    missing = sorted(code for code, values in repeats.items() if not values)
    if missing:
        raise ValueError(f"Head shape {source} is missing landmarks: {missing}")
    if not points:
        raise ValueError(f"Head shape {source} contains no head-shape points")

    fiducials = {
        CTF_FIDUCIAL_CODES[code]: _ctf_als_cm_to_head_ras_m(np.mean(values, axis=0))[0]
        for code, values in repeats.items()
    }
    return HeadShape(
        fiducials=fiducials,
        points=_ctf_als_cm_to_head_ras_m(np.asarray(points, dtype=float)),
        source=source,
    )


def head_fiducials(info: mne.Info) -> dict[str, np.ndarray]:
    """The cardinal landmarks a recording localized, in its head frame."""
    found = {
        _CARDINAL_IDENTS[point["ident"]]: np.asarray(point["r"], dtype=float)
        for point in info["dig"] or ()
        if point["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL
        and point["ident"] in _CARDINAL_IDENTS
    }
    missing = sorted(set(FIDUCIAL_ORDER) - set(found))
    if missing:
        raise ValueError(f"Recording digitization is missing landmarks: {missing}")
    return found


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation and translation carrying ``source`` points onto ``target``.

    Kabsch, with the reflection case suppressed so the result stays a proper
    rotation. Returns a ``(4, 4)`` homogeneous matrix.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(
            f"Point sets must correspond: got {source.shape} and {target.shape}"
        )

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    parity = np.sign(np.linalg.det(right.T @ left.T))
    rotation = right.T @ np.diag([1.0, 1.0, parity]) @ left.T

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def apply_rigid(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous rigid transform to an ``(n, 3)`` point array."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ transform[:3, :3].T + transform[:3, 3]


def _match_landmarks(
    source: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """The rigid map carrying one landmark set onto another, and its residual.

    The residual is what no rigid map can reconcile, per landmark, in
    millimetres and in :data:`FIDUCIAL_ORDER`.
    """
    source, target = _ordered(source), _ordered(target)
    transform = rigid_transform(source, target)
    residual_mm = np.linalg.norm(apply_rigid(transform, source) - target, axis=1) * 1000
    return transform, residual_mm


def headshape_in_head_frame(
    headshape: HeadShape,
    info: mne.Info,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry a head shape into a recording's head frame via the landmarks.

    Returns the transformed points and the residual, which here measures
    digitization quality: it is how far the two sessions disagree about the
    same three physical points, and so bounds how much of the coregistration
    error is already spent before ICP begins.
    """
    transform, residual_mm = _match_landmarks(headshape.fiducials, head_fiducials(info))
    return apply_rigid(transform, headshape.points), residual_mm


def info_with_headshape(info: mne.Info, points: np.ndarray) -> mne.Info:
    """A copy of ``info`` whose digitization carries the dense head shape.

    The recording's own landmarks are kept; only the extra points are added.
    """
    recorded = head_fiducials(info)
    montage = mne.channels.make_dig_montage(
        nasion=recorded["nasion"],
        lpa=recorded["lpa"],
        rpa=recorded["rpa"],
        hsp=np.asarray(points, dtype=float),
        coord_frame="head",
    )
    info = info.copy()
    with info._unlock():
        info["dig"] = montage.dig
    return info


def head_to_mri(trans: mne.transforms.Transform) -> mne.transforms.Transform:
    """Normalize a MEG-MRI transform to the head-to-MRI direction.

    Both directions are in circulation -- ``Coregistration.trans`` and the
    ``-trans.fif`` convention are head-to-MRI, while several MNE functions want
    the inverse -- and applying one where the other is meant produces a plausible
    number rather than an error. Normalizing at the boundary keeps the direction
    out of the arithmetic.
    """
    if trans["from"] == mne.io.constants.FIFF.FIFFV_COORD_MRI:
        return mne.transforms.invert_transform(trans)
    return trans


def scalp_surface(fs_subject: str, subjects_dir: str | Path) -> np.ndarray:
    """An anatomy's scalp vertices in MRI coordinates, in metres.

    Densest available: FreeSurfer's ``head-dense`` if it was built, else the
    decimated ``head`` or the BEM scalp.
    """
    surface = mne.get_head_surf(
        fs_subject,
        ("head-dense", "head", "bem"),
        subjects_dir=subjects_dir,
        verbose=False,
    )
    return surface["rr"]


def scalp_distances(
    points: np.ndarray,
    trans: mne.transforms.Transform,
    scalp: np.ndarray,
) -> np.ndarray:
    """Distance from each head-frame point to the scalp surface, in metres.

    Nearest-vertex, as MNE itself measures it, over *every* point.
    ``mne.dig_mri_distances`` is the obvious call and the wrong one here: it
    routes through ``get_fitting_dig``, which silently drops the low points a
    sphere fit should not see. Right for fitting a sphere, wrong for reporting
    coverage -- the dropped points are real digitization, and whether they land
    on the scalp is the question.
    """
    from scipy.spatial import cKDTree

    in_mri = mne.transforms.apply_trans(head_to_mri(trans), np.atleast_2d(points))
    return cKDTree(scalp).query(in_mri)[0]


def has_own_anatomy(subjects_dir: str | Path, fs_subject: str) -> bool:
    """Whether a real FreeSurfer reconstruction exists to coregister against.

    A scaled template does not count, and ``mne.scale_mri``'s leftover
    ``MRI scaling parameters.cfg`` is the only thing separating the two on disk
    -- a scaled subject has a ``bem/`` and a ``surf/`` like any other. Without
    that check a borrowed anatomy would be relabelled as reconstructed the
    moment it was written.
    """
    anatomy = Path(subjects_dir) / fs_subject
    return (anatomy / "bem").is_dir() and not (
        anatomy / "MRI scaling parameters.cfg"
    ).is_file()


def reference_session(infos: Sequence[mne.Info]) -> int:
    """Index of the session whose head frame is most typical of the subject.

    The fit is carried to every other session from this one, so the session
    minimizing total landmark distance to the rest spreads the residual drift
    evenly instead of anchoring it to whichever recording came first.
    """
    landmarks = np.array([_ordered(head_fiducials(info)) for info in infos])
    spread = np.linalg.norm(landmarks[:, None] - landmarks[None, :], axis=-1).sum((1, 2))
    return int(np.argmin(spread))


def session_alignment(
    reference_info: mne.Info,
    session_info: mne.Info,
) -> tuple[np.ndarray, np.ndarray]:
    """The rigid map from one recording's head frame into another's.

    Head coordinates are *defined* by the head coils, and the coils are placed
    afresh for each recording, so a subject's ten sessions do not all share one
    head frame. Most agree to well under a millimetre; some do not.

    Because both frames are anchored to the same three physical landmarks, the
    map between them is closed-form -- which is what makes fitting ICP once per
    subject and correcting each session exactly the cheap option, rather than
    re-running ICP ten times or absorbing the drift as error.

    Returns the transform and its per-landmark residual in millimetres.
    """
    return _match_landmarks(
        head_fiducials(reference_info), head_fiducials(session_info)
    )


def compose_trans(
    trans: mne.transforms.Transform,
    alignment: np.ndarray,
) -> mne.transforms.Transform:
    """Carry a head-to-MRI transform into another recording's head frame.

    ``alignment`` maps the reference head frame onto the session's, so reaching
    the MRI from the session means undoing it first.
    """
    trans = head_to_mri(trans)
    return mne.transforms.Transform(
        "head", "mri", trans["trans"] @ np.linalg.inv(alignment)
    )


def coregister(
    info: mne.Info,
    fs_subject: str,
    subjects_dir: str | Path,
    *,
    headshape: Optional[HeadShape] = None,
    coarse_iterations: int = 6,
    fine_iterations: int = 20,
    omit_distance: float = 0.005,
    scale_mode: Optional[str] = None,
) -> CoregistrationResult:
    """Fit one MEG-MRI transform for a recording.

    With a head shape, this is the standard MNE sequence: fit the landmarks,
    run a coarse ICP, drop points further than ``omit_distance`` from the scalp
    (they are face and neck traces the FreeSurfer surface does not model), then
    run a fine ICP weighted toward the nasion.

    Without a head shape there is nothing for ICP to work on, so the fit stops
    at the landmarks and the result is markedly less accurate -- it is reported
    with ``n_points`` of zero so downstream QC can see that.

    ``fs_subject`` names a FreeSurfer anatomy, not a cohort subject: the two
    coincide for the 30 reconstructed subjects, but H07 and H10 have no anatomy
    of their own and borrow ``fsaverage``. Mapping one to the other belongs to
    the caller.

    ``scale_mode`` (``"uniform"`` or ``"3-axis"``) additionally fits the
    anatomy's size to the head shape, which only makes sense for such a
    borrowed template. The returned ``trans`` then targets an anatomy scaled by
    ``scale``, which the caller must materialize with ``mne.scale_mri`` before
    the transform means anything.
    """
    subjects_dir = Path(subjects_dir)

    residual_mm = np.zeros(len(FIDUCIAL_ORDER))
    if headshape is not None:
        points, residual_mm = headshape_in_head_frame(headshape, info)
        info = info_with_headshape(info, points)
        n_points = len(points)
    else:
        n_points = 0

    fit = Coregistration(info, fs_subject, subjects_dir, fiducials="auto")
    if scale_mode is not None:
        fit.set_scale_mode(scale_mode)
    fit.fit_fiducials(verbose=False)
    if n_points:
        fit.fit_icp(n_iterations=coarse_iterations, nasion_weight=2.0, verbose=False)
        fit.omit_head_shape_points(distance=omit_distance)
        fit.fit_icp(n_iterations=fine_iterations, nasion_weight=10.0, verbose=False)

    n_fitted = 0
    if n_points:
        # `compute_dig_mri_distances` scores the three fiducials alongside the
        # head shape, so the surviving points are three fewer than it returns.
        n_fitted = len(fit.compute_dig_mri_distances()) - len(FIDUCIAL_ORDER)

    # Score every digitized point, not just the survivors: those were selected
    # for being within ``omit_distance``, so their median is near-constant
    # however good the fit was. A scaled fit is the exception -- its anatomy
    # exists only inside the fit object, so the surface on disk is a different
    # head, and the caller re-fits against the materialized one.
    distances_mm = 1000 * (
        scalp_distances(points, fit.trans, scalp_surface(fs_subject, subjects_dir))
        if n_points and scale_mode is None
        else fit.compute_dig_mri_distances()
    )

    # `fiducials="auto"` silently estimates the MRI landmarks from `fsaverage`
    # when no `-fiducials.fif` was ever picked, and an estimated set is markedly
    # less accurate. Which one was used separates a reported number from a
    # trusted one.
    picked = subjects_dir / fs_subject / "bem" / f"{fs_subject}-fiducials.fif"
    return CoregistrationResult(
        fs_subject=fs_subject,
        trans=fit.trans,
        fiducial_source="file" if picked.is_file() else "estimated",
        fiducial_residual_mm=residual_mm,
        dig_mri_distances_mm=distances_mm,
        n_points=n_points,
        n_omitted=max(n_points - n_fitted, 0),
        scale=np.asarray(fit.scale, dtype=float) if scale_mode else None,
    )
