"""Source reconstruction: noise covariance, head model, forward and inverse
operators. Specification and rationale: ``docs/meg.md``.

    empty room -> trunk's temporal steps -> noise covariance     (per subject)
    BEM, source space                                             (per subject)
    forward                                                       (per run)
    stream's cleaning map, estimated -> inverse                  (per run and stream)

No source estimate is stored: an inverse is one matrix, applied to the epochs
by whichever analysis needs it.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.bem import _surfaces_to_bem
from mne.surface import _CheckInside
from mne.io.constants import FIFF
from scipy.spatial import cKDTree

from meg_tokens.meg.preprocessing import (
    HIGH_PASS,
    LINE_FREQ,
    LINE_HARMONICS,
    LOW_PASS,
    RESAMPLE_HZ,
    HEAD_COILS,
    read_staged_raw,
    repair_flux_jumps,
)

COVARIANCE_METHOD = "diagonal_fixed"
BEM_CONDUCTIVITY = (0.3, 0.006, 0.3)
BEM_ICO = 4
BEM_GAP_MM = 1.0
SPACING = "oct6"
MINDIST_MM = 5.0
LOOSE = 0.2
DEPTH = 0.8


def noise_covariance(path: str | Path, *, method: str = COVARIANCE_METHOD) -> tuple[mne.Covariance, dict]:
    """Noise covariance from an empty room, taken through the trunk's temporal steps.

    The cleaned recordings are at compensation grade 3, ZapLine'd, band-passed
    0.5-150 Hz and at 600 Hz; the empty rooms were acquired at grade 0 and
    unfiltered. A covariance of the empty room as acquired would describe other
    sensors and another band. The spatial cleaning (DSS or ICA, the tonic
    stage) has no counterpart in an empty room; the inverse passes the
    covariance through the stream's own map (:func:`cleaning_operator`).
    """
    from mne_denoise.zapline import ZapLine

    raw, grade = read_staged_raw(path)
    jump_annotations, jumps = repair_flux_jumps(raw)
    raw.set_annotations(jump_annotations)
    raw = ZapLine(sfreq=raw.info["sfreq"], line_freq=LINE_FREQ, n_harmonics=LINE_HARMONICS).fit_transform(raw)
    raw.filter(HIGH_PASS, LOW_PASS, picks="mag", fir_design="firwin", verbose=False)
    raw.resample(RESAMPLE_HZ, verbose=False)
    cov = mne.compute_raw_covariance(raw, picks="mag", method=method, verbose=False)
    return cov, {
        "emptyroom": Path(path).name,
        "compensation_grade_source": grade,
        "flux_jumps": len(jumps),
        "seconds": float(raw.times[-1]),
        "method": method,
    }


def _inside(surface: dict, points: np.ndarray) -> np.ndarray:
    """Which points lie inside a closed surface, by MNE's own containment test."""
    closed = mne.surface.complete_surface_info({"rr": surface["rr"] / 1000, "tris": surface["tris"]}, copy=False, verbose=False)
    return _CheckInside(closed)(points / 1000)


def _clear_of(outer: dict, inner: dict, gap_mm: float) -> tuple[int, float]:
    """Push ``outer`` out along its normals until ``inner`` lies ``gap_mm`` inside it.

    Only the outer vertices within 4 mm of an inner vertex that is outside or
    too close move; the rest of ``outer`` is left as it is. In place.
    """
    original = outer["rr"].copy()
    for _ in range(50):
        tree = cKDTree(outer["rr"])
        distance, _ = tree.query(inner["rr"])
        signed = np.where(_inside(outer, inner["rr"]), distance, -distance)
        close = signed < gap_mm
        if not close.any():
            moved = np.linalg.norm(outer["rr"] - original, axis=1)
            return int((moved > 0).sum()), float(moved.max())
        normals = mne.surface.complete_surface_info({"rr": outer["rr"], "tris": outer["tris"]}, copy=False, verbose=False)["nn"]
        shift = np.zeros(len(outer["rr"]))
        for point, needed in zip(inner["rr"][close], gap_mm - signed[close]):
            near = tree.query_ball_point(point, r=4.0)
            shift[near] = np.maximum(shift[near], needed + 0.1)
        outer["rr"] = outer["rr"] + normals * shift[:, np.newaxis]
    raise ValueError("the outer surface could not be cleared of the inner one")


def head_model(
    subject: str,
    subjects_dir: str | Path,
    *,
    conductivity: tuple[float, ...] = BEM_CONDUCTIVITY,
    spacing: str = SPACING,
) -> tuple[mne.bem.ConductorModel, mne.SourceSpaces, dict]:
    """The subject's BEM solution and cortical source space.

    The watershed surfaces of five subjects (H05, H16, H24, H25, H28) let the
    inner skull cross the outer skull by up to 2.5 mm, which no BEM accepts.
    The inner skull is the surface MEG depends on, so it is kept as it is and
    the outer skull is moved out locally to clear it by :data:`BEM_GAP_MM`,
    and the scalp in turn to clear the outer skull (H05, H28).
    """
    bem_dir = Path(subjects_dir) / subject / "bem"
    surfaces = [
        mne.read_surface(bem_dir / f"{name}.surf", return_dict=True)[2]
        for name in ("inner_skull", "outer_skull", "outer_skin")[: len(conductivity)]
    ]
    record = {"conductivity": list(conductivity)}
    if len(surfaces) == 3:
        inner, outer, skin = surfaces
        for name, surface, within in (("outer_skull", outer, inner), ("outer_skin", skin, outer)):
            moved, shift = _clear_of(surface, within, BEM_GAP_MM)
            record.update({f"{name}_moved": moved, f"{name}_moved_max_mm": round(shift, 2)})
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN, FIFF.FIFFV_BEM_SURF_ID_SKULL, FIFF.FIFFV_BEM_SURF_ID_HEAD][: len(surfaces)]
    model = _surfaces_to_bem(surfaces, ids, list(conductivity), ico=BEM_ICO)
    src = mne.setup_source_space(subject, spacing=spacing, add_dist=False, subjects_dir=subjects_dir, verbose=False)
    return mne.make_bem_solution(model, verbose=False), src, record


def forward(info: mne.Info, trans: str | Path, src: mne.SourceSpaces, bem: mne.bem.ConductorModel) -> mne.Forward:
    """The run's forward model: magnetometers at the run's compensation grade, from its own transform."""
    return mne.make_forward_solution(info, trans=str(trans), src=src, bem=bem, meg=True, eeg=False, mindist=MINDIST_MM, verbose=False)


def cleaning_operator(cleaned: mne.io.BaseRaw, artifact: mne.io.BaseRaw, rank: int) -> tuple[np.ndarray, dict]:
    """The linear map the stream applied to the magnetometers, estimated from its own output.

    DSS, ICA and the tonic stages subtract components through their own
    patterns and filters: the cleaned magnetometers are ``M`` times the trunk's,
    an oblique map of rank ``rank``. The trunk is the cleaned recording plus
    the artifact estimate, so ``M`` is the least-squares fit of the one on the
    other, with the head-coil channels as further regressors for the
    head-movement step. ASR acts window by window and is not linear; what it
    leaves unexplained is the fit's residual.
    """
    picks = mne.pick_types(cleaned.info, meg="mag", ref_meg=False, exclude=[])
    names = [cleaned.ch_names[pick] for pick in picks]
    output = cleaned.get_data(picks=names)
    trunk = output + artifact.get_data(picks=names)
    coils = cleaned.get_data(picks=[name for name in cleaned.ch_names if name.startswith(HEAD_COILS)])
    regressors = np.vstack([trunk, coils])
    regressors -= regressors.mean(axis=1, keepdims=True)
    output = output - output.mean(axis=1, keepdims=True)
    scale = regressors.std(axis=1)
    varying = scale > 0
    coefficients, *_ = np.linalg.lstsq((regressors[varying] / scale[varying, np.newaxis]).T, output.T, rcond=None)
    residual = np.sum((output - coefficients.T @ (regressors[varying] / scale[varying, np.newaxis])) ** 2) / np.sum(output ** 2)
    operator = (coefficients.T / scale[varying])[:, : len(names)]
    singular = np.linalg.svd(operator, compute_uv=False)
    return operator, {
        "rank": rank,
        "operator_residual": float(residual),
        # How sharply the fitted map falls to zero beyond the recorded rank.
        "operator_rank_gap": float(singular[rank - 1] / singular[rank]) if rank < len(names) else None,
    }


def inverse(
    info: mne.Info,
    fwd: mne.Forward,
    cov: mne.Covariance,
    operator: np.ndarray,
    rank: int,
    *,
    loose: float = LOOSE,
    depth: float | None = DEPTH,
) -> mne.minimum_norm.InverseOperator:
    """Minimum-norm inverse of what the stream left: forward and noise both through its map.

    The cleaned data hold ``M G s + M n``, so the inverse is built from the
    forward ``M G`` and the noise covariance ``M C M^T``, in the ``rank``
    dimensions the stream kept.
    """
    names = [info.ch_names[pick] for pick in mne.pick_types(info, meg="mag", ref_meg=False, exclude=[])]
    fwd = fwd.copy()
    rows = [fwd["sol"]["row_names"].index(name) for name in names]
    gain = fwd["sol"]["data"].copy()
    gain[rows] = operator @ gain[rows]
    fwd["sol"]["data"] = gain
    order = [cov.ch_names.index(name) for name in names]
    noise = operator @ cov.data[np.ix_(order, order)] @ operator.T
    cov = mne.Covariance(noise, names, [], [], cov["nfree"], method=cov.get("method"))
    return mne.minimum_norm.make_inverse_operator(
        info, fwd, cov, loose=loose, depth=depth, rank={"mag": rank}, verbose=False,
    )


__all__ = [
    "COVARIANCE_METHOD", "BEM_CONDUCTIVITY", "SPACING", "LOOSE", "DEPTH",
    "noise_covariance", "head_model", "forward", "cleaning_operator", "inverse",
]
