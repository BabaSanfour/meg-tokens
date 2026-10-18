"""Source reconstruction: noise covariance, head model, forward and inverse
operators. Specification and rationale: ``docs/meg.md``.

    empty room -> trunk's temporal steps -> noise covariance     (per subject)
    BEM, source space: cortex and deep structures                 (per subject)
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
# The deep structures of the paper's mixed source space, from FreeSurfer 6's
# aseg, on a 5 mm grid: its speed-accuracy effects were found there.
VOLUME_LABELS = (
    "Left-Accumbens-area", "Left-Pallidum", "Left-Caudate", "Left-Putamen", "Left-Amygdala",
    "Left-Hippocampus", "Left-Thalamus-Proper", "Left-Cerebellum-Cortex", "Brain-Stem",
    "Right-Accumbens-area", "Right-Pallidum", "Right-Caudate", "Right-Putamen", "Right-Amygdala",
    "Right-Hippocampus", "Right-Thalamus-Proper", "Right-Cerebellum-Cortex",
)
VOLUME_SPACING_MM = 5.0
MINDIST_MM = 5.0
# A volume source has no surface to be normal to, so its orientation is free.
LOOSE = {"surface": 0.2, "volume": 1.0}
DEPTH = 0.8
# How analyses apply the operator (docs/meg.md, Decisions), and so how the
# deep structures' leakage is measured.
METHOD = "sLORETA"
LAMBDA2 = 1.0
BEAMFORMER_REG = 0.05


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
    """The subject's BEM solution and mixed source space: the cortex and the
    deep structures of :data:`VOLUME_LABELS`, within the inner skull.

    The structures form one volume source space: MNE writes a block the size
    of the MRI grid for each volume space, 67 MB, so seventeen of them would
    make every forward and inverse 1.2 GB. Which structure a source lies in is
    read back from the aseg (:func:`deep_structures`), so the volume needs no
    interpolator onto the MRI grid.

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
    bem = mne.make_bem_solution(_surfaces_to_bem(surfaces, ids, list(conductivity), ico=BEM_ICO), verbose=False)
    src = mne.setup_source_space(subject, spacing=spacing, add_dist=False, subjects_dir=subjects_dir, verbose=False)
    src += mne.setup_volume_source_space(
        subject, pos=VOLUME_SPACING_MM, mri="aseg.mgz", bem=bem, volume_label=list(VOLUME_LABELS),
        subjects_dir=subjects_dir, single_volume=True, add_interpolator=False, verbose=False,
    )
    return bem, src, record


def deep_structures(
    src: mne.SourceSpaces, subjects_dir: str | Path, *, mri_head_t: mne.transforms.Transform | None = None,
) -> dict[str, np.ndarray]:
    """For each deep structure, the positions of its sources among the volume's, by the subject's aseg.

    A forward's or inverse's source space is in head coordinates; its
    ``mri_head_t`` takes the positions back to the MRI.
    """
    import nibabel

    volume = src[-1]
    positions = volume["rr"][volume["vertno"]]
    if volume["coord_frame"] == FIFF.FIFFV_COORD_HEAD:
        positions = mne.transforms.apply_trans(mne.transforms.invert_transform(mri_head_t), positions)
    aseg = nibabel.load(Path(subjects_dir) / volume["subject_his_id"] / "mri" / "aseg.mgz")
    to_voxel = np.linalg.inv(aseg.header.get_vox2ras_tkr())
    voxels = np.round(mne.transforms.apply_trans(to_voxel, 1000 * positions)).astype(int)
    codes = np.asarray(aseg.dataobj)[tuple(voxels.T)]
    lut, _ = mne.read_freesurfer_lut()
    return {name: np.flatnonzero(codes == lut[name]) for name in VOLUME_LABELS}


def deep_volume(subject: str, subjects_dir: str | Path, bem: mne.bem.ConductorModel | None = None,
                *, interpolator: bool = False) -> mne.SourceSpaces:
    """The 17 deep structures as one volume source space, as the subject models build it.

    With ``interpolator`` it also carries the map to the MRI grid that MNE's
    volume morph needs (the stored models leave it out: it is the size of
    the MRI grid). ``bem`` limits the grid to the inner skull, as the models
    do; fsaverage's volume, the target of the morph, takes none.
    """
    return mne.setup_volume_source_space(
        subject, pos=VOLUME_SPACING_MM, mri="aseg.mgz", bem=bem, volume_label=list(VOLUME_LABELS),
        subjects_dir=subjects_dir, single_volume=True, add_interpolator=interpolator, verbose=False,
    )


def deep_morph(volume: mne.SourceSpaces, subjects_dir: str | Path) -> mne.SourceMorph:
    """The morph of a subject's deep sources onto the same structures' grid on ``fsaverage``.

    MNE's volume morph: the subject's ``brain.mgz`` registered to fsaverage's,
    affine then symmetric diffeomorphic (dipy), at the grid's 5 mm; its matrix
    computed once, so applying it is one sparse product.
    """
    morph = mne.compute_source_morph(
        volume, volume[0]["subject_his_id"], "fsaverage", subjects_dir=subjects_dir,
        src_to=deep_volume("fsaverage", subjects_dir), verbose=False,
    )
    return morph.compute_vol_morph_mat(verbose=False)


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


def _magnetometers(info: mne.Info) -> list[str]:
    return [info.ch_names[pick] for pick in mne.pick_types(info, meg="mag", ref_meg=False, exclude=[])]


def cleaned_forward(info: mne.Info, fwd: mne.Forward, operator: np.ndarray) -> mne.Forward:
    """The forward the stream's output carries, ``M G``."""
    names = _magnetometers(info)
    fwd = fwd.copy()
    rows = [fwd["sol"]["row_names"].index(name) for name in names]
    gain = fwd["sol"]["data"].copy()
    gain[rows] = operator @ gain[rows]
    fwd["sol"]["data"] = gain
    return fwd


def cleaned_noise(info: mne.Info, cov: mne.Covariance, operator: np.ndarray) -> mne.Covariance:
    """The noise covariance the stream's output carries, ``M C M^T``."""
    names = _magnetometers(info)
    order = [cov.ch_names.index(name) for name in names]
    noise = operator @ cov.data[np.ix_(order, order)] @ operator.T
    return mne.Covariance(noise, names, [], [], cov["nfree"], method=cov.get("method"))


def inverse(
    info: mne.Info,
    fwd: mne.Forward,
    cov: mne.Covariance,
    operator: np.ndarray,
    rank: int,
    *,
    loose: dict[str, float] = LOOSE,
    depth: float | None = DEPTH,
) -> mne.minimum_norm.InverseOperator:
    """Minimum-norm inverse of what the stream left: forward and noise both through its map.

    The cleaned data hold ``M G s + M n``, so the inverse is built from the
    forward ``M G`` and the noise covariance ``M C M^T``, in the ``rank``
    dimensions the stream kept.
    """
    return mne.minimum_norm.make_inverse_operator(
        info, cleaned_forward(info, fwd, operator), cleaned_noise(info, cov, operator),
        loose=loose, depth=depth, rank={"mag": rank}, verbose=False,
    )


def beamformer(
    info: mne.Info,
    fwd: mne.Forward,
    cov: mne.Covariance,
    data_cov: mne.Covariance,
    operator: np.ndarray,
    rank: int,
) -> mne.beamformer.Beamformer:
    """LCMV filters through the same map, the check a deep result is held to.

    A beamformer estimates each location on its own, from the data's own
    covariance, so cortical activity does not spread into the deep structures
    the way a minimum-norm estimate spreads it (Andersen et al., 2020). Three
    orientations per source, unit-noise-gain normalised so that depth does not
    set the scale.
    """
    return mne.beamformer.make_lcmv(
        info, cleaned_forward(info, fwd, operator), data_cov, reg=BEAMFORMER_REG,
        noise_cov=cleaned_noise(info, cov, operator), pick_ori="vector",
        weight_norm="unit-noise-gain-invariant", rank={"mag": rank}, verbose=False,
    )


def kernel(info: mne.Info, inv: mne.minimum_norm.InverseOperator) -> tuple[list[str], np.ndarray]:
    """The operator as analyses apply it: sLORETA at their λ², as a matrix on the sensors.

    Returns the sensors it reads and an array of sources x 3 x sensors, each
    source's three components in head coordinates. MNE refuses a normal
    orientation on a mixed source space, so callers take the cortex's normal
    component from it themselves; the sLORETA scaling is one factor per source,
    so each source's rows stand on their own.
    """
    names = list(inv["eigen_fields"]["col_names"])
    picks = mne.pick_types(info, meg=True, ref_meg=True, exclude=[])
    sub = mne.pick_info(info, picks)
    sub["bads"] = []
    identity = np.zeros((len(sub.ch_names), len(names)))
    identity[[sub.ch_names.index(name) for name in names], np.arange(len(names))] = 1.0
    return names, mne.minimum_norm.apply_inverse(
        mne.EvokedArray(identity, sub, verbose=False), inv, LAMBDA2, METHOD, pick_ori="vector", verbose=False,
    ).data


def deep_leakage(
    info: mne.Info,
    inv: mne.minimum_norm.InverseOperator,
    fwd: mne.Forward,
    operator: np.ndarray,
    subjects_dir: str | Path,
) -> dict[str, dict[str, float]]:
    """How far each deep structure's sLORETA estimate is its own, from the resolution matrix.

    True sources are the cortex's normal components and the deep sources'
    three, through the map the data carry (``M G``); estimates are the three
    components of every source, at the λ² analyses use. Per structure:
    ``own``, ``cortex`` and ``other_deep``, the shares of its estimate's
    cross-talk energy from each; ``found``, the share of its sources whose
    point-spread peaks inside it; ``ghosts``, the cortical sources whose
    point-spread peaks inside it.
    """
    names, estimate = kernel(info, inv)

    carried = mne.convert_forward_solution(cleaned_forward(info, fwd, operator), surf_ori=True, verbose=False)
    rows = [carried["sol"]["row_names"].index(name) for name in names]
    gain = carried["sol"]["data"][rows]
    n_cortex = carried["src"][0]["nuse"] + carried["src"][1]["nuse"]
    n_sources = estimate.shape[0]
    columns = np.r_[3 * np.arange(n_cortex) + 2, np.arange(3 * n_cortex, 3 * n_sources)]
    true_source = np.r_[np.arange(n_cortex), n_cortex + np.arange(3 * (n_sources - n_cortex)) // 3]

    structures = deep_structures(carried["src"], subjects_dir, mri_head_t=carried["mri_head_t"])
    region = np.full(n_sources, -1)
    for index, where in enumerate(structures.values()):
        region[n_cortex + where] = index
    energy = np.zeros((len(structures), len(structures) + 1))  # estimate structure x (structures..., cortex)
    peak = np.empty(columns.size, dtype=int)
    flat = estimate.reshape(-1, estimate.shape[2])
    for start in range(0, columns.size, 1000):
        block = (flat @ gain[:, columns[start : start + 1000]]).reshape(n_sources, 3, -1)
        size = (block ** 2).sum(axis=1)  # estimate source x true column
        peak[start : start + size.shape[1]] = size.argmax(axis=0)
        true_region = region[true_source[start : start + size.shape[1]]]
        for index in range(len(structures)):
            mine = size[region == index]
            for target in range(len(structures)):
                energy[index, target] += mine[:, true_region == target].sum()
            energy[index, -1] += mine[:, true_region == -1].sum()

    record = {}
    for index, name in enumerate(structures):
        total = energy[index].sum()
        deep_columns = true_source >= n_cortex
        own_columns = deep_columns & (region[true_source] == index)
        record[name] = {
            "own": round(float(energy[index, index] / total), 4),
            "cortex": round(float(energy[index, -1] / total), 4),
            "other_deep": round(float((total - energy[index, index] - energy[index, -1]) / total), 4),
            "found": round(float(np.mean(region[peak[own_columns]] == index)), 4),
            "found_in_cortex": round(float(np.mean(region[peak[own_columns]] == -1)), 4),
            "ghosts": int(np.sum(region[peak[~deep_columns]] == index)),
        }
    return record


__all__ = [
    "COVARIANCE_METHOD", "BEM_CONDUCTIVITY", "SPACING", "VOLUME_LABELS", "VOLUME_SPACING_MM", "LOOSE", "DEPTH",
    "METHOD", "LAMBDA2", "BEAMFORMER_REG",
    "noise_covariance", "head_model", "deep_structures", "forward", "cleaning_operator",
    "cleaned_forward", "cleaned_noise", "inverse", "beamformer", "kernel", "deep_leakage",
]
