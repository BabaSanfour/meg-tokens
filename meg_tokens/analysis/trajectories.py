"""PCA trajectories of condition means, fitted two ways (meg.md §2).

Means are laid out subject x condition x time x source, so that the rows of a
PCA (one per condition and time, sources as features) are a reshape, not a
copy. ``grandaverage`` fits one PCA, centred, on the grand average of the
subjects' condition means and projects each subject's means without centring
them, as the preprint's code did (MATLAB ``nmGetPCsNoDB``). ``pooled`` fits one
PCA on every subject's condition means pooled at equal weight and projects
them centred, as the trajectory primer does. Both use coco-pipe's PCA.
"""

from __future__ import annotations

import numpy as np
from coco_pipe.dim_reduction import DimReduction

FITS = ("grandaverage", "pooled")


def rows(means: np.ndarray) -> np.ndarray:
    """``(..., time, source)`` as PCA rows, one per leading index and time."""
    return means.reshape(-1, means.shape[-1])


def centre_log10(log10_means: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Log10 power centred per subject, band and source: ``centre`` is the mean
    log10 power over every trial and time, subject x source, broadcast over
    conditions and times."""
    return log10_means - centre[:, None, None, :]


def align_signs(fields: np.ndarray) -> np.ndarray:
    """One sign per subject and region that makes the subjects' signed fields agree.

    A region's field is the sign-flipped mean of its sources, whose overall sign
    follows each subject's folding. Each subject's region is flipped, one
    subject at a time, while its time courses over all conditions point away
    from the sum of the other subjects'; every flip raises the summed
    agreement between subjects, so the loop ends. One sign covers all of a
    subject's conditions, which leaves condition contrasts untouched.
    ``fields`` is subject x condition x time x region; returns subject x region.
    """
    n_subjects, *_, n_regions = fields.shape
    vectors = np.moveaxis(fields, -1, 1).reshape(n_subjects, n_regions, -1)
    signs = np.ones((n_subjects, n_regions))
    total = np.einsum("sr,srk->rk", signs, vectors)
    flipped = True
    while flipped:
        flipped = False
        for subject in range(n_subjects):
            own = signs[subject][:, None] * vectors[subject]
            against = np.einsum("rk,rk->r", own, total - own) < 0
            if against.any():
                signs[subject, against] *= -1
                total[against] -= 2 * own[against]
                flipped = True
    return signs


def fit(means: np.ndarray, how: str, n_components: int) -> DimReduction:
    """A PCA of ``means`` (subject x condition x time x source): of their grand
    average (``grandaverage``) or of every subject's pooled at equal weight
    (``pooled``)."""
    if how not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {how!r}")
    # The grand average is small and may sit far from zero (raw power), so it is
    # fitted in float64; pooled means (centred log10, dB, signed fields) are
    # fitted in their own precision, float32 from the files, which halves the
    # memory of the largest fits.
    data = means.mean(axis=0, dtype=np.float64) if how == "grandaverage" else means
    # Exact, and it never forms the centred copy of the rows.
    return DimReduction("PCA", n_components=n_components, svd_solver="covariance_eigh").fit(rows(data))


def project(reducer: DimReduction, means: np.ndarray, how: str) -> np.ndarray:
    """``means`` (... x time x source) on the components, ... x time x component:
    uncentred after a ``grandaverage`` fit, centred after a ``pooled`` one."""
    scores = reducer.transform(rows(means), center=how != "grandaverage")
    return scores.reshape(*means.shape[:-1], -1)


def describe(reducer: DimReduction, means: np.ndarray) -> dict[str, np.ndarray | float]:
    """Explained variance per component, participation ratio of the retained
    components, and each subject's reconstruction error (relative to the fit's
    centre) from ``means``, subject x condition x time x source."""
    pca = reducer.reducer
    return {
        "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_),
        "participation_ratio": float(pca.participation_ratio_),
        "reconstruction_error": np.array([pca.reconstruction_error(rows(subject)) for subject in means]),
    }
