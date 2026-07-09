"""Principal-component estimators for neural state-space trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class NeuralPCAResult:
    """PCA loadings and condition trajectories."""

    condition_means: np.ndarray
    loadings: np.ndarray
    variance_ratio: np.ndarray
    trajectory: np.ndarray
    fit_scores: np.ndarray
    fit_sample_condition: np.ndarray
    fit_sample_time_index: np.ndarray
    feature_mask: np.ndarray
    fit_time_mask: np.ndarray
    pca_mean: np.ndarray


def apply_neural_transform(data: np.ndarray, transform: Optional[str] = None) -> np.ndarray:
    """Apply an explicit neural-space transform before condition averaging."""
    values = np.asarray(data, dtype=float)
    if transform in (None, "none"):
        return values.copy()
    if transform == "sqrt":
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmin(finite) < 0:
            raise ValueError("sqrt transform requires non-negative data")
        return np.sqrt(values)
    if transform == "signed-sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    raise ValueError("transform must be one of: none, sqrt, signed-sqrt")


def nanmean_no_warning(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute ``nanmean`` while preserving all-NaN slices as NaN."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    totals = np.nansum(values, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = totals / counts
    return np.where(counts == 0, np.nan, mean)


def _fit_time_mask(time: Optional[Sequence[float]], n_times: int, fit_time_range: Optional[tuple[float, float]]) -> np.ndarray:
    if time is None:
        time_values = np.arange(n_times, dtype=float)
    else:
        time_values = np.asarray(time, dtype=float)
    if time_values.shape[0] != n_times:
        raise ValueError(f"time coordinate length {time_values.shape[0]} does not match n_times={n_times}")
    if fit_time_range is None:
        return np.ones(n_times, dtype=bool)
    start, stop = fit_time_range
    if stop < start:
        raise ValueError("fit_time_range stop must be greater than or equal to start")
    return (time_values >= start) & (time_values <= stop)


def fit_condition_pca(
    condition_means: np.ndarray,
    *,
    time: Optional[Sequence[float]] = None,
    n_components: int = 20,
    min_variance: Optional[float] = None,
    fit_time_range: Optional[tuple[float, float]] = None,
    project_centered: bool = False,
) -> NeuralPCAResult:
    """Fit PCA on condition-by-time observations and project trajectories.

    This mirrors the MATLAB ``nmCalcPCANoDB`` / ``nmGetPCsNoDB`` behavior used
    by the trajectory scripts: PCA loadings are estimated from centered
    condition-time samples, then condition mean time courses are projected onto
    those shared loading axes. By default projection uses the raw condition
    means, matching ``nmGetPCsNoDB``.
    """
    means = np.asarray(condition_means, dtype=float)
    if means.ndim != 3:
        raise ValueError(f"condition_means must have shape condition x feature x time, got {means.shape}")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if min_variance is not None and not (0 < min_variance <= 1):
        raise ValueError("min_variance must be in (0, 1]")

    n_conditions, n_features, n_times = means.shape
    fit_time_mask = _fit_time_mask(time, n_times, fit_time_range)
    if not np.any(fit_time_mask):
        raise ValueError("fit_time_range did not include any time samples")

    selected_times = np.where(fit_time_mask)[0]
    fit_matrix = np.transpose(means[:, :, fit_time_mask], (0, 2, 1)).reshape(-1, n_features)
    feature_mask = np.any(np.isfinite(fit_matrix), axis=0)
    if not np.any(feature_mask):
        raise ValueError("No finite features are available for PCA")

    fit_matrix = fit_matrix[:, feature_mask]
    row_mask = np.all(np.isfinite(fit_matrix), axis=1)
    if int(row_mask.sum()) < 2:
        raise ValueError("PCA requires at least two finite condition-time samples")

    max_components = min(int(n_components), int(feature_mask.sum()), int(row_mask.sum()))
    pca = PCA(n_components=max_components, svd_solver="full")
    fit_scores = pca.fit_transform(fit_matrix[row_mask])
    variance_ratio = pca.explained_variance_ratio_

    if min_variance is not None:
        keep = int(np.searchsorted(np.cumsum(variance_ratio), min_variance, side="left") + 1)
        keep = min(keep, max_components)
        fit_scores = fit_scores[:, :keep]
        variance_ratio = variance_ratio[:keep]
        components = pca.components_[:keep]
    else:
        components = pca.components_

    loadings = np.full((n_features, components.shape[0]), np.nan, dtype=float)
    loadings[feature_mask, :] = components.T

    trajectory = np.full((n_conditions, components.shape[0], n_times), np.nan, dtype=float)
    selected_means = means[:, feature_mask, :]
    for condition_idx in range(n_conditions):
        for time_idx in range(n_times):
            values = selected_means[condition_idx, :, time_idx]
            if not np.all(np.isfinite(values)):
                continue
            if project_centered:
                values = values - pca.mean_
            trajectory[condition_idx, :, time_idx] = values @ components.T

    row_condition = np.repeat(np.arange(n_conditions), len(selected_times))[row_mask]
    row_time = np.tile(selected_times, n_conditions)[row_mask]

    return NeuralPCAResult(
        condition_means=means,
        loadings=loadings,
        variance_ratio=variance_ratio,
        trajectory=trajectory,
        fit_scores=fit_scores,
        fit_sample_condition=row_condition,
        fit_sample_time_index=row_time,
        feature_mask=feature_mask,
        fit_time_mask=fit_time_mask,
        pca_mean=pca.mean_,
    )
