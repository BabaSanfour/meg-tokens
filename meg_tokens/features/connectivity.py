"""Connectivity feature operations built around MNE estimators."""

from __future__ import annotations

from typing import Optional, Sequence

import mne
import numpy as np

try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    spectral_connectivity_epochs = None


def extract_roi_time_courses(stc_data, labels, src, sfreq: float = 600.0, mode: str = "mean_flip"):
    """Extract label time courses from an MNE source estimate."""
    if isinstance(stc_data, (mne.SourceEstimate, mne.VectorSourceEstimate, mne.VolSourceEstimate)):
        label_ts = mne.extract_label_time_course(stc_data, labels, src, mode=mode)
        return np.expand_dims(label_ts, axis=0)
    raise TypeError("stc_data must be an MNE source estimate")


def infer_sfreq_from_times(times: Sequence[float]) -> float:
    """Infer sample rate from a monotonic time coordinate."""
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two time samples are required to infer sfreq")
    diffs = np.diff(values)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    if finite.size == 0:
        raise ValueError("Time coordinate must be increasing to infer sfreq")
    return float(1.0 / np.median(finite))


def select_time_window(data: np.ndarray, times: Sequence[float], window: tuple[float, float]) -> np.ndarray:
    """Select a time window from ``epochs x signals x time`` data."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Connectivity input must be 3D, got shape {values.shape}")
    time_values = np.asarray(times, dtype=float)
    if time_values.shape[0] != values.shape[-1]:
        raise ValueError("time coordinate length does not match data")
    start, stop = window
    if stop <= start:
        raise ValueError("window stop must be greater than start")
    mask = (time_values >= start) & (time_values <= stop)
    if not np.any(mask):
        raise ValueError(f"Window {window} did not include any samples")
    return values[:, :, mask]


def _as_real_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(np.real_if_close(matrix))
    if np.iscomplexobj(values):
        values = np.abs(values)
    return np.asarray(values, dtype=float)


def _matrix_from_vector(vector: np.ndarray, n_signals: int) -> np.ndarray:
    values = np.asarray(vector)
    if values.size == n_signals * n_signals:
        return values.reshape(n_signals, n_signals)
    if values.size == n_signals * (n_signals - 1) // 2:
        out = np.zeros((n_signals, n_signals), dtype=values.dtype)
        tri = np.tril_indices(n_signals, k=-1)
        out[tri] = values
        return out + out.T
    raise ValueError(f"Cannot reshape {values.size} connectivity values into a {n_signals}x{n_signals} matrix")


def _dense_connectivity_data(connectivity, n_signals: int) -> np.ndarray:
    if isinstance(connectivity, tuple):
        connectivity = connectivity[0]

    if hasattr(connectivity, "get_data"):
        try:
            data = connectivity.get_data(output="dense")
        except TypeError:
            data = connectivity.get_data()
    else:
        data = connectivity

    data = np.asarray(data)
    if data.ndim == 4 and data.shape[:2] == (n_signals, n_signals):
        return np.nanmean(data, axis=-1)
    if data.ndim == 3 and data.shape[:2] == (n_signals, n_signals):
        return data
    if data.ndim == 3 and data.shape[0] in {n_signals * n_signals, n_signals * (n_signals - 1) // 2}:
        data = np.nanmean(data, axis=-1)
    if data.ndim == 2:
        bands = []
        for band_idx in range(data.shape[1]):
            bands.append(_matrix_from_vector(data[:, band_idx], n_signals))
        return np.stack(bands, axis=-1)
    raise ValueError(f"Unsupported connectivity output shape: {data.shape}")


def compute_spectral_connectivity(
    data: np.ndarray,
    method: str = "imcoh",
    sfreq: float = 600.0,
    fmin: Sequence[float] = (2, 4, 8, 15),
    fmax: Sequence[float] = (4, 8, 15, 30),
    mode: str = "fourier",
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute band-averaged spectral connectivity matrices.

    Parameters
    ----------
    data
        Array shaped ``epochs x signals x time``.

    Returns
    -------
    ndarray
        Connectivity matrices shaped ``band x signal x signal``.
    """
    if spectral_connectivity_epochs is None:
        raise ImportError(
            "mne-connectivity is required for spectral connectivity. "
            "Install the project dependencies before running connectivity analyses."
        )

    values = np.asarray(data, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Connectivity input must be 3D, got shape {values.shape}")
    if values.shape[1] < 2:
        raise ValueError("At least two signals are required for connectivity")
    if values.shape[2] < 2:
        raise ValueError("At least two time samples are required for connectivity")
    if len(fmin) != len(fmax):
        raise ValueError("fmin and fmax must have the same length")

    connectivity = spectral_connectivity_epochs(
        values,
        method=method,
        fmin=list(fmin),
        fmax=list(fmax),
        mode=mode,
        sfreq=float(sfreq),
        faverage=True,
        n_jobs=n_jobs,
    )

    dense = _dense_connectivity_data(connectivity, values.shape[1])
    if dense.shape[-1] != len(fmin):
        raise ValueError(f"Expected {len(fmin)} connectivity bands, got {dense.shape[-1]}")

    matrices = []
    for band_idx in range(len(fmin)):
        matrix = _as_real_matrix(dense[:, :, band_idx])
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    return np.stack(matrices, axis=0)
