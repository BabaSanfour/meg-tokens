"""Phase-amplitude coupling estimators."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def select_time_window(data: np.ndarray, times: Sequence[float], window: tuple[float, float] | None) -> np.ndarray:
    """Select a time interval from data with time on the last axis."""
    values = np.asarray(data)
    if window is None:
        return values
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1 or time_values.size != values.shape[-1]:
        raise ValueError("time coordinate length does not match data")
    start, stop = window
    if stop <= start:
        raise ValueError("window stop must be greater than start")
    mask = (time_values >= start) & (time_values <= stop)
    if not np.any(mask):
        raise ValueError(f"Window {window} did not include any samples")
    return values[..., mask]


def modulation_index(
    phase: np.ndarray,
    amplitude: np.ndarray,
    *,
    n_bins: int = 18,
) -> np.ndarray:
    """Compute Tort-style phase-amplitude modulation index per feature.

    Parameters
    ----------
    phase
        Low-frequency phase array shaped ``trial x feature x time``.
    amplitude
        High-frequency amplitude array with the same shape.
    n_bins
        Number of phase bins spanning ``[-pi, pi]``.

    Returns
    -------
    ndarray
        Coupling values shaped ``feature``.
    """
    phase_values = np.asarray(phase, dtype=float)
    amp_values = np.asarray(amplitude, dtype=float)
    if phase_values.shape != amp_values.shape:
        raise ValueError(f"phase and amplitude shapes differ: {phase_values.shape} != {amp_values.shape}")
    if phase_values.ndim != 3:
        raise ValueError(f"PAC inputs must be trial x feature x time, got {phase_values.shape}")
    if n_bins < 4:
        raise ValueError("n_bins must be at least 4")

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    out = np.full(phase_values.shape[1], np.nan, dtype=float)
    for feature_idx in range(phase_values.shape[1]):
        ph = phase_values[:, feature_idx, :].reshape(-1)
        amp = amp_values[:, feature_idx, :].reshape(-1)
        valid = np.isfinite(ph) & np.isfinite(amp) & (amp >= 0)
        if not np.any(valid):
            continue
        ph = ((ph[valid] + np.pi) % (2.0 * np.pi)) - np.pi
        amp = amp[valid]

        means = np.zeros(n_bins, dtype=float)
        for bin_idx in range(n_bins):
            if bin_idx == n_bins - 1:
                in_bin = (ph >= bins[bin_idx]) & (ph <= bins[bin_idx + 1])
            else:
                in_bin = (ph >= bins[bin_idx]) & (ph < bins[bin_idx + 1])
            if np.any(in_bin):
                means[bin_idx] = float(np.mean(amp[in_bin]))

        total = float(np.sum(means))
        if total <= 0:
            continue
        probs = means / total
        positive = probs > 0
        entropy = -float(np.sum(probs[positive] * np.log(probs[positive])))
        out[feature_idx] = (np.log(n_bins) - entropy) / np.log(n_bins)
    return out
