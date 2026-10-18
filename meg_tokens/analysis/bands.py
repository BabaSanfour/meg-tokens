"""Frequency bands read off spectral peaks, as the preprint chose its own (its Fig S1), and the checks of it.

The preprint fitted specparam (then FOOOF) to every subject's sensor spectra
and kept the bands where the histogram of peak centre frequencies rose above a
line it did not define. Here the line is the histogram's mean bin and every
subject weighs the same (each subject's peaks as shares of its own); the
edges' intervals come from resampling subjects. Neither MNE nor specparam reads
bands off a histogram, so the thresholding, the shares and the resampling are
written here; the spectra are MNE's and the model specparam's.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from specparam.sim.gen import gen_aperiodic

# FOOOF's defaults, which the preprint ran with: peaks 0.5-12 Hz wide, as many as found, any height above 2 SD.
FOOOF_SETTINGS = dict(peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0, peak_threshold=2.0)


def periodic_power(log_power: np.ndarray, aperiodic: np.ndarray, freqs: np.ndarray, mode: str) -> np.ndarray:
    """Log10 power above each spectrum's aperiodic fit: ``log_power`` spectra x frequencies, ``aperiodic``
    the fit's parameters per spectrum (specparam's order for ``mode``)."""
    return log_power - np.array([gen_aperiodic(freqs, mode, params) for params in aperiodic])


def peak_shares(centre_frequencies: np.ndarray, owners: np.ndarray, n_subjects: int, edges: np.ndarray) -> np.ndarray:
    """Each subject's peaks per bin as a percentage of its own peaks: subjects x bins."""
    shares = np.zeros((n_subjects, edges.size - 1))
    for subject in range(n_subjects):
        mine = centre_frequencies[owners == subject]
        if mine.size:
            shares[subject] = np.histogram(mine, edges)[0] / mine.size * 100
    return shares


def bands_above(values: np.ndarray, edges: np.ndarray, line: float) -> list[tuple[float, float]]:
    """The frequency ranges where a histogram (``values`` per bin between ``edges``) is above ``line``."""
    above = np.diff(np.r_[False, values > line, False].astype(int))
    return [(float(edges[a]), float(edges[b])) for a, b in zip(np.flatnonzero(above == 1), np.flatnonzero(above == -1))]


def band_at(found: list[tuple[float, float]], band: tuple[float, float]) -> tuple[float, float] | None:
    """The range of ``found`` holding ``band``'s centre, or ``None``."""
    centre = (band[0] + band[1]) / 2
    return next(((lo, hi) for lo, hi in found if lo <= centre < hi), None)


def resampled_bands(
    shares: np.ndarray, edges: np.ndarray, bands: Mapping[str, tuple[float, float]], resamples: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Subjects resampled with replacement (``resamples``: resamples x subjects, indices): the mean shares
    of each resample (resamples x bins) and, per band, the edges of the range above that resample's mean bin
    holding the band's centre (resamples x 2, NaN where none does)."""
    means = np.stack([shares[pick].mean(axis=0) for pick in resamples])
    found = {name: np.full((len(resamples), 2), np.nan) for name in bands}
    for i, mean in enumerate(means):
        above = bands_above(mean, edges, mean.mean())
        for name, band in bands.items():
            match = band_at(above, band)
            if match is not None:
                found[name][i] = match
    return means, found


def line_sweep(shares: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Which bins of the mean shares lie above each line, ``lines`` in multiples of the mean bin: lines x bins."""
    return shares[None, :] > lines[:, None] * shares.mean()


__all__ = [
    "FOOOF_SETTINGS", "periodic_power", "peak_shares", "bands_above", "band_at", "resampled_bands", "line_sweep",
]
